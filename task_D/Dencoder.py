"""
d_encoder.py - train and evaluate the attention-based style encoder (Module D, step 1)

What it learns
    Input : the sentence embeddings (CISR, 768-d) of ONE piece of writing by a user
    Model : projection -> 1 Transformer self-attention layer -> learned-query attention pooling -> style vector
    Loss  : contrastive (InfoNCE). Positive pair = two DIFFERENT-TOPIC texts by the same author (control doc X
            vs control doc Y); negatives = other authors in the batch. So the vector is pushed to encode
            "who writes like this", not "what is it about".

Evaluation (participant-disjoint, cross-topic)
    query = doc X of author i, gallery = doc Y of all held-out authors.
    top-1 accuracy, MRR, verification AUROC; compared for  mean-pool  vs  parameter-free attention  vs  trained.

Usage
    python d_encoder.py --data logs --out results_d                        # train + evaluate (needs torch)
    python d_encoder.py --data logs --out results_d --skip-train           # only evaluate mean vs attention
    python d_encoder.py --data logs --out results_d --sent-embedder mock --skip-train   # test without models
Then:  python module_d.py --encoder trained --encoder-ckpt results_d/style_encoder.pt ...
"""
import argparse, json, os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ModuleD import (
    load_participants,
    make_split,
    make_sentence_embedder,
    split_sentences,
    mean_pool,
    attention_pool,
    l2,
    SEED
)

# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
def doc_sets(data, pids, emb):
    """pid -> [sentence-embedding matrix of control doc 0, of control doc 1]"""
    out = {}
    for pid in pids:
        c = data[pid]["control"]
        if len(c) >= 2:
            out[pid] = [emb.embed(split_sentences(c[0]["text"])), emb.embed(split_sentences(c[1]["text"]))]
    return out

# ------------------------------------------------------------------
# RETRIEVAL / VERIFICATION METRICS
# ------------------------------------------------------------------
def retrieval_metrics(VA, VB):
    """Row i of VA and row i of VB are the same author (different topics)."""
    sim = l2(VA) @ l2(VB).T
    n = len(sim)
    res = {}
    for name, s in (("A->B", sim), ("B->A", sim.T)):
        ranks = np.array([(s[i] > s[i, i]).sum() + 1 for i in range(n)])
        res[name] = (float((ranks == 1).mean()), float((1.0 / ranks).mean()))
    same = np.diag(sim)
    diff = sim[~np.eye(n, dtype=bool)]
    auc = roc_auc_score([1] * len(same) + [0] * len(diff), list(same) + list(diff))
    return dict(top1=(res["A->B"][0] + res["B->A"][0]) / 2, mrr=(res["A->B"][1] + res["B->A"][1]) / 2,
                auroc=float(auc), chance_top1=1.0 / n, n_authors=n)


def evaluate_sets(sets, center, tau, trained=None):
    pids = sorted(sets)
    rows = []
    modes = [("mean_pool", lambda S: mean_pool(S, center)[0]),
             ("attention (parameter-free)", lambda S: attention_pool(S, tau, center)[0])]
    if trained is not None:
        modes.append(("attention (trained)", lambda S: trained.encode(S)[0]))
    for name, f in modes:
        VA = np.array([f(sets[p][0]) for p in pids])
        VB = np.array([f(sets[p][1]) for p in pids])
        rows.append(dict(encoder=name, **retrieval_metrics(VA, VB)))
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# TORCH MODEL (imported lazily so the rest of Module D works without torch)
# ------------------------------------------------------------------
def make_model(d_in=768, d=256, heads=4, layers=1):
    import torch
    import torch.nn as nn

    class AttnStyleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(d_in, d)
            layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
            self.norm = nn.LayerNorm(d)

        def forward(self, x, pad_mask):                       # x (B,n,d_in); pad_mask (B,n) True = padding
            h = self.enc(self.proj(x), src_key_padding_mask=pad_mask)
            q = self.query.expand(x.size(0), -1, -1)
            out, w = self.pool(q, h, h, key_padding_mask=pad_mask, need_weights=True)   # w: (B,1,n)
            z = torch.nn.functional.normalize(self.norm(out.squeeze(1)), dim=-1)
            return z, w.squeeze(1)

    return AttnStyleEncoder()


class TrainedEncoder:
    """Wrapper used by module_d.StyleEncoder(mode='trained')."""
    def __init__(self, ckpt, device="cpu"):
        import torch
        self.torch = torch
        blob = torch.load(ckpt, map_location=device)
        self.model = make_model(**blob["cfg"]).to(device)
        self.model.load_state_dict(blob["state"])
        self.model.eval()
        self.device = device

    def encode(self, S):
        t = self.torch
        x = t.tensor(np.asarray(S), dtype=t.float32, device=self.device)[None]
        mask = t.zeros(1, x.size(1), dtype=t.bool, device=self.device)
        with t.no_grad():
            z, w = self.model(x, mask)
        return z[0].cpu().numpy(), w[0].cpu().numpy()


def _pad(batch, torch, device):
    n = max(len(b) for b in batch)
    d = batch[0].shape[1]
    x = np.zeros((len(batch), n, d), dtype=np.float32)
    m = np.ones((len(batch), n), dtype=bool)
    for i, b in enumerate(batch):
        x[i, :len(b)] = b
        m[i, :len(b)] = False
    return torch.tensor(x, device=device), torch.tensor(m, device=device)


def train(train_sets, val_sets, args, d_in):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(d_in=d_in, d=256, heads=4, layers=1)
    model = make_model(**cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    pids = sorted(train_sets)
    best, best_state = -1.0, None

    def subset(S):
        k = rng.integers(min(3, len(S)), len(S) + 1)
        return S[np.sort(rng.choice(len(S), k, replace=False))]

    def val_mrr():
        model.eval()
        vp = sorted(val_sets)
        with torch.no_grad():
            V = []
            for j in (0, 1):
                x, m = _pad([val_sets[p][j].astype(np.float32) for p in vp], torch, dev)
                V.append(model(x, m)[0].cpu().numpy())
        model.train()
        return retrieval_metrics(V[0], V[1])["mrr"]

    model.train()
    for ep in range(1, args.epochs + 1):
        rng.shuffle(pids)
        losses = []
        for i in range(0, len(pids), args.batch):
            b = pids[i:i + args.batch]
            if len(b) < 2:
                continue
            A, P = [], []
            for p in b:
                a, c = (0, 1) if rng.random() < 0.5 else (1, 0)           # random role swap
                A.append(subset(train_sets[p][a]).astype(np.float32))
                P.append(subset(train_sets[p][c]).astype(np.float32))
            za = model(*_pad(A, torch, dev))[0]
            zp = model(*_pad(P, torch, dev))[0]
            logits = za @ zp.T / args.temp
            tgt = torch.arange(len(b), device=dev)
            loss = (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 10 == 0 or ep == args.epochs:
            v = val_mrr()
            print(f"epoch {ep:4d}  loss {np.mean(losses):.3f}  val MRR {v:.3f}")
            if v > best:
                best, best_state = v, {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"best val MRR {best:.3f}")
    return model, cfg

# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset"); ap.add_argument("--out", default="results_d")
    ap.add_argument("--sent-embedder", choices=["cisr", "mock"], default="cisr")
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.1); ap.add_argument("--device", default=None)
    ap.add_argument("--n-val", type=int, default=8, help="training participants held back for model selection")
    ap.add_argument("--skip-train", action="store_true")
    args, _ = ap.parse_known_args()
    os.makedirs(args.out, exist_ok=True)

    data = load_participants(args.data)
    split = make_split(list(data), 0.3, SEED)                       # same split as module_d.py
    json.dump(split, open(os.path.join(args.out, "D_split.json"), "w"), indent=1)
    emb = make_sentence_embedder(args.sent_embedder)
    allsent = [s for p in data.values() for c in p["control"] for s in split_sentences(c["text"])]
    center = emb.embed(allsent).mean(0)

    train_ids = split["train"]
    val_ids, fit_ids = train_ids[-args.n_val:], train_ids[:-args.n_val]
    held = doc_sets(data, split["heldout"], emb)
    trained = None

    if not args.skip_train:
        fit, val = doc_sets(data, fit_ids, emb), doc_sets(data, val_ids, emb)
        print(f"train authors {len(fit)} | val authors {len(val)} | held-out authors {len(held)}")
        d_in = next(iter(fit.values()))[0].shape[1]
        model, cfg = train(fit, val, args, d_in)
        import torch
        ck = os.path.join(args.out, "style_encoder.pt")
        torch.save(dict(cfg=cfg, state=model.state_dict()), ck)
        print("saved", ck)
        trained = TrainedEncoder(ck, args.device or "cpu")

    res = evaluate_sets(held, center, args.tau, trained)
    res.to_csv(os.path.join(args.out, "D_encoder_eval.csv"), index=False)
    print("\nCROSS-TOPIC AUTHOR RETRIEVAL ON HELD-OUT PARTICIPANTS")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()