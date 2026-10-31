"""
module_d.py - Module D: Cross-Topic, Attention-Based Style Capture + Style-Conditioned Drafting

Question: if the LLM draft is conditioned on the user's OWN writing (on a different topic),
is the UNEDITED draft already closer to the user's style than the paper's generic zero-shot draft?

Per participant (2 control docs on topics X and Y, 4 treatment docs on other topics Z):
    style SOURCE = control doc on X      (used to build the style vector / exemplars / style card)
    style REFERENCE = control doc on Y   (HELD OUT: never shown to the generator, used only to score)
    both directions are used (X->Y and Y->X). Target topic Z is different from X and Y (cross-topic).

Arms (all LLM arms share the same generator, so the only thing that changes is the style conditioning)
    baseline_o4mini  paper's zero-shot GPT-o4-mini draft (from the data)
    zero_shot        OUR generator, no style information            <- fair baseline
    style_card       measured stylometric description of the source text (contractions, sentence length ...)
    rag_attn         source sentences chosen by ATTENTION weights over sentence embeddings (the "RAG" step)
    rag_random       same number of RANDOM source sentences          <- ablation: does attention help?
    rag_rerank       N candidates from rag_attn, pick the one closest to the source style vector
    rag_edit         (optional, needs Module B outputs) rag_attn + the user's editing habits + words to avoid

Scoring (primary = LUAR as in the paper; selection model is different from the scoring model)
    luar_ref   LUAR cosine(draft, held-out control doc)     higher = more like the user
    sty_ref    CISR cosine(draft, held-out control doc)     higher = better (NOT independent for rag_rerank)
    stylo_dist z-scored stylometric distance to held-out doc lower = better
    wer_final / luar_final   distance to the user's own post-edited final text (proxy for editing burden;
                             biased AGAINST new drafts because the final text was edited FROM the o4-mini draft)
    ai_rate    AI-lexicon hits per 100 words (needs --lexicon from build_lexicon.py)

Usage
    python module_d.py --stage all --data logs --out results_d --backend hf --model Qwen/Qwen2.5-3B-Instruct
    python module_d.py --stage generate --backend anthropic --model claude-haiku-4-5-20251001
    python module_d.py --stage evaluate --out results_d
    python module_d.py --stage all --backend mock --sent-embedder mock --luar-embedder mock   # pipeline test, no GPU
"""
import argparse, glob, json, os, re, zlib, difflib
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
# Please VERIFY these against Table 9 / the prompts of the source paper. Unknown labels fall back to the raw label.
SCENARIO_DESC = {
    "apology": "an apology letter",
    "condolence": "a condolence message",
    "letter": "a catch-up letter to a friend",
    "thank": "a thank-you letter",
    "reassurance": "a message reassuring someone close to the writer",
    "wedding": "wedding vows",
    "speech": "a wedding speech",
}
FIRST_PERSON = {"i", "my", "me", "i'm", "i've", "i'll", "mine", "myself"}
STOP = set("""the and for that with this from have has had was were are but not you your they their them what when
where which while about would could should there then than into over also just very more most some such only
being been will shall can its it's his her our out all any each other these those because after before again""".split())
SEED = 0
ARMS_DEFAULT = ["zero_shot", "style_card", "rag_attn", "rag_random", "rag_rerank"]


def norm(x):
    return ("" if x is None else str(x)).replace("\u2019", "'").replace("\u2018", "'").strip()


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if p and p.strip()]


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def l2(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def cos(a, b):
    return float(np.dot(l2(a), l2(b)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
def load_participants(folder):
    out = {}
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    for fp in files:
        d = json.load(open(fp, encoding="utf-8"))
        pid = d["user_info"]["id"]
        ctrl, treat = [], []
        for rid, r in d.get("responses", {}).items():
            shown = r.get("model_generation_shown")
            if shown is None:
                continue
            base = dict(rid=rid, scenario=r.get("scenario", "unknown"), details=norm(r.get("details")))
            if int(shown) == 0:
                t = norm(r.get("final_version"))
                if t:
                    ctrl.append(dict(base, text=t))
            else:
                dr, fi = norm(r.get("model_generation")), norm(r.get("final_version"))
                if dr and fi:
                    treat.append(dict(base, draft=dr, final=fi))
        ctrl.sort(key=lambda c: c["scenario"])
        treat.sort(key=lambda c: c["scenario"])
        out[pid] = dict(control=ctrl, treat=treat)
    print(f"participants loaded: {len(out)} (from {len(files)} files)")
    return out


def make_split(pids, frac=0.3, seed=SEED):
    """Participant-disjoint split, shared with d_encoder.py (same seed => same split)."""
    pids = sorted(pids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pids))
    n_hold = max(1, int(round(frac * len(pids))))
    hold = sorted(pids[i] for i in order[:n_hold])
    train = sorted(pids[i] for i in order[n_hold:])
    return dict(train=train, heldout=hold)


def build_items(data, pids, directions="both"):
    items = []
    for pid in pids:
        c, t = data[pid]["control"], data[pid]["treat"]
        if len(c) < 2:
            continue
        dirs = [(0, 1), (1, 0)] if directions == "both" else [(0, 1)]
        for tr in t:
            for s, r in dirs:
                items.append(dict(
                    item_id=f"{pid}|{tr['rid']}|{c[s]['scenario']}", target_key=f"{pid}|{tr['rid']}",
                    pid=pid, rid=tr["rid"], target_scenario=tr["scenario"], details=tr["details"],
                    baseline=tr["draft"], final=tr["final"],
                    src_text=c[s]["text"], src_scenario=c[s]["scenario"],
                    ref_text=c[r]["text"], ref_scenario=c[r]["scenario"]))
    return items

# ------------------------------------------------------------------
# EMBEDDERS (real + mock for testing)
# ------------------------------------------------------------------
def _hash_vec(tokens, dim):
    v = np.zeros(dim)
    for t in tokens:
        v[zlib.crc32(t.encode()) % dim] += 1.0
    return v


class MockSentenceEmbedder:
    dim = 64
    def __init__(self): self.cache = {}
    def embed(self, texts):
        out = []
        for t in texts:
            if t not in self.cache:
                tl = t.lower()
                grams = [tl[i:i + 3] for i in range(max(len(tl) - 2, 1))]
                self.cache[t] = l2(_hash_vec(grams, self.dim) + 0.1)
            out.append(self.cache[t])
        return np.array(out)


class MockLuar:
    dim = 64
    def __init__(self): self.cache = {}
    def embed(self, texts):
        out = []
        for t in texts:
            if t not in self.cache:
                self.cache[t] = l2(_hash_vec(words(t), self.dim) + 0.1)
            out.append(self.cache[t])
        return np.array(out)


class CisrEmbedder:
    """Sentence-level, content-independent style embeddings (Wegmann et al. 2022; reference [4])."""
    def __init__(self, name="AnnaWegmann/Style-Embedding", device=None):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name, device=device)
        self.cache = {}
    def embed(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            E = self.m.encode(todo, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
            self.cache.update(dict(zip(todo, E)))
        return np.array([self.cache[t] for t in texts])


class LuarEmbedder:
    """Document-level LUAR (reference [2]); one document per episode, as in Module A2."""
    def __init__(self, name="rrivera1849/LUAR-MUD", device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True).to(self.dev).eval()
        self.cache = {}
    def embed(self, texts):
        for t in dict.fromkeys(texts):
            if t in self.cache:
                continue
            x = self.tok([t], max_length=512, padding="max_length", truncation=True, return_tensors="pt")
            ids = x["input_ids"].reshape(1, 1, -1).to(self.dev)
            am = x["attention_mask"].reshape(1, 1, -1).to(self.dev)
            with self.torch.inference_mode():
                o = self.model(input_ids=ids, attention_mask=am)
            o = o[0] if isinstance(o, tuple) else o
            self.cache[t] = l2(o.detach().cpu().numpy()[0])
        return np.array([self.cache[t] for t in texts])


def make_sentence_embedder(kind):
    return MockSentenceEmbedder() if kind == "mock" else CisrEmbedder()


def make_luar(kind):
    return MockLuar() if kind == "mock" else LuarEmbedder()

# ------------------------------------------------------------------
# ATTENTION-BASED STYLE ENCODER (parameter-free version + wrapper for the trained one)
# ------------------------------------------------------------------
def mean_pool(S, center=None):
    S = np.asarray(S, float)
    X = S - center if center is not None else S
    v = l2(X.mean(0))
    return v, np.full(len(S), 1.0 / len(S))


def attention_pool(S, tau=0.1, center=None):
    """
    Scaled dot-product attention (Vaswani et al. 2017, reference [9]) over sentence embeddings.
      1. self-attention:     H = softmax(X X^T / tau) X    each sentence is re-expressed through the sentences
                                                           that share its style
      2. attention pooling:  w = softmax(H q / tau),  v = sum_i w_i H_i,   q = mean of H
    Embeddings are L2-normalised, so the 1/sqrt(d_k) scale is replaced by a temperature tau.
    Optional `center` removes the component shared by all writers (global mean sentence embedding).
    Returns the style vector v and the per-sentence weights w (how representative each sentence is).
    """
    S = np.asarray(S, float)
    X = l2(S - center) if center is not None else l2(S)
    H = softmax(X @ X.T / tau, axis=1) @ X
    q = l2(H.mean(0))
    w = softmax(H @ q / tau)
    v = l2((w[:, None] * H).sum(0))
    return v, w


class StyleEncoder:
    def __init__(self, sent_emb, mode="attn", center=None, tau=0.1, trained=None):
        self.emb, self.mode, self.center, self.tau, self.trained = sent_emb, mode, center, tau, trained

    def encode(self, text):
        sents = split_sentences(text)
        if not sents:
            return None, np.array([]), []
        S = self.emb.embed(sents)
        if self.mode == "mean":
            v, w = mean_pool(S, self.center)
        elif self.mode == "trained":
            v, w = self.trained.encode(S)
        else:
            v, w = attention_pool(S, self.tau, self.center)
        return v, w, sents

    def top_sentences(self, text, k=4):
        v, w, sents = self.encode(text)
        if v is None:
            return []
        idx = sorted(np.argsort(-w)[:k])           # keep original order for readability
        return [sents[i] for i in idx]


def load_trained(ckpt, device="cpu"):
    from Dencoder import TrainedEncoder
    return TrainedEncoder(ckpt, device)

# ------------------------------------------------------------------
# STYLOMETRIC FEATURES + STYLE CARD
# ------------------------------------------------------------------
CONTR = re.compile(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", re.I)
FEATS = ["sent_len", "word_len", "contr100", "fp100", "comma_sent", "dash100", "excl100", "quest100",
         "semi100", "ellipsis100", "long_frac"]


def stylo_features(text):
    text = norm(text)
    w = words(text)
    n = max(len(w), 1)
    ns = max(len(split_sentences(text)), 1)
    return np.array([
        n / ns,
        np.mean([len(x) for x in w]) if w else 0.0,
        100 * len(CONTR.findall(text)) / n,
        100 * sum(x in FIRST_PERSON for x in w) / n,
        text.count(",") / ns,
        100 * (text.count("\u2014") + text.count("\u2013")) / n,
        100 * text.count("!") / n,
        100 * text.count("?") / n,
        100 * (text.count(";") + text.count(":")) / n,
        100 * (text.count("...") + text.count("\u2026")) / n,
        sum(len(x) >= 9 for x in w) / n,
    ])


def style_card(text):
    f = dict(zip(FEATS, stylo_features(text)))
    L = []
    sl = f["sent_len"]
    L.append(f"Sentence length: about {sl:.0f} words on average ({'short' if sl < 11 else 'long' if sl > 20 else 'medium'} sentences).")
    L.append("Contractions: " + ("rarely used; writes 'do not', 'I am'." if f["contr100"] < 0.5 else
                                 f"used often (about {f['contr100']:.1f} per 100 words)." if f["contr100"] > 2 else "used sometimes."))
    L.append("First person ('I', 'my'): " + ("mostly avoided." if f["fp100"] < 2 else f"frequent (about {f['fp100']:.1f} per 100 words)."))
    L.append("Em dashes: " + ("never used." if f["dash100"] == 0 else "used."))
    L.append("Exclamation marks: " + ("never used." if f["excl100"] == 0 else "used sparingly." if f["excl100"] < 1.5 else "used often."))
    if f["ellipsis100"] > 0:
        L.append("Uses ellipses ('...').")
    L.append(f"Commas: about {f['comma_sent']:.1f} per sentence.")
    L.append("Vocabulary: " + ("mostly short, everyday words." if f["long_frac"] < 0.05 else "includes longer, more formal words."))
    return "\n".join("- " + x for x in L)

# ------------------------------------------------------------------
# MODULE B INTEGRATION (optional)
# ------------------------------------------------------------------
def load_edit_table(path):
    return pd.read_csv(path) if path and os.path.exists(path) else None


def edit_cues(tab, pid, rid):
    """Editing habits of this user from Module B, computed from their OTHER documents (no leakage)."""
    g = tab[(tab.pid == pid) & (tab.doc_id != f"{pid}_{rid}")]
    if len(g) == 0:
        return ""
    rules = [("et_em_dash_removed", 0.05, "removes em dashes"), ("et_contraction_added", 0.10, "adds contractions"),
             ("et_shortened", 0.10, "shortens long sentences"), ("et_simplified_words", 0.10, "swaps fancy words for plain ones"),
             ("et_first_person_added", 0.10, "makes the text more personal (adds I / my)"),
             ("et_lengthened", 0.15, "adds extra personal detail")]
    hab = [txt for col, th, txt in rules if col in g and g[col].mean() > th]
    return ("When this writer edits AI drafts they usually: " + "; ".join(hab) + ".") if hab else ""

# ------------------------------------------------------------------
# PROMPTS + GENERATORS
# ------------------------------------------------------------------
def desc(scn):
    return SCENARIO_DESC.get(scn, f"a piece of personal writing ({scn})")


def build_prompt(arm, it, enc, rng, k=4, edit_tab=None, avoid=None):
    target_words = max(len(it["baseline"].split()), 30)
    head = (f"Write {desc(it['target_scenario'])} in the first person, as if the writer wrote it themselves, "
            f"using the details below. Write about {target_words} words. "
            f"Output only the text itself, with no title or commentary.\n\nDETAILS:\n{it['details']}\n")
    if arm == "zero_shot":
        return head
    tail = ""
    if arm == "style_card":
        tail = ("\nSTYLE GUIDE (measured from the writer's own earlier writing on an unrelated topic):\n"
                + style_card(it["src_text"]) + "\nFollow this style guide. Do not mention it.\n")
    else:
        if arm == "rag_random":
            sents = split_sentences(it["src_text"])
            idx = sorted(rng.choice(len(sents), size=min(k, len(sents)), replace=False))
            ex = [sents[i] for i in idx]
        else:
            ex = enc.top_sentences(it["src_text"], k)
        tail = ("\nEXAMPLE SENTENCES the writer wrote earlier on an unrelated topic. Imitate their voice "
                "(rhythm, word choice, punctuation, tone) but do NOT reuse their facts, names or events:\n"
                + "\n".join(f"- {s}" for s in ex) + "\n")
        if arm == "rag_edit":
            cue = edit_cues(edit_tab, it["pid"], it["rid"]) if edit_tab is not None else ""
            if cue:
                tail += "\n" + cue + " Write the draft so these edits are not needed.\n"
            if avoid:
                tail += "Avoid these words: " + ", ".join(avoid) + ".\n"
    return head + tail


class MockGenerator:
    """Deterministic fake LLM for testing the pipeline (not a real result)."""
    def generate(self, prompt, n, temperature=0.8, max_new_tokens=450):
        det = prompt.split("DETAILS:\n")[1].split("\n\n")[0].split("\n")[0:6] if "DETAILS:" in prompt else ["hello"]
        outs = []
        for c in range(n):
            rng = np.random.default_rng(zlib.crc32((prompt + str(c)).encode()))
            body = " ".join(f"I {rng.choice(['really', 'truly', 'just'])} want to talk about {d.strip().lower()}." for d in det if d.strip())
            if "never used" in prompt or "Contractions: used" in prompt:
                body = body.replace(" want", "'ve wanted")
            outs.append(body)
        return outs


class HFGenerator:
    def __init__(self, name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")
    def generate(self, prompt, n, temperature=0.8, max_new_tokens=450):
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        x = self.tok(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**x, do_sample=True, temperature=temperature, top_p=0.9, max_new_tokens=max_new_tokens,
                                  num_return_sequences=n, pad_token_id=self.tok.eos_token_id)
        L = x["input_ids"].shape[1]
        return [self.tok.decode(o[L:], skip_special_tokens=True).strip() for o in out]


class AnthropicGenerator:
    def __init__(self, name):
        import anthropic
        self.client, self.name = anthropic.Anthropic(), name       # reads ANTHROPIC_API_KEY
    def generate(self, prompt, n, temperature=0.8, max_new_tokens=450):
        res = []
        for _ in range(n):
            r = self.client.messages.create(model=self.name, max_tokens=max_new_tokens, temperature=temperature,
                                            messages=[{"role": "user", "content": prompt}])
            res.append(r.content[0].text.strip())
        return res


def make_generator(backend, model):
    if backend == "mock":
        return MockGenerator()
    return HFGenerator(model) if backend == "hf" else AnthropicGenerator(model)

# ------------------------------------------------------------------
# STAGE 1: GENERATE
# ------------------------------------------------------------------
def pick_pids(data, args):
    split = make_split(list(data), 0.3, SEED)
    os.makedirs(args.out, exist_ok=True)
    json.dump(split, open(os.path.join(args.out, "D_split.json"), "w"), indent=1)
    pids = split["heldout"] if args.eval_split == "heldout" else sorted(data)
    if args.limit_participants:
        pids = pids[:args.limit_participants]
    return pids


def make_encoder(args, sent_emb, data, allowed_pids=None):
    center, trained = None, None

    if args.encoder != "trained":
        # IMPORTANT:
        # Compute the global CISR center only from training participants.
        # This avoids using held-out participants during evaluation.
        pids = allowed_pids if allowed_pids is not None else sorted(data)

        allsent = [
            s
            for pid in pids
            for c in data[pid]["control"]
            for s in split_sentences(c["text"])
        ]

        if allsent:
            center = sent_emb.embed(allsent).mean(0)

    else:
        trained = load_trained(args.encoder_ckpt)

    return StyleEncoder(
        sent_emb,
        args.encoder,
        center,
        args.tau,
        trained
    )


def load_gens(path):
    G = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            G.setdefault((r["arm"], r["key"]), {})[r["cand"]] = r["text"]
    return G


def run_generate(args):
    data = load_participants(args.data)
    pids = pick_pids(data, args)
    items = build_items(data, pids, args.directions)
    print(f"generating for {len(pids)} participants, {len(items)} items, arms={args.arms}")
    sent_emb = make_sentence_embedder(args.sent_embedder)
    split = make_split(list(data), 0.3, SEED)
    train_pids = split["train"]

    enc = make_encoder(
        args,
        sent_emb,
        data,
        allowed_pids=train_pids
    )
    gen = make_generator(args.backend, args.model)
    edit_tab = load_edit_table(args.edit_table)
    avoid = [w.strip() for w in open(args.lexicon)][:15] if args.lexicon and os.path.exists(args.lexicon) else None
    arms = list(args.arms)
    if "rag_edit" not in arms and edit_tab is not None:
        arms.append("rag_edit")

    path = os.path.join(args.out, "D_generations.jsonl")
    G = load_gens(path)
    fh = open(path, "a", encoding="utf-8")

    def emit(arm, key, cand, text):
        G.setdefault((arm, key), {})[cand] = text
        fh.write(json.dumps(dict(arm=arm, key=key, cand=cand, text=text)) + "\n"); fh.flush()

    done = 0
    for it in items:
        rng = np.random.default_rng(zlib.crc32(it["item_id"].encode()))
        for arm in arms:
            if arm == "rag_rerank":
                continue
            key = it["target_key"] if arm == "zero_shot" else it["item_id"]
            if (arm, key) in G and len(G[(arm, key)]) >= args.n_candidates:
                continue
            prompt = build_prompt(arm, it, enc, rng, args.k_exemplars, edit_tab, avoid)
            for c, t in enumerate(gen.generate(prompt, args.n_candidates, args.temperature)):
                emit(arm, key, c, t)
        if "rag_rerank" in arms and ("rag_rerank", it["item_id"]) not in G and ("rag_attn", it["item_id"]) in G:
            sv, _, _ = enc.encode(it["src_text"])
            cands = G[("rag_attn", it["item_id"])]
            scores = {c: (cos(sv, enc.encode(t)[0]) if enc.encode(t)[0] is not None else -1) for c, t in cands.items()}
            best = max(scores, key=scores.get)
            emit("rag_rerank", it["item_id"], 0, cands[best])
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{len(items)} items")
    fh.close()
    print("saved", path)

# ------------------------------------------------------------------
# STAGE 2: EVALUATE
# ------------------------------------------------------------------
def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    if not r:
        return np.nan
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(len(r) + 1), np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / len(r)


def content_words(details):
    return {w for w in words(details) if len(w) >= 5 and w not in STOP}


def _nanmean(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else np.nan


METRICS = ["luar_ref", "sty_ref", "stylo_dist", "wer_final", "luar_final", "ai_rate", "len_ratio", "detail_cov"]
HIGHER_BETTER = {"luar_ref": True, "sty_ref": True, "stylo_dist": False, "wer_final": False, "luar_final": True,
                 "ai_rate": False, "len_ratio": None, "detail_cov": True}


def run_evaluate(args):
    data = load_participants(args.data)
    pids = pick_pids(data, args)
    items = build_items(data, pids, args.directions)
    G = load_gens(os.path.join(args.out, "D_generations.jsonl"))
    luar, cisr = make_luar(args.luar_embedder), make_sentence_embedder(args.sent_embedder)
    lex = {w.strip() for w in open(args.lexicon)} if args.lexicon and os.path.exists(args.lexicon) else set()

    ctrl_feats = np.array([stylo_features(c["text"]) for p in data.values() for c in p["control"]])
    mu, sd = ctrl_feats.mean(0), ctrl_feats.std(0)
    sd = np.where(sd < 0.05, 1.0, sd)            # near-constant features (e.g. ellipses) must not explode the z-score
    zf = lambda t: np.clip((stylo_features(t) - mu) / sd, -5, 5)

    def doc_cisr(t):
        S = cisr.embed(split_sentences(t))
        return l2(S.mean(0))

    rows = []
    for it in items:
        ref_l, fin_l, src_l = (luar.embed([it[k]])[0] for k in ("ref_text", "final", "src_text"))
        ref_c, ref_f = doc_cisr(it["ref_text"]), zf(it["ref_text"])
        base_len, cw = len(words(it["baseline"])), content_words(it["details"])
        ceiling = cos(src_l, ref_l)
        texts = {"baseline_o4mini": [it["baseline"]]}
        for arm in args.arms + ["rag_edit"]:
            key = it["target_key"] if arm == "zero_shot" else it["item_id"]
            if (arm, key) in G:
                texts[arm] = [G[(arm, key)][c] for c in sorted(G[(arm, key)])]
        for arm, tl in texts.items():
            ms = []
            for t in tl:
                if not t.strip():
                    continue
                w = words(t)
                lg = luar.embed([t])[0]
                ms.append(dict(
                    luar_ref=cos(lg, ref_l), sty_ref=cos(doc_cisr(t), ref_c),
                    stylo_dist=float(np.abs(zf(t) - ref_f).mean()),
                    wer_final=wer(it["final"], t), luar_final=cos(lg, fin_l),
                    ai_rate=(100 * sum(x in lex for x in w) / max(len(w), 1)) if lex else np.nan,
                    len_ratio=len(w) / max(base_len, 1),
                    detail_cov=(len(cw & set(w)) / len(cw)) if cw else np.nan))
            if ms:
                rows.append(dict(item_id=it["item_id"], pid=it["pid"], target=it["target_scenario"], src=it["src_scenario"],
                                 ref=it["ref_scenario"], arm=arm, n_texts=len(ms), luar_ceiling=ceiling,
                                 **{m: _nanmean([x[m] for x in ms]) for m in METRICS}))
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(args.out, "D_item_metrics.csv"), index=False)

    # ---- summary by arm + gap closed
    summ = M.groupby("arm")[METRICS].mean()
    base = summ.loc["baseline_o4mini"] if "baseline_o4mini" in summ.index else None
    ceil = M.drop_duplicates("item_id").luar_ceiling.mean()
    if base is not None:
        denominator = ceil - base.luar_ref

        if abs(denominator) < 1e-12:
            summ["luar_gap_closed"] = np.nan
        else:
            summ["luar_gap_closed"] = (
                summ.luar_ref - base.luar_ref
            ) / denominator

    summ["n_items"] = M.groupby("arm").item_id.nunique()
    summ.to_csv(os.path.join(args.out, "D_summary_by_arm.csv"))
    print(f"\nLUAR ceiling (source doc vs held-out doc, same author, different topic): {ceil:.3f}")
    print(summ.round(3).to_string())

    # ---- paired tests at participant level
    P = M.groupby(["pid", "arm"])[METRICS].mean().reset_index()
    tests = []
    for m in METRICS:
        fam = []
        for arm in sorted(P.arm.unique()):
            for comp in ("zero_shot", "baseline_o4mini"):
                if arm == comp or comp not in P.arm.values:
                    continue
                a = P[P.arm == arm].set_index("pid")[m]
                b = P[P.arm == comp].set_index("pid")[m]
                d = (a - b).dropna()
                if len(d) < 3:
                    continue
                rng = np.random.default_rng(0)
                boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
                try:
                    p = stats.wilcoxon(d.values).pvalue
                except ValueError:
                    p = np.nan
                fam.append(dict(metric=m, arm=arm, vs=comp, n_participants=len(d), mean_diff=d.mean(),
                                ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5),
                                dz=d.mean() / (d.std(ddof=1) + 1e-12), p=p, higher_is_better=HIGHER_BETTER[m]))
        ps = np.array([r["p"] for r in fam], dtype=float)
        ok = ~np.isnan(ps)
        holm = np.full(len(fam), np.nan)
        order = np.argsort(ps[ok]); run = 0.0
        for rank, i in enumerate(order):
            run = max(run, (ok.sum() - rank) * ps[ok][i]); holm[np.where(ok)[0][i]] = min(run, 1.0)
        for r, h in zip(fam, holm):
            r["p_holm"] = h
        tests += fam
    T = pd.DataFrame(tests)
    T.to_csv(os.path.join(args.out, "D_paired_tests.csv"), index=False)
    print("\nPAIRED TESTS (participant level; Holm within metric)")
    if len(T):
        print(T[T.metric.isin(["luar_ref", "stylo_dist", "wer_final", "ai_rate"])].round(3).to_string(index=False))

    # ---- qualitative examples
    ex = ["# Module D examples\n"]
    if "rag_rerank" in M.arm.values and "zero_shot" in M.arm.values:
        piv = M.pivot_table(index="item_id", columns="arm", values="luar_ref")
        best = (piv["rag_rerank"] - piv["zero_shot"]).dropna().sort_values(ascending=False).head(5).index
        for iid in best:
            it = next(i for i in items if i["item_id"] == iid)
            ex.append(f"## {iid}\n**Source ({it['src_scenario']}) exemplars:** {enc_free_top(it['src_text'])}\n\n"
                      f"**Details:** {it['details']}\n\n**o4-mini baseline:** {it['baseline']}\n\n"
                      f"**zero_shot:** {G.get(('zero_shot', it['target_key']), {}).get(0, '')}\n\n"
                      f"**rag_rerank:** {G.get(('rag_rerank', iid), {}).get(0, '')}\n")
    open(os.path.join(args.out, "D_examples.md"), "w", encoding="utf-8").write("\n".join(ex))
    print("\nsaved D_item_metrics.csv, D_summary_by_arm.csv, D_paired_tests.csv, D_examples.md in", args.out)


def enc_free_top(text, k=3):
    return " / ".join(split_sentences(text)[:k])

# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["generate", "evaluate", "all"], default="all")
    ap.add_argument("--data", default="dataset"); ap.add_argument("--out", default="results_d")
    ap.add_argument("--eval-split", choices=["heldout", "all"], default="heldout")
    ap.add_argument("--limit-participants", type=int, default=0)
    ap.add_argument("--directions", choices=["both", "one"], default="both")
    ap.add_argument("--arms", nargs="+", default=ARMS_DEFAULT)
    ap.add_argument("--backend", choices=["hf", "anthropic", "mock"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n-candidates", type=int, default=3); ap.add_argument("--k-exemplars", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--encoder", choices=["mean", "attn", "trained"], default="attn")
    ap.add_argument("--encoder-ckpt", default="results_d/style_encoder.pt"); ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--sent-embedder", choices=["cisr", "mock"], default="cisr")
    ap.add_argument("--luar-embedder", choices=["luar", "mock"], default="luar")
    ap.add_argument("--lexicon", default=None, help="ai_lexicon.txt from build_lexicon.py (Module B)")
    ap.add_argument("--edit-table", default=None, help="B_sentence_table.csv from Module B (adds the rag_edit arm)")
    args, _ = ap.parse_known_args()
    os.makedirs(args.out, exist_ok=True)
    if args.stage in ("generate", "all"):
        run_generate(args)
    if args.stage in ("evaluate", "all"):
        run_evaluate(args)


if __name__ == "__main__":
    main()