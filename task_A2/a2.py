"""
module_a2_patched.py  -  Module A2: Similarity vs. Vocabulary Richness

Changes vs. your version:
  1. Bootstrap 95% CIs for Spearman and partial Spearman (ci_low, ci_high).
  2. LUAR embeddings are cached (each text is embedded once, not hundreds of times).
  3. Sanity check on control/treatment counts per participant.
  4. WER can be word-only (WER_USE_TOKENIZER=True, your original) or raw whitespace
     tokens with punctuation/case kept (False, closer to a plain WER).
  5. Data folder / output folder / --no-luar can be passed on the command line.

Usage:
  pip install pandas numpy scipy statsmodels torch transformers
  python module_a2_patched.py --data logs --out results
  python module_a2_patched.py --data logs --out results --no-luar   # lexical + WER only
"""

import argparse, glob, json, os, re
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# CONFIG
# ============================================================
LUAR_MODEL_NAME = "rrivera1849/LUAR-MUD"
LUAR_MAX_LENGTH = 512
N_BOOT = 5000
WER_USE_TOKENIZER = True   # True: lowercase words only | False: raw .split() tokens


# ============================================================
# TEXT UTILITIES
# ============================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def tokenize(text):
    text = clean_text(text)
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower()) if text else []


# ============================================================
# LOAD DATA
# ============================================================
def load_data(data_folder):
    files = glob.glob(os.path.join(data_folder, "*.json"))

    print("JSON files found:", len(files))

    if not files:
        raise FileNotFoundError(
            f"No JSON files found in: {data_folder}"
        )

    rows = []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        pid = data["user_info"]["id"]

        for rid, r in data.get("responses", {}).items():

            # ------------------------------------------------
            # Check whether this response has the required field
            # ------------------------------------------------
            if "model_generation_shown" not in r:
                print(
                    f"WARNING: Skipping response without "
                    f"'model_generation_shown': {fp} | response={rid}"
                )
                continue

            shown = int(r["model_generation_shown"])
            treat = shown == 1

            rows.append({
                "participant_id": pid,
                "response_id": rid,
                "scenario": r.get("scenario", ""),

                "condition":
                    "treatment" if treat else "control",

                # Only treatment uses the LLM generation
                "model_generation": (
                    clean_text(
                        r.get("model_generation", "")
                    )
                    if treat
                    else None
                ),

                # Participant's final writing
                "final_text": clean_text(
                    r.get("final_version", "")
                ),
            })

    df = pd.DataFrame(rows)

    print("Valid responses loaded:", len(df))

    return df


# ============================================================
# VOCABULARY METRICS
# ============================================================
def calculate_ttr(tokens):
    return len(set(tokens)) / len(tokens) if tokens else np.nan


def mtld_one_pass(tokens, threshold=0.72):
    if not tokens:
        return np.nan
    factors, types, count = 0.0, set(), 0
    for t in tokens:
        types.add(t)
        count += 1
        if len(types) / count <= threshold:
            factors += 1
            types, count = set(), 0
    if count > 0:
        factors += (1 - len(types) / count) / (1 - threshold)
    return len(tokens) / factors if factors else np.nan


def calculate_mtld(tokens):
    if len(tokens) < 10:
        return np.nan
    vals = [v for v in (mtld_one_pass(tokens), mtld_one_pass(tokens[::-1])) if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def calculate_yules_k(tokens):
    N = len(tokens)
    if N == 0:
        return np.nan
    fof = Counter(Counter(tokens).values())
    s = sum(f * f * c for f, c in fof.items())
    return 10000 * (s - N) / (N * N)


def vocabulary_profile(text):
    tokens = tokenize(text)
    return {
        "number_of_tokens": len(tokens),
        "number_of_unique_words": len(set(tokens)),
        "TTR": calculate_ttr(tokens),
        "MTLD": calculate_mtld(tokens),
        "Yules_K": calculate_yules_k(tokens),
    }


def participant_control_profile(control_rows):
    texts = [t for t in control_rows["final_text"].fillna("").tolist() if t.strip()]
    profile = vocabulary_profile("\n".join(texts))
    profile["number_of_control_responses"] = len(texts)
    return profile


# ============================================================
# WER
# ============================================================
def calculate_wer(reference, hypothesis):
    if WER_USE_TOKENIZER:
        ref, hyp = tokenize(reference), tokenize(hypothesis)
    else:
        ref, hyp = reference.split(), hypothesis.split()
    n, m = len(ref), len(hyp)
    if n == 0:
        return np.nan
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp[n, m] / n


# ============================================================
# LUAR (with cache)
# ============================================================
def load_luar():
    import torch
    from transformers import AutoModel, AutoTokenizer
    print("\nLoading LUAR:", LUAR_MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(LUAR_MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(LUAR_MODEL_NAME, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print("LUAR device:", device)
    return tok, model, device, torch


_EMB_CACHE = {}


def get_luar_embedding(text, tokenizer, model, device, torch):
    text = clean_text(text)
    if not text:
        return None
    if text in _EMB_CACHE:                       # <- cache: each text embedded once
        return _EMB_CACHE[text]

    inp = tokenizer([text], max_length=LUAR_MAX_LENGTH, padding="max_length",
                    truncation=True, return_tensors="pt")
    ids = inp["input_ids"].reshape(1, 1, -1).to(device)
    am = inp["attention_mask"].reshape(1, 1, -1).to(device)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=am)
    if isinstance(out, tuple):
        out = out[0]
    emb = out.detach().cpu().numpy()[0]
    norm = np.linalg.norm(emb)
    emb = emb / norm if norm > 0 else None
    _EMB_CACHE[text] = emb
    return emb


def cosine_similarity(a, b):
    if a is None or b is None:
        return np.nan
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else np.nan


def similarity_to_control(target_text, control_texts, tokenizer, model, device, torch):
    if not target_text:
        return np.nan
    t = get_luar_embedding(target_text, tokenizer, model, device, torch)
    sims = []
    for c in control_texts:
        s = cosine_similarity(t, get_luar_embedding(c, tokenizer, model, device, torch))
        if not np.isnan(s):
            sims.append(s)
    return float(np.mean(sims)) if sims else np.nan


# ============================================================
# RESPONSE- AND PARTICIPANT-LEVEL TABLES
# ============================================================
def create_response_level_results(df, luar):
    results = []
    for pid, g in df.groupby("participant_id"):
        control_texts = [t for t in g[g.condition == "control"]["final_text"] if t.strip()]
        for _, row in g[g.condition == "treatment"].iterrows():
            llm, final = row["model_generation"], row["final_text"]
            if not llm or not final:
                continue
            wer = calculate_wer(llm, final)
            if luar is not None:
                before = similarity_to_control(llm, control_texts, *luar)
                after = similarity_to_control(final, control_texts, *luar)
                delta = after - before if not (np.isnan(before) or np.isnan(after)) else np.nan
            else:
                before = after = delta = np.nan
            results.append({
                "participant_id": pid, "response_id": row["response_id"],
                "scenario": row["scenario"], "WER": wer,
                "LUAR_similarity_before": before, "LUAR_similarity_after": after,
                "delta_similarity": delta,
            })
    return pd.DataFrame(results)


def create_participant_level_table(df, resp):
    rows = []
    for pid, g in df.groupby("participant_id"):
        voc = participant_control_profile(g[g.condition == "control"])
        r = resp[resp.participant_id == pid]
        rows.append({
            "participant_id": pid,
            "control_tokens": voc["number_of_tokens"],
            "control_unique_words": voc["number_of_unique_words"],
            "TTR": voc["TTR"], "MTLD": voc["MTLD"], "Yules_K": voc["Yules_K"],
            "mean_WER": r["WER"].mean(),
            "mean_LUAR_similarity_before": r["LUAR_similarity_before"].mean(),
            "mean_LUAR_similarity_after": r["LUAR_similarity_after"].mean(),
            "mean_delta_similarity": r["delta_similarity"].mean(),
            "number_of_control_responses": voc["number_of_control_responses"],
            "number_of_treatment_responses": len(r),
        })
    return pd.DataFrame(rows)


# ============================================================
# STATISTICS
# ============================================================
def _spearman(x, y):
    return stats.spearmanr(x, y)[0]


def _partial_rho_p(x, y, z):
    """Partial Spearman: rank everything, residualise x,y on z, correlate residuals."""
    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    Z = np.column_stack([np.ones(len(rz)), rz])
    res = lambda a: a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    return stats.pearsonr(res(rx), res(ry))


def _bootstrap_ci(stat_fn, arrays, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    size = len(arrays[0])
    vals = []
    for _ in range(n):
        i = rng.integers(0, size, size)
        try:
            v = stat_fn(*[a[i] for a in arrays])
        except Exception:
            v = np.nan
        vals.append(v)
    vals = np.array(vals, dtype=float)
    if np.isnan(vals).all():
        return np.nan, np.nan
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def spearman_correlation(data, x_col, y_col):
    s = data[[x_col, y_col]].dropna()
    if len(s) < 3:
        return {"N": len(s), "rho": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    x, y = s[x_col].to_numpy(), s[y_col].to_numpy()
    rho, p = stats.spearmanr(x, y)
    lo, hi = _bootstrap_ci(_spearman, [x, y])
    return {"N": len(s), "rho": rho, "p_value": p, "ci_low": lo, "ci_high": hi}


def partial_spearman(data, x_col, y_col, z_col):
    s = data[[x_col, y_col, z_col]].dropna()
    if len(s) < 5:
        return {"N": len(s), "rho": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    x, y, z = (s[c].to_numpy() for c in (x_col, y_col, z_col))
    rho, p = _partial_rho_p(x, y, z)
    lo, hi = _bootstrap_ci(lambda a, b, c: _partial_rho_p(a, b, c)[0], [x, y, z])
    return {"N": len(s), "rho": rho, "p_value": p, "ci_low": lo, "ci_high": hi}


def apply_holm_correction(results):
    from statsmodels.stats.multitest import multipletests
    results = results.copy()
    results["p_value_holm"] = np.nan
    results["significant_holm"] = False
    valid = results["p_value"].notna()
    if valid.any():
        rej, adj, _, _ = multipletests(results.loc[valid, "p_value"], method="holm")
        results.loc[valid, "p_value_holm"] = adj
        results.loc[valid, "significant_holm"] = rej
    return results


METRICS = ["TTR", "MTLD", "Yules_K"]
OUTCOMES = [("mean_WER", "editing_effort"), ("mean_delta_similarity", "similarity_improvement")]


def correlation_analysis(pt):
    rows = []
    for m in METRICS:
        for col, name in OUTCOMES:
            rows.append({"analysis": "Spearman", "vocabulary_metric": m, "outcome": name,
                         **spearman_correlation(pt, m, col)})
    return apply_holm_correction(pd.DataFrame(rows))


def partial_correlation_analysis(pt):
    rows = []
    for m in METRICS:
        for col, name in OUTCOMES:
            rows.append({"analysis": "Partial Spearman", "vocabulary_metric": m, "outcome": name,
                         "control_variable": "control_tokens",
                         **partial_spearman(pt, m, col, "control_tokens")})
    return apply_holm_correction(pd.DataFrame(rows))



def sanity_check(df):
    print("\n===== SANITY CHECK =====")

    counts = (
        df.groupby(["participant_id", "condition"])
        .size()
        .unstack(fill_value=0)
    )

    print("\nResponses per participant:")
    print(counts)

    # Expected structure from the dataset:
    # 2 control responses
    # 4 treatment responses

    bad_control = counts[
        counts.get("control", 0) != 2
    ]

    bad_treatment = counts[
        counts.get("treatment", 0) != 4
    ]

    if len(bad_control) > 0:
        print("\nWARNING: Participants with control count != 2:")
        print(bad_control)

    if len(bad_treatment) > 0:
        print("\nWARNING: Participants with treatment count != 4:")
        print(bad_treatment)

    if len(bad_control) == 0 and len(bad_treatment) == 0:
        print("\nPASS: Every participant has 2 control and 4 treatment responses.")

    print("========================\n")

# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        default=r"D:\ML_LAB\MLdataset\personal_style_postedit-main\logs"
    )

    ap.add_argument(
        "--out",
        default=r"D:\ML_LAB\MLdataset\personal_style_postedit-main\results"
    )

    ap.add_argument("--no-luar", action="store_true")

    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load_data(a.data)

    print(
        "Participants:",
        df.participant_id.nunique(),
        "| control:",
        (df.condition == "control").sum(),
        "| treatment:",
        (df.condition == "treatment").sum()
    )

    sanity_check(df)

    luar = None if a.no_luar else load_luar()

    resp = create_response_level_results(df, luar)
    pt = create_participant_level_table(df, resp)

    resp.to_csv(
        os.path.join(a.out, "A2_response_level.csv"),
        index=False
    )

    pt.to_csv(
        os.path.join(a.out, "A2_participant_level.csv"),
        index=False
    )

    corr = correlation_analysis(pt)
    part = partial_correlation_analysis(pt)

    corr.to_csv(
        os.path.join(a.out, "A2_correlations.csv"),
        index=False
    )

    part.to_csv(
        os.path.join(a.out, "A2_partial_correlations.csv"),
        index=False
    )

    cols = [
        "vocabulary_metric",
        "outcome",
        "N",
        "rho",
        "ci_low",
        "ci_high",
        "p_value",
        "p_value_holm"
    ]

    print("\nSPEARMAN\n", corr[cols].round(3).to_string(index=False))

    print(
        "\nPARTIAL SPEARMAN (controlling for control_tokens)\n",
        part[cols].round(3).to_string(index=False)
    )

    print("\nDone. Files saved in", a.out)


if __name__ == "__main__":
    main()