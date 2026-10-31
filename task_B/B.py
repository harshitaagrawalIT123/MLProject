"""
build_lexicon.py - build a data-driven "AI-sounding word" lexicon for Module B

Idea: a word is "AI-sounding" if it is much MORE frequent in the LLM drafts than in the
participants' own unassisted (control) writing.

  corpus 1 (AI)    : model_generation of every treatment response (the LLM drafts)
  corpus 2 (human) : final_version of every control response (typed from scratch)

Statistic: log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008),
reported as a z-score. The prior shrinks rare words, so one-off words don't dominate.

Guards against false positives (topic words like "wedding"):
  - min count in AI drafts
  - appears in drafts of >= MIN_PARTICIPANTS different participants
  - appears in drafts of >= MIN_SCENARIOS different scenarios
  - rate in AI drafts >= MIN_RATIO x rate in human text

Also exports a removal table: how often each draft word is deleted/replaced by editors.

Usage:
  python build_lexicon.py --data "D:\ML_LAB\MLProjectA2Task\A2-ML\logs" --out lexicon
  python build_lexicon.py --data "D:\ML_LAB\MLProjectA2Task\A2-ML" --out lexicon --exclude-pids id1,id2   # for fold-wise use
"""

import argparse, glob, json, os, re, difflib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

def words(text):
    return WORD.findall((text or "").replace("\u2019", "'").lower())

# ------------------------------------------------------------------
def collect(folder, exclude=()):
    """One pass over the files. Counts are accumulated per file (incremental) and merged."""
    ai, human = Counter(), Counter()
    ai_pids, ai_scen = defaultdict(set), defaultdict(set)
    removed, seen = Counter(), Counter()
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    used = 0
    for fp in files:
        d = json.load(open(fp, encoding="utf-8"))
        pid = d["user_info"]["id"]
        if pid in exclude:
            continue
        used += 1
        for r in d["responses"].values():
            shown = r.get("model_generation_shown")

            if shown is None:
                continue

            if int(shown) == 1:
                draft, final = r.get("model_generation") or "", r.get("final_version") or ""
                dw, fw = words(draft), words(final)
                ai.update(dw)
                for w in set(dw):
                    ai_pids[w].add(pid); ai_scen[w].add(r["scenario"])
                # editor removal: draft words that are deleted/replaced in the final text
                seen.update(dw)
                for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, dw, fw, autojunk=False).get_opcodes():
                    if tag in ("delete", "replace"):
                        removed.update(dw[i1:i2])
            else:
                human.update(words(r.get("final_version")))
    print(f"files used: {used} | AI tokens: {sum(ai.values())} | human tokens: {sum(human.values())}")
    return ai, human, ai_pids, ai_scen, removed, seen

# ------------------------------------------------------------------
def log_odds_table(ai, human, ai_pids, ai_scen, alpha0=500.0):
    vocab = sorted(set(ai) | set(human))
    n1, n2 = sum(ai.values()), sum(human.values())
    rows = []
    for w in vocab:
        y1, y2 = ai[w], human[w]
        a = alpha0 * (y1 + y2) / (n1 + n2)                     # informative prior from pooled corpus
        d1 = np.log((y1 + a) / (n1 + alpha0 - y1 - a))
        d2 = np.log((y2 + a) / (n2 + alpha0 - y2 - a))
        z = (d1 - d2) / np.sqrt(1 / (y1 + a) + 1 / (y2 + a))
        rows.append(dict(word=w, count_ai=y1, count_human=y2,
                         rate_ai_per_10k=1e4 * y1 / n1, rate_human_per_10k=1e4 * y2 / n2,
                         ratio=((y1 + 0.5) / n1) / ((y2 + 0.5) / n2), z=z,
                         n_participants=len(ai_pids[w]), n_scenarios=len(ai_scen[w])))
    return pd.DataFrame(rows).sort_values("z", ascending=False).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset"); ap.add_argument("--out", default="lexicon")
    ap.add_argument("--min-count", type=int, default=8)
    ap.add_argument("--min-participants", type=int, default=10)
    ap.add_argument("--min-scenarios", type=int, default=3)
    ap.add_argument("--min-ratio", type=float, default=3.0)
    ap.add_argument("--min-z", type=float, default=3.0)
    ap.add_argument("--exclude-pids", default="")
    a, _ = ap.parse_known_args(); os.makedirs(a.out, exist_ok=True)

    ai, human, pids, scen, removed, seen = collect(a.data, set(filter(None, a.exclude_pids.split(","))))
    tab = log_odds_table(ai, human, pids, scen)
    tab.to_csv(f"{a.out}/word_scores_all.csv", index=False)

    keep = tab[(tab.count_ai >= a.min_count) & (tab.n_participants >= a.min_participants) &
               (tab.n_scenarios >= a.min_scenarios) & (tab.ratio >= a.min_ratio) & (tab.z >= a.min_z)]
    keep.to_csv(f"{a.out}/ai_lexicon.csv", index=False)
    with open(f"{a.out}/ai_lexicon.txt", "w") as f:
        f.write("\n".join(keep.word))
    print(f"\nAI lexicon: {len(keep)} words -> {a.out}/ai_lexicon.txt")
    print(keep.head(40)[["word", "count_ai", "count_human", "ratio", "z", "n_participants"]].round(2).to_string(index=False))

    rem = pd.DataFrame([(w, seen[w], removed[w], removed[w] / seen[w]) for w in seen if seen[w] >= 10],
                       columns=["word", "times_in_drafts", "times_removed", "removal_rate"])
    rem.sort_values("removal_rate", ascending=False).to_csv(f"{a.out}/word_removal_rates.csv", index=False)
    print(f"Saved removal table -> {a.out}/word_removal_rates.csv")

if __name__ == "__main__":
    main()