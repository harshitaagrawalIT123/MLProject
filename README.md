# Closing the Style Gap: Measuring, Explaining, and Guiding the Post-Editing of LLM-Generated Text

Course project (DS501 – Machine Learning, IIT Bhilai). We build on the dataset and findings of
*"Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style"*
(Baumler et al., ACL 2026). That paper shows that people who post-edit an LLM draft move closer to their
own writing style, but the result still sounds more like the LLM than like the person, and people cannot
tell. The authors released data but no code or tools. This repository adds:

1. **An audit of the style metric (LUAR)** and a held-out replication check (Modules A1–A3).
2. **Tools that act during editing**: an edit recommender (B), a sentence-level AI-likeness classifier (C),
   and a document-level "% still from the LLM draft" estimator (E).
3. **A style-conditioned drafting pipeline** (D) — built, real-generation evaluation still pending.

## Team

| Member | Roll no. | Modules |
|---|---|---|
| Dange Pooja Sanjay | B24DS006 | A1, B, E |
| Samarth Goyal | B24DS026 | A3, C, D |
| Harshita Agrawal | B24DS501 | A2, B, D |

## Results at a glance

| Module | Question | Main result |
|---|---|---|
| **A1** | Does LUAR similarity depend on document length or on how much was edited? | Edit amount (WER) predicts self-similarity (β = 0.105, p < .001; repeated-measures r = 0.32). Length does not (β = 0.0002, p = .19). LUAR scores also rise with input size: 0.61 at 20 words → about 0.73 (60-word chunks) or 0.78 (30-word chunks) at 100 words. |
| **A2** | Do writers with richer vocabulary need less editing or improve more? | No evidence. No vocabulary metric (TTR, MTLD, Yule's K) is significant after Holm correction. Closest: MTLD vs editing effort, ρ = 0.20, p(Holm) = .43. |
| **A3** | Does the paper's H1a gain replicate on a participant-disjoint split? | Not with our baseline pipeline: mean Δ self-similarity = −0.026 (median −0.023), 40 of 57 participants negative. Suggests the effect is sensitive to how texts are embedded. |
| **B** | Can we rank which sentences a given user would edit? | New users: AUROC 0.555 vs 0.503 random. Returning users: personalized 0.820 vs 0.530 non-personalized; P@3 0.368 vs 0.364, so the within-document gain is small. |
| **C** | Can we flag AI-sounding sentences? | RoBERTa-base on 24 unseen participants (1,895 sentences): accuracy 0.90, macro-F1 0.87 (human F1 0.80, AI F1 0.93). |
| **D** | Is a style-conditioned draft closer to the user's style than a zero-shot draft? | Pipeline built and tested end to end. **Evaluation on real LLM generations is pending; no result yet.** |
| **E** | What % of a finished document still comes from the LLM draft? | Post-edited test docs: MAE 8.1 points (95% CI 5.8–10.8), r = 0.80. Guessing the post-edited average gives 13.7; guessing the overall average gives 32.8. Human-only docs are still scored about 19% AI. |

These are small-sample results (81 participants, one LLM). Negative and small results are reported as they are.

## Dataset

The 81 participant logs are from the original study
(<https://github.com/ctbaumler/personal_style_postedit>) and are stored in `logs/`.
Each participant wrote 2 texts alone (**control**) and post-edited 4 LLM-generated drafts (**treatment**):
486 usable responses in total (162 control, 324 treatment). One participant has a seventh, empty
response entry, which our code ignores. Every log contains the LLM draft (`model_generation`), the final
text (`final_version`), the content details, and the edit actions. The paper's original field description is
in `task_A1/personal_style_postedit/README.md`.

**Shared split.** Participants are split into 57 (fit/train) and 24 (held-out test), listed in
`task_A3/train_files.txt` and `task_A3/test_files.txt`. Modules C and E reuse this split.
Inside the 57, Module C draws 45 train and 12 validation participants (`GroupShuffleSplit`, seed 42);
Module E repeats that draw to find the same 12 validation participants.

## Repository structure

```
.
├── logs/                         81 participant logs (original dataset)
├── task_A1/                      length vs edit-amount audit
│   └── 2_A1_similarity_vs_length.ipynb
├── task_A2/                      vocabulary richness
│   ├── a2.py                     main script
│   ├── Untitled4.ipynb
│   └── A2_*.csv                  correlations, partial correlations, per-participant and per-response tables
├── task_A3/                      held-out validation of the paper's H1a
│   ├── taskA3_reproduce_baseline.ipynb
│   └── train_files.txt, test_files.txt     the 57 / 24 participant split
├── task_B/                       personalized edit recommender
│   ├── B.py                      builds the AI-word lexicon
│   ├── moduleB.py                alignment, features, model, evaluation, recommend()
│   ├── task_B.ipynb
│   ├── lexicon/                  ai_lexicon.txt/.csv, word scores, removal rates
│   └── results_b/                evaluation CSVs, trained model, user profiles, top substitutions
├── task_C/                       sentence-level AI-likeness classifier
│   ├── taskC_ai_likenes_feedback.ipynb
│   └── roberta_ai_human_model/   fine-tuned model and evaluation.json (model file is Git LFS, ~500 MB)
├── task_D/                       style-conditioned drafting (to be added: ModuleD.py, Dencoder.py)
└── task_E/                       document-level % AI
    ├── E_document_level_ai.ipynb
    └── task_E_results/           results_table.csv, calibration.json, scored_documents.csv, fig_calibration.png
```

## Setup

```bash
pip install pandas numpy scipy scikit-learn statsmodels pingouin jiwer joblib \
            torch transformers datasets nltk sentence-transformers matplotlib seaborn
python -c "import nltk; nltk.download('punkt_tab')"
```

The trained classifier (`task_C/roberta_ai_human_model/model.safetensors`) is stored with **Git LFS**:

```bash
git lfs install
git clone https://github.com/harshitaagrawalIT123/MLProject
```

Some notebooks contain a hard-coded local path near the top (for example `path = r"D:\..."` in Module C).
Change it to your local folder before running. A GPU is helpful for C and D; A, B and E's analysis
parts run on a CPU, though embedding with LUAR and scoring sentences with RoBERTa is slow without one.

## Modules

### A1 – Similarity vs. length and edit amount
*Notebook:* `task_A1/2_A1_similarity_vs_length.ipynb`

- **Edit amount** = word error rate (WER) between the LLM draft and the final text (`jiwer`); `words_changed` is the same as a raw count.
- **Embeddings:** LUAR-CRUD, texts cut into 60-word chunks. A person's own style is the average vector of their two control texts.
- **Scores:** `self_sim` (final text vs own control profile) and `llm_sim` (final text vs the original LLM draft).
- **Statistics:** Pearson/Spearman, repeated-measures correlation (`pingouin`), and OLS `self_sim ~ n_words + wer + scenario` with errors clustered by participant.
- **Truncation test:** keep the first 20/40/60/80/100 words of each text and re-score, with 60-word and 30-word chunks.
- **Caveats:** documents are all about 160 words (SD 22), so there is little length variation to detect an effect. `llm_sim` falling with WER is partly mechanical. The result is a correlation, not a causal claim.

### A2 – Similarity vs. vocabulary richness
*Script:* `task_A2/a2.py` (`python a2.py --data logs --out results [--no-luar]`)

Computes TTR, MTLD and Yule's K on each participant's control writing and correlates them with
(i) similarity improvement after editing and (ii) editing effort (WER), using Spearman correlation with bootstrap
confidence intervals (5,000 resamples), partial correlations and Holm correction. Embeddings: LUAR-MUD.
Outputs: `A2_correlations.csv`, `A2_partial_correlations.csv`, `A2_participant_level.csv`, `A2_response_level.csv`.
With 81 participants and strict correction, "no significant effect" does not prove there is none.

### A3 – Held-out validation of the paper's H1a
*Notebook:* `task_A3/taskA3_reproduce_baseline.ipynb`

For each participant, a reference style vector is built from the two control texts (concatenated). For each of the
four treatment documents we compute `similarity(final, reference) − similarity(LLM draft, reference)` with LUAR-MUD.
A positive value means the editing moved the text toward the person's own writing. Result: mean −0.026, 17 positive and
40 negative of the 57 fit-set participants. This is *our baseline pipeline's* result, not proof that the paper is wrong;
differences in chunking and embedding setup likely matter. A1 and A3/A2 use different LUAR checkpoints (CRUD vs MUD)
and different ways of building the control profile, which is worth checking in any follow-up.

### B – Personalized edit recommendation
*Files:* `task_B/B.py`, `task_B/moduleB.py`, `task_B/task_B.ipynb`

```bash
cd task_B
python B.py --data logs --out lexicon                      # build the AI-word lexicon
python moduleB.py --data logs --out results_b --lexicon lexicon/ai_lexicon.txt
```

1. Align each draft sentence with the final text using `difflib`. A sentence counts as **edited** if its similarity ratio falls below 0.90 (677 of 3,435 draft sentences, 19.7%).
2. Label edit types by rules: em dash removed, contraction added, shortened, AI word removed, simplified words, first person added, exclamation added, rewritten, and so on.
3. Build a data-driven **AI lexicon** (31 words): words much more frequent in LLM drafts than in control writing (log-odds with a prior), kept only if seen across several participants and scenarios.
4. Build a **per-user edit profile** (how often each user makes each edit type) and cluster editors with k-means.
5. Train a balanced logistic regression. The personalized model adds "sentence trigger × user habit" features and the user's overall edit rate.
6. Evaluate two ways: **participant-disjoint** (new users, grouped cross-validation) and **warm start** (the same person appears in train and test, but a test document's profile is built only from that person's other documents).

Metrics: AUROC, P@3, Recall@3, NDCG@3. The AUROC gain for returning users is large, but P@3 barely moves. The
personalized features include the user's overall edit rate, so much of the AUROC gain likely reflects *who edits a lot*
rather than *which sentence to edit*. `recommend()` in `moduleB.py` returns ranked sentences with suggestions.

### C – Sentence-level AI-likeness classifier
*Notebook:* `task_C/taskC_ai_likenes_feedback.ipynb`

- **Data:** 4,441 sentences from the 57 fit participants. AI (label 1) = sentences of the LLM drafts (3,339); human (label 0) = sentences of the participants' control writing (1,102).
- **Model:** `roberta-base`, sequence classification, class-weighted loss (human class weighted about 2.0 vs about 0.67 for AI), max 128 tokens. 45 train / 12 validation participants.
- **Cutoff:** 0.80 AI probability, chosen on validation participants and applied once to the test set.
- **Test (24 unseen participants, 1,895 sentences):** accuracy 0.90, macro-F1 0.87; confusion matrix (human, AI) = [[380, 108], [78, 1329]].
- **Note:** 74% of test sentences are AI, so always predicting "AI" would score about 74% accuracy. Use macro-F1 and per-class F1. About 22% of human sentences are flagged as AI.
- The model was trained on pure LLM vs pure human text. Its behaviour on post-edited text is tested in Module E (section 10).
- Not implemented: the browser red/yellow/green heat-map interface, training on RAID/HC3, and testing other LLMs. The sentence probabilities are what such an interface would use.

### D – Cross-topic, attention-based style capture and style-conditioned drafting
*Files (to add under `task_D/`):* `ModuleD.py`, `Dencoder.py`

Question: if a draft is conditioned on the user's own writing on a *different* topic, is the **unedited** draft already closer to their style than a zero-shot draft?

- **Setup:** per participant, one control text is the style *source* and the other control text is the held-out *reference* used only for scoring (both directions used). The target topic differs from both.
- **Style encoder:** parameter-free scaled dot-product attention over sentence embeddings, plus an optional trained encoder (one Transformer layer, learned-query attention pooling, contrastive loss: same author on different topics = positive pair). `Dencoder.py` trains and evaluates it with participant-disjoint retrieval metrics.
- **Arms (same generator, only conditioning differs):** `zero_shot`, `style_card` (measured stylometric description), `rag_attn` (source sentences picked by attention), `rag_random` (ablation: random sentences), `rag_rerank` (best of several candidates), optional `rag_edit` (uses Module B edit habits). The paper's own GPT-o4-mini draft is a reference baseline.
- **Scoring:** LUAR cosine to the held-out control text (primary), plus CISR cosine, stylometric distance and distance to the user's final text.

```bash
python ModuleD.py --stage all --data logs --out results_d --backend hf --model Qwen/Qwen2.5-3B-Instruct
python ModuleD.py --stage evaluate --out results_d
python Dencoder.py --data logs --out results_d
```

**Status:** pipeline complete; no results from real generations yet. `results_mock/` contains outputs of a smoke test that used
placeholder embedders and a fake generator to check the code runs. **They have no scientific meaning and must not be read as results.**

### E – Document-level AI-content quantification
*Notebook:* `task_E/E_document_level_ai.ipynb`

1. **Answer key:** a word of the final text is "AI" if it lies in a run of at least 3 consecutive words shared with the LLM draft (`difflib`). Control texts get 0%.
2. **Scoring:** Module C's classifier scores each sentence; a sentence is AI if its probability is at least 0.8. The main document score is the **length-weighted share of AI sentences** (chosen before looking at results).
3. **No leakage:** only the 12 validation and 24 test participants are scored (the classifier trained on the other 45).
4. **Calibration:** a straight line fitted on the 12 validation participants (`true% = −10.7 + 1.09 × predicted%`), applied to the 24 test participants. 95% intervals from 1,000 participant-level bootstraps.
5. **Extra check (section 10):** the classifier separates "copied" from "rewritten" sentences in post-edited text with AUROC 0.937 (770 copied, 89 rewritten sentences; 30% of rewritten sentences are still flagged as AI).

Error is small for lightly edited documents (about 4 points when over 95% of the text is from the draft) and large for heavily
edited ones (about 23 points when under 50%, only 9 test documents). Raw scores are slightly better than calibrated ones on post-edited
text; calibration mainly reduces over-flagging of human-only text (error 19.2 → 13.9). Treat the % as a rough estimate, not as proof of AI use.

## Limitations

- One dataset (81 participants), one LLM (GPT-o4-mini) and a small set of writing tasks.
- A1 uses LUAR-CRUD with 60-word chunks; A2 and A3 use LUAR-MUD. Style profiles are built differently (averaged vectors in A1, concatenated text in A3).
- A3's headline number is computed on the 57-participant fit set.
- B: new-user performance is close to chance; the lexicon was built from the released data and should be built fold-wise for a strict evaluation (`B.py --exclude-pids`).
- C: false alarms on human text (about 22%) and no test on other LLMs.
- E: the answer key is a copy-overlap proxy, and most documents are 85% or more AI, so the range of true values is narrow.
- D: no results from real generations yet.
- Planned in our project proposal but **not implemented**: voice/prosody-informed style capture (Module F) and the keystroke edit-process logger and analysis (Module G).

## Reference

Baumler, C., Bao, C., Nghiem, H., Yang, X., Carpuat, M., and Daumé III, H. (2026).
*Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style.*
Proceedings of the 64th Annual Meeting of the ACL, pp. 43867–43895.
Dataset: <https://github.com/ctbaumler/personal_style_postedit>
