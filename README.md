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

## Summary Pipeline

1. **Ingest the released dataset.** Load control, LLM-draft and post-edited text for every participant and task, plus the keystroke logs (`logs/`). Participants are split by person into 57 train and 24 held-out test (the 57 are further split into 45 train and 12 validation).

2. **Module A: statistical diagnostics (A1, A2, A3).**
   - **A1, similarity vs. length:** Editing amount (WER) is a stronger predictor of style shift than document length. Repeated-measures correlation: WER vs. self-similarity r = 0.316, vs. LLM-similarity r = -0.700 (both p < .001). Length has no clear independent effect on self-similarity (β = 0.0002, p = .19) while WER does (β = 0.105, p < .001). LUAR scores rise with the number of input chunks, so short-span scores are not comparable with document-level scores.
   - **A2, vocabulary diversity:** TTR, MTLD and Yule's K show no significant link with editing effort or similarity improvement (N = 81, Holm-corrected, including partial correlations controlling for token count).
   - **A3, held-out validation:** After fixing the paired draft-vs-final comparison, the mean similarity-to-own-style change on the 57 training participants is +0.054 (55/57 positive). On the 24 held-out participants it is +0.045 (22/24 positive, Wilcoxon p ≈ 4e-5), consistent with the paper's H1a.

3. **Module C: AI-content classifier.** Fine-tune RoBERTa on labeled pairs, with the AI-probability cutoff (0.80) chosen on validation participants. On the 24 test participants (1,895 sentences): accuracy 0.921, AI-class F1 0.948 (recall 0.967), human-class F1 0.838 (recall 0.791). It also underlies Module E.

4. **Modules B and C: interactive editing tool.** Recommendations, live feedback and process logging, piloted on volunteer writers performing the paper's own tasks.
   - **Module B (offline evaluation of "which sentences get edited"):** On participant-disjoint data the non-personalized model reaches AUROC 0.555 against 0.503 for random, so it adds little. With a warm start (some of the user's own history), the personalized model reaches AUROC 0.820 against 0.530 non-personalized; P@3 barely moves (0.368 vs. 0.364). A 30-word AI lexicon is derived from word-removal rates.
   - **Module E (document-level AI score from Module C):** On 24 test participants (144 documents), the calibrated estimate of "% still from the LLM draft" has MAE 9.0 points on all documents (always guessing the mean: 42.1), r = 0.94 overall and 0.84 on post-edited documents. Calibration cuts the false-alarm bias on human-written text from 16.5% to 10.8%. Sentence-level AUROC for copied vs. rewritten text is 0.949.

5. **Module D: style-conditioned generation.** Reduce the editing burden at the source by conditioning generation on the writer's style. Arms compared: plain baseline, zero-shot, RAG with random / attention / rerank retrieval, and a style card (24 participants, 192 items). Similarity to the writer's own reference text (LUAR) improves over the baseline most with `rag_random` (+0.030, Holm p = 0.011). Other arms are smaller or not significant, and the style card does not improve LUAR (`luar_ref`) similarity. The generated text's AI-ness rate drops from about 5.2 (baseline) to about 2.8–3.2 for all personalized arms.

6. **Module G: edit-process analysis.** Replay the released keystroke logs (about 216,000 edit events over 486 tasks from 81 participants) to reconstruct how people edit an LLM draft. No new data collection is needed. Replay rebuilds the final text exactly for 296 of 324 tasks (26 more match except for spaces).
   - **Speed:** about 41 s before the first edit, about 100 s of active editing, about 83 characters per minute in about 7 short bursts, touching about 22% of sentences.
   - **Order:** people edit top to bottom (median rank correlation 1.0, 90% of tasks above 0). Module C's sentence flags do not predict what gets fixed first, because Module C scores almost every draft sentence about 0.95.
   - **Prediction:** edit amount (WER) alone explains the most style recovery (out-of-sample R² = 0.33). Process features add nothing measurable.
   - **Personal style:** how much people edit is a stable personal habit (ICC about 0.54), while how soon they start is not (0.32).

   Findings feed back into the design of Modules B, C and D.

7. **Evaluation.** Evaluate all modules against the paper's original H1, H2 and H3 metrics to test whether the gaps they report shrink.
parts run on a CPU, though embedding with LUAR and scoring sentences with RoBERTa is slow without one.

