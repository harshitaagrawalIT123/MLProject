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
