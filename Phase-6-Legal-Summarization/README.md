# Phase 6: Abstractive Legal Summarization

[← Back to the main README](../README.md)

## Task
Summarize a judgment. Primary metric: BERTScore F1. Best official score: **0.8312** (rank 93, 6 valid submissions).

## Approach
The decoder-only model is trained on `[source] SUMMARY: [summary]` with the loss on the summary only. If a judgment is too long, sentences are ranked by TF-IDF similarity to the whole document and the best ones that fit in 535 tokens are kept in original order. Decoding is greedy with no repeated 4-gram and 224 to 448 new tokens.

## Notebooks and results
Each notebook is one experiment/submission. M1 to M4 are the Phase-1 backbones (M1 GPT-style, M2 LLaMA-style, M3 GQA + sliding window, M4 hybrid); all numbers are taken from the final report and the notebook outputs.

| Repository file | Original notebook | Backbone + method | Trainable / total params | Official BERTScore F1 |
|---|---|---|---|---|---|
| `submission_01.ipynb` | `phase6-model1_llama_lora.ipynb` | M2 LLaMA-style + LoRA | 1.12M / 52.16M | 0.8309 |
| `submission_02.ipynb` | `phase6-model2_gqa_lorma_decode.ipynb` | M3 GQA+SWA + LoRMA (decoding from a saved checkpoint) | 1.31M / 51.97M | 0.8207 |
| `submission_03.ipynb` | `phase6-model3_hybrid_houlsby.ipynb` | M4 hybrid + Houlsby adapters | 1.07M / 51.45M | **0.8312** |
| `submission_04.ipynb` | `phase6-model4_gpt_pfeiffer.ipynb` | M1 GPT-style + Pfeiffer adapters | 0.53M / 52.00M | 0.8248 |
| `submission_05.ipynb` | `phase6-model5_llama_qlora.ipynb` | M2 LLaMA-style + QLoRA | 1.12M / 52.16M | 0.8295 |
| `submission_06.ipynb` | `phase6-model6_hybrid_houlsby_kd.ipynb` | M4 distilled student | 38.58M / 38.58M | 0.8238 |

## Notes
- The three best runs (0.8312, 0.8309, 0.8295) are within 0.0017, so "Houlsby is best" is a very narrow call.
- All systems write shorter summaries than the references (279 to 374 words vs 677) and repeat more; BERTScore says little about completeness.
- `submission_02` only decodes from a saved LoRMA checkpoint, so it has no dev loss in the report.

## How to run
1. The notebooks were run on a free Kaggle Tesla T4 (16 GB) with mixed precision and seed 42. They contain `/kaggle/input/...` paths, so Kaggle is the easiest place to run them; elsewhere, edit the paths in the setup/config cell.
2. Attach the required inputs as Kaggle datasets: the legal tokenizer file (see [`tokenizer/`](../tokenizer/README.md)), and the Phase-1 checkpoint of the backbone listed in the table above (the best-validation checkpoint from the matching Phase-1 notebook).
3. The official data are loaded from the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. An `HF_TOKEN` secret is read if present; no secret is stored in the notebooks.
4. Run the cells in order. Training takes from about 3 minutes to a few hours depending on the notebook (see the report appendix for minutes per run).
5. The last cells write the Codabench submission file.
