# Phase 7: Prior Case Retrieval

[← Back to the main README](../README.md)

## Task
For each query case, return the 7 most relevant prior cases from a candidate pool. Primary metric: micro-F1@7. Best official score: **0.2640** (rank 110, 6 valid submissions).

## Approach
A bi-encoder: four 512-token chunks spread over each case are encoded, the chunk vectors are averaged, projected to 256 dimensions and normalised. Training uses InfoNCE (temperature 0.05) with one relevant case, one BM25 hard negative and in-batch negatives. At test time the cosine score is fused with BM25 (own implementation, k1 = 1.2, b = 0.75) using a weight of 0.1 tuned on dev, and the top 7 are returned.

## Notebooks and results
Each notebook is one experiment/submission. M1 to M4 are the Phase-1 backbones (M1 GPT-style, M2 LLaMA-style, M3 GQA + sliding window, M4 hybrid); all numbers are taken from the final report and the notebook outputs.

| Repository file | Original notebook | Backbone + method | Trainable / total params | Official micro-F1@7 |
|---|---|---|---|---|---|
| `submission_01.ipynb` | `phase7-mdel1_llama_lora.ipynb` | M2 LLaMA-style + LoRA | 0.69M / 51.73M | 0.2522 |
| `submission_02.ipynb` | `phase7-model2_gqa_lorma.ipynb` | M3 GQA+SWA + LoRMA | 0.79M / 51.45M | 0.2416 |
| `submission_03.ipynb` | `phase7-model3_hybrid_adapter.ipynb` | M4 hybrid + adapters | 1.19M / 51.58M | 0.2439 |
| `submission_04.ipynb` | `phase7-model4_gpt_partial.ipynb` | M1 GPT-style + partial fine-tuning | 12.74M / 51.61M | **0.2640** |
| `submission_05.ipynb` | `phase7-model5_llama_qlora.ipynb` | M2 LLaMA-style + QLoRA | 0.69M / 51.73M | 0.2475 |
| `submission_06.ipynb` | `phase7-model6_gpt_partial_kd.ipynb` | M1 distilled student (teacher: partial run) | 39.00M / 39.00M | 0.2475 |

## Notes
- Fusing BM25 (weight 0.1) improved all six runs on dev by 0.02 to 0.03 (BM25 alone scores 0.0989 on dev). Dense-only runs were not submitted, so this gain is measured on dev only.
- On the official test set the whole spread is only 0.022, and the best run differs from the best on dev.

## How to run
1. The notebooks were run on a free Kaggle Tesla T4 (16 GB) with mixed precision and seed 42. They contain `/kaggle/input/...` paths, so Kaggle is the easiest place to run them; elsewhere, edit the paths in the setup/config cell.
2. Attach the required inputs as Kaggle datasets: the legal tokenizer file (see [`tokenizer/`](../tokenizer/README.md)), and the Phase-1 checkpoint of the backbone listed in the table above (the best-validation checkpoint from the matching Phase-1 notebook).
3. The official data are loaded from the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. An `HF_TOKEN` secret is read if present; no secret is stored in the notebooks.
4. Run the cells in order. Training takes from about 3 minutes to a few hours depending on the notebook (see the report appendix for minutes per run).
5. The last cells write the Codabench submission file.
