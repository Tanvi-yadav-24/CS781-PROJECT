# Phase 4: Court Judgment Prediction

[← Back to the main README](../README.md)

## Task
Predict the binary outcome of a long judgment. Primary metric: macro-F1. Best official score: **0.6442** (rank 106, 6 valid submissions).

## Approach
Only the **tail** of each judgment is read (up to 6 chunks of 512 tokens), since the outcome is usually near the end. Each chunk is encoded by its last-token state, additive attention pools the chunks, and a class-weighted cross-entropy head predicts the label. The decision threshold is tuned on the dev set. Training uses a fixed random subset of 8,000 cases, while the whole dev set is scored.

## Notebooks and results
Each notebook is one experiment/submission. M1 to M4 are the Phase-1 backbones (M1 GPT-style, M2 LLaMA-style, M3 GQA + sliding window, M4 hybrid); all numbers are taken from the final report and the notebook outputs.

| Repository file | Original notebook | Backbone + method | Trainable / total params | Official macro-F1 |
|---|---|---|---|---|---|
| `submission_01.ipynb` | `phase4-model1_llama_lora.ipynb` | M2 LLaMA-style + LoRA | 0.69M / 51.73M | 0.5992 |
| `submission_02.ipynb` | `phase4-model2_gqa_lorma.ipynb` | M3 GQA+SWA + LoRMA | 0.79M / 51.45M | 0.5899 |
| `submission_03.ipynb` | `phase4-model3_hybrid_adapter.ipynb` | M4 hybrid + adapters | 0.66M / 51.05M | 0.6418 |
| `submission_04.ipynb` | `phase4-model4_gpt_partial.ipynb` | M1 GPT-style + partial fine-tuning | 12.74M / 51.61M | 0.6365 |
| `submission_05.ipynb` | `phase4-model5_llama_unfreeze.ipynb` | M2 + gradual unfreezing | 0.13M at start / 51.17M | 0.5955 |
| `submission_06.ipynb` | `phase4-model6_hybrid_adapter_kd.ipynb` | M4 distilled student | 38.71M / 38.71M | **0.6442** |

## Notes
- The top three runs (0.6442, 0.6418, 0.6365) are within 0.008 and are treated as tied; the clear benefit of the distilled student is speed (about 43 ms vs 76 ms per case).
- All tuned thresholds are below 0.5 (0.227 to 0.448). The dev threshold is tuned on the dev set, so dev scores are slightly optimistic.

## How to run
1. The notebooks were run on a free Kaggle Tesla T4 (16 GB) with mixed precision and seed 42. They contain `/kaggle/input/...` paths, so Kaggle is the easiest place to run them; elsewhere, edit the paths in the setup/config cell.
2. Attach the required inputs as Kaggle datasets: the legal tokenizer file (see [`tokenizer/`](../tokenizer/README.md)), and the Phase-1 checkpoint of the backbone listed in the table above (the best-validation checkpoint from the matching Phase-1 notebook).
3. The official data are loaded from the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. An `HF_TOKEN` secret is read if present; no secret is stored in the notebooks.
4. Run the cells in order. Training takes from about 3 minutes to a few hours depending on the notebook (see the report appendix for minutes per run).
5. The last cells write the Codabench submission file.
