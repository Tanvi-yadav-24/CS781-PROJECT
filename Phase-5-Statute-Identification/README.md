# Phase 5: Legal Statute Identification

[← Back to the main README](../README.md)

## Task
Assign a set of statute IDs (out of 100) to a document. Primary metric: macro-F1. Best official score: **0.2990** (rank 115, 6 valid submissions).

## Approach
Six 512-token chunks **spread evenly** over each document are encoded, pooled with attention, and one sigmoid per statute is trained with positively weighted BCE (weight capped at 20). A global threshold plus per-statute thresholds (for statutes with at least five dev positives) are tuned on dev, and every document gets at least its top statute. Training uses a random subset of 10,000 documents.

## Notebooks and results
Each notebook is one experiment/submission. M1 to M4 are the Phase-1 backbones (M1 GPT-style, M2 LLaMA-style, M3 GQA + sliding window, M4 hybrid); all numbers are taken from the final report and the notebook outputs.

| Repository file | Original notebook | Backbone + method | Trainable / total params | Official macro-F1 |
|---|---|---|---|---|---|
| `submission_01.ipynb` | `phase5-model1_llama_lora.ipynb` | M2 LLaMA-style + LoRA | 0.74M / 51.78M | 0.2824 |
| `submission_02.ipynb` | `phase5-model2_gqa_lorma.ipynb` | M3 GQA+SWA + LoRMA | 0.84M / 51.50M | 0.2635 |
| `submission_03.ipynb` | `phase5-model3_hybrid_adapter.ipynb` | M4 hybrid + adapters | 0.71M / 51.10M | 0.2790 |
| `submission_04.ipynb` | `phase5-model4_gpt_partial.ipynb` | M1 GPT-style + partial fine-tuning | 12.79M / 51.66M | 0.2743 |
| `submission_05.ipynb` | `phase5-model5_llama_qlora.ipynb` | M2 LLaMA-style + QLoRA | 0.74M / 51.78M | 0.2842 |
| `submission_06.ipynb` | `phase5-model6_llama_qlora_kd.ipynb` | M2 distilled student (teacher: QLoRA run) | 38.57M / 38.57M | **0.2990** |

## Notes
- The distilled student is the best run on both dev (0.3112) and the official test (0.2990) and runs about 2.2 times faster than its QLoRA teacher.
- Performance follows label frequency: the rarest statutes are much harder than the most frequent ones.

## How to run
1. The notebooks were run on a free Kaggle Tesla T4 (16 GB) with mixed precision and seed 42. They contain `/kaggle/input/...` paths, so Kaggle is the easiest place to run them; elsewhere, edit the paths in the setup/config cell.
2. Attach the required inputs as Kaggle datasets: the legal tokenizer file (see [`tokenizer/`](../tokenizer/README.md)), and the Phase-1 checkpoint of the backbone listed in the table above (the best-validation checkpoint from the matching Phase-1 notebook).
3. The official data are loaded from the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. An `HF_TOKEN` secret is read if present; no secret is stored in the notebooks.
4. Run the cells in order. Training takes from about 3 minutes to a few hours depending on the notebook (see the report appendix for minutes per run).
5. The last cells write the Codabench submission file.
