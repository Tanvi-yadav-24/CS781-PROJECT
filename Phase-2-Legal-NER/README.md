# Phase 2: Legal Named Entity Recognition

[← Back to the main README](../README.md)

## Task
Find entity spans (start, end, label) of 12 types in a judgment. A span counts only if offsets and label match exactly. Primary metric: strict macro-F1. Best official score: **0.3799** (rank 107, 6 valid submissions).

## Approach
Token classification over 25 BIO tags on 512-token windows with 128 overlap (each token is predicted in the window where it has the most left context). The Phase-1 backbone is adapted with one of the methods below, and arg-max tags are merged into spans. 60 labelled documents are used for training and 10 for local validation.

## Notebooks and results
Each notebook is one experiment/submission. M1 to M4 are the Phase-1 backbones (M1 GPT-style, M2 LLaMA-style, M3 GQA + sliding window, M4 hybrid); all numbers are taken from the final report and the notebook outputs.

| Repository file | Original notebook | Backbone + method | Trainable / total params | Official strict macro-F1 |
|---|---|---|---|---|---|
| `submission_01.ipynb` | `phase-2-5th-mdel-lora-llama.ipynb` | M2 LLaMA-style + LoRA | 0.57M / 51.61M | 0.3597 |
| `submission_02.ipynb` | `phase2-model1_llama_full.ipynb` | M2 LLaMA-style + full fine-tuning | 51.05M / 51.05M | **0.3799** |
| `submission_03.ipynb` | `phase2-model2_gqa_lorma.ipynb` | M3 GQA+SWA + LoRMA | 0.67M / 51.33M | 0.3407 |
| `submission_04.ipynb` | `phase2-model3_hybrid_adapter.ipynb` | M4 hybrid + bottleneck adapters | 0.54M / 50.93M | 0.3472 |
| `submission_05.ipynb` | `phase2-model4_gpt_bitfit.ipynb` | M1 GPT-style + BitFit | 0.07M / 51.49M | 0.2675 |
| `submission_06.ipynb` | `phase2-model6_llama_full_kd.ipynb` | M2 distilled student (4 layers) | 38.40M / 38.40M | 0.3309 |

## Notes
- The teacher of the distilled student (`submission_06`) is the full fine-tuned M2 model of `submission_02`.
- On the 10-document local validation set full fine-tuning and LoRA are tied (0.3131 vs 0.3142), so the official gap should not be over-read.
- BIO tags cannot represent overlapping entities, which is a limitation of this design.

## How to run
1. The notebooks were run on a free Kaggle Tesla T4 (16 GB) with mixed precision and seed 42. They contain `/kaggle/input/...` paths, so Kaggle is the easiest place to run them; elsewhere, edit the paths in the setup/config cell.
2. Attach the required inputs as Kaggle datasets: the legal tokenizer file (see [`tokenizer/`](../tokenizer/README.md)), and the Phase-1 checkpoint of the backbone listed in the table above (the best-validation checkpoint from the matching Phase-1 notebook).
3. The official data are loaded from the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. An `HF_TOKEN` secret is read if present; no secret is stored in the notebooks.
4. Run the cells in order. Training takes from about 3 minutes to a few hours depending on the notebook (see the report appendix for minutes per run).
5. The last cells write the Codabench submission file.
