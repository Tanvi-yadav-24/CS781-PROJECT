# Phase 3: Rhetorical Role Prediction

[← Back to the main README](../README.md)

## Task
Give each sentence of a Corporate Law or Information Technology judgment one of 13 rhetorical roles. Primary metric: macro-F1. Best official score: **0.5088** (rank 53, 5 valid submissions).

## Approach
Each document becomes `sent_1 <sep> sent_2 <sep> ...`. The backbone reads 512-token windows (128 overlap); the hidden state at each separator is the sentence vector, and a BiGRU over the sentence vectors adds right-hand context. Classes are weighted by inverse square-root frequency, both domains are trained together, and the best epoch is chosen on the official dev set. Training documents are cut into segments of up to 3,072 tokens, but the backbone itself always sees windows of 512 tokens.

## Notebooks and results
Each notebook is one experiment/submission. M1 to M4 are the Phase-1 backbones (M1 GPT-style, M2 LLaMA-style, M3 GQA + sliding window, M4 hybrid); all numbers are taken from the final report and the notebook outputs.

| Repository file | Original notebook | Backbone + method | Trainable / total params (incl. BiGRU head) | Official macro-F1 |
|---|---|---|---|---|---|
| `submission_01.ipynb` | `phase3-mode5_llama_qlora.ipynb` | M2 + QLoRA | not counted (see note) | not in the report |
| `submission_02.ipynb` | `phase3-model1_llama_lora.ipynb` | M2 LLaMA-style + LoRA | 1.75M / 52.79M | 0.4701 |
| `submission_03.ipynb` | `phase3-model2_gqa_lorma.ipynb` | M3 GQA+SWA + LoRMA | 1.84M / 52.51M | 0.4850 |
| `submission_04.ipynb` | `phase3-model3_hybrid_rslora.ipynb` | M4 hybrid + rsLoRA | 1.74M / 52.13M | 0.4464 |
| `submission_05.ipynb` | `phase3-model4_gpt_ia3.ipynb` | M1 GPT-style + IA3 | 1.22M / 52.69M | 0.4126 |
| `submission_06.ipynb` | `phase3-model6_gqa_lorma_kd.ipynb` | M3 distilled student | 40.77M / 40.77M | **0.5088** |

## Notes
- **Note on `submission_01` (QLoRA):** this notebook is kept in the repository for completeness. It is not one of the 40 valid submissions counted in the report and does not appear in the report's tables or figures; please refer to the report and the Codabench page for the official status of the Phase-3 submissions.
- The distilled student has the best official score but is below its teacher (LoRMA) on the official dev split (0.4320 vs 0.4602). Both sets have only 10 documents, so the report does not claim that distillation helps here.
- PrecedentOverruled (3 training sentences) and Dissent are never predicted correctly.

## How to run
1. The notebooks were run on a free Kaggle Tesla T4 (16 GB) with mixed precision and seed 42. They contain `/kaggle/input/...` paths, so Kaggle is the easiest place to run them; elsewhere, edit the paths in the setup/config cell.
2. Attach the required inputs as Kaggle datasets: the legal tokenizer file (see [`tokenizer/`](../tokenizer/README.md)), and the Phase-1 checkpoint of the backbone listed in the table above (the best-validation checkpoint from the matching Phase-1 notebook).
3. The official data are loaded from the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. An `HF_TOKEN` secret is read if present; no secret is stored in the notebooks.
4. Run the cells in order. Training takes from about 3 minutes to a few hours depending on the notebook (see the report appendix for minutes per run).
5. The last cells write the Codabench submission file.
