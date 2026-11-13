# Phase 1: Legal Language Model Pretraining

[← Back to the main README](../README.md)

## Task
Continue the opening of a legal document with exactly 128 words. Primary metric: BERTScore F1 (ROUGE-L and BLEU-4 are secondary). Best official score: **0.8291** (rank 89, 5 valid submissions).

## Approach
Four decoder-only language models of about 50M parameters (width 512, context 1024, tied embeddings) were trained from scratch on the official legal text (191M training tokens, one pass), with the same data and recipe, so that differences come from the architecture. M1 is GPT-style, M2 LLaMA-style, M3 LLaMA-style with grouped-query attention and a 256-token sliding window (9 layers), and M4 an attention + gated causal convolution hybrid. Decoding uses sampling (temperature 0.8, top-k 50, top-p 0.95) with the end-of-document token suppressed, and every output is cut to 128 words.

## Notebooks and results
Each notebook is one experiment/submission. M1 to M4 are the Phase-1 backbones (M1 GPT-style, M2 LLaMA-style, M3 GQA + sliding window, M4 hybrid); all numbers are taken from the final report and the notebook outputs.

| Repository file | Original notebook | Backbone / tokenizer | Params | Official BERTScore F1 |
|---|---|---|---|---|---|
| `submission_01.ipynb` | `model1-phase1-bpe (1).ipynb` | M1 GPT-style, legal BPE | 51.48M | 0.8260 |
| `submission_02.ipynb` | `model1-phase1-gpt.ipynb` | M1 GPT-style, **GPT-2 tokenizer** | 51.48M | 0.8252 |
| `submission_03.ipynb` | `phase1-model2-llama (1).ipynb` | M2 LLaMA-style, legal BPE | 51.04M | 0.8287 |
| `submission_04.ipynb` | `phase1-model3-gqa-swa (1).ipynb` | M3 GQA + sliding window, legal BPE | 50.66M | **0.8291** |
| `submission_05.ipynb` | `phase1-model4-hybrid.ipynb` | M4 hybrid, legal BPE | 50.39M | 0.8281 |

## Notes
- Validation losses on the legal BPE are 2.7752 (M1), 2.3844 (M2), 2.4079 (M3) and 2.3898 (M4). The GPT-2-tokenizer run (`submission_02`) shows a validation loss of 2.6569 in its notebook output, but a loss per token is not comparable across tokenizers.
- The four legal-BPE models are within 0.0031 BERTScore of each other, so the report treats them as practically tied.
- The notebooks load the tokenizer from a `tokenizer*.json` file attached as a Kaggle dataset. If none is found they fall back to the GPT-2 tokenizer, so check the first lines of the output (`Loaded tokenizer: ...` or `Using GPT-2 BPE tokenizer`) to see which one was used.

## How to run
1. The notebooks were run on a free Kaggle Tesla T4 (16 GB) with mixed precision and seed 42. They contain `/kaggle/input/...` paths, so Kaggle is the easiest place to run them; elsewhere, edit the paths in the setup/config cell.
2. Attach the required inputs as Kaggle datasets: the legal tokenizer file (see [`tokenizer/`](../tokenizer/README.md)); Phase 1 itself needs no earlier checkpoint.
3. The official data are loaded from the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. An `HF_TOKEN` secret is read if present; no secret is stored in the notebooks.
4. Run the cells in order. Training takes from about 3 minutes to a few hours depending on the notebook (see the report appendix for minutes per run).
5. The last cells write the Codabench submission file.
