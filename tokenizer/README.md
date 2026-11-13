# Tokenizer

[← Back to the main README](../README.md)

The report describes a legal-domain byte-level BPE tokenizer (vocabulary 50,257, end-of-document token id 0) built from the Phase-1 training files. All runs use it except the first Phase-1 M1 run (`Phase-1-Language-Model/submission_02.ipynb`), which uses the GPT-2 tokenizer.

## What is in this repository
- The notebooks **load** the tokenizer from a `tokenizer*.json` file that was attached as a Kaggle dataset (the outputs show `Loaded tokenizer: .../tokenizer (1).json`, vocabulary 50,257, EOS id 0).
- The tokenizer file itself and the code that trained it are **not included** in this repository. To reproduce the runs, place the tokenizer file in this folder as `tokenizer.json` and attach it to the notebooks as a dataset.
- If no tokenizer file is found, the Phase-1 notebooks silently fall back to the GPT-2 tokenizer, which changes the vocabulary and end-of-document id. Always check the first lines of the output.
