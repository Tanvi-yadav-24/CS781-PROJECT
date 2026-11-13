# CS781 Legal AI Capstone: In-House Legal Intelligence System

**Author:** Tanvi Yadav (roll number 2401094)  
**Course:** CS781 Capstone Project, Indian Institute of Technology Kanpur  
**Report:** [`CS781-CP-Tanvi-Yadav-241094-report.pdf`](CS781-CP-Tanvi-Yadav-241094-report.pdf)

A confidential, self-hosted legal NLP pipeline built under the capstone rules: no pretrained weights and only the official Indian legal data (`Exploration-Lab/CS-781-Capstone` on Hugging Face). Phase 1 trains four decoder-only language models of about 50M parameters from scratch; Phases 2 to 7 adapt those checkpoints to six legal tasks, with every fine-tuning method (LoRA, rsLoRA, LoRMA, QLoRA, IA3, BitFit, adapters, gradual unfreezing, knowledge distillation) written in plain PyTorch. All runs used a free Kaggle Tesla T4.

## Results (best official run per phase)

Ranks were recorded on 9 November 2026. Metrics differ by task, so scores must not be compared across phases.

| Phase | Task | Main approach | Metric | Best score | Rank |
|---|---|---|---|---:|---:|
| 1 | [Legal LM pretraining](Phase-1-Language-Model/) | Four ~50M models trained from scratch (GPT-style, LLaMA-style, GQA + sliding window, hybrid) | BERTScore F1 | **0.8291** | 89 |
| 2 | [Legal NER](Phase-2-Legal-NER/) | BIO token classification on 512-token windows; full / LoRA / LoRMA / adapters / BitFit / distillation | Strict macro-F1 | **0.3799** | 107 |
| 3 | [Rhetorical roles](Phase-3-Rhetorical-Role/) | Sentence vectors at separators + BiGRU; LoRA / LoRMA / rsLoRA / IA3 / distillation | Macro-F1 | **0.5088** | 53 |
| 4 | [Judgment prediction](Phase-4-Judgment-Prediction/) | Tail of the judgment (6x512), attention pooling, tuned threshold | Macro-F1 | **0.6442** | 106 |
| 5 | [Statute identification](Phase-5-Statute-Identification/) | Six evenly spread chunks, one sigmoid per statute, per-statute thresholds | Macro-F1 | **0.2990** | 115 |
| 6 | [Legal summarization](Phase-6-Legal-Summarization/) | `[source] SUMMARY:` continuation, TF-IDF source selection, adapters / LoRA | BERTScore F1 | **0.8312** | 93 |
| 7 | [Prior case retrieval](Phase-7-Prior-Case-Retrieval/) | Dense bi-encoder fused with BM25 (weight 0.1) | Micro-F1@7 | **0.2640** | 110 |

Many gaps between methods are small compared with the noise of the small evaluation sets, every configuration was run once (seed 42), and no single adaptation method won in every phase. See the report for the full run tables, error analysis and limitations.

## Repository structure

```text
CS781-Capstone-Legal-Language-Model/
├── README.md
├── CS781-CP-Kavya-Jain-240534-report.pdf
├── tokenizer/
│   └── README.md
├── Phase-1-Language-Model/        submission_01 ... submission_05
├── Phase-2-Legal-NER/             submission_01 ... submission_06
├── Phase-3-Rhetorical-Role/       submission_01 ... submission_06
├── Phase-4-Judgment-Prediction/   submission_01 ... submission_06
├── Phase-5-Statute-Identification/ submission_01 ... submission_06
├── Phase-6-Legal-Summarization/   submission_01 ... submission_06
└── Phase-7-Prior-Case-Retrieval/  submission_01 ... submission_06
```

Each phase folder has its own README with a table that maps every `submission_XX.ipynb` to its original notebook name, backbone, method, parameter count and official score. Each notebook is one experiment.

## Backbones used in Phases 2 to 7

| Name | Architecture |
|---|---|
| M1 | GPT-style: learned positions, LayerNorm, GELU |
| M2 | LLaMA-style: rotary positions, RMSNorm, SwiGLU |
| M3 | M2 with grouped-query attention (8 query heads, 2 key/value heads) and a 256-token sliding window |
| M4 | Hybrid: alternating causal attention and gated causal depthwise convolution |

A "distilled student" keeps every second block of a backbone and is trained from the best earlier fine-tuned model of that phase.

## Notebook and submission counts

- The repository has **41 notebooks**: 5 for Phase 1 and 6 for each of Phases 2 to 7.
- The report counts **40 valid submissions** (5, 6, 5, 6, 6, 6, 6 for Phases 1 to 7). The one notebook not counted is the Phase-3 QLoRA notebook (`Phase-3-Rhetorical-Role/submission_01.ipynb`), which is kept for completeness and is not part of the report's tables.

## Reproducibility and limitations

- The notebooks refer to Kaggle paths (`/kaggle/input/...`), the Hugging Face dataset, the tokenizer file and Phase-1 checkpoints. The checkpoints and the tokenizer file are **not included** in this repository because of their size; see [`tokenizer/README.md`](tokenizer/README.md). Update the paths in each notebook's setup cell if you run them elsewhere.
- No credentials are stored in the notebooks; an optional `HF_TOKEN` is read from Kaggle secrets.
- Local validation sets are small in several phases (for example 10 documents in Phases 2 and 3), so local scores are only a rough guide, and official differences of about 0.005 or less should be treated as ties.
- The scores are benchmark results, not a measure of legal correctness. A real deployment would need expert review, source traceability and evaluation on a firm's own matters.

## Acknowledgement

Course dataset: `Exploration-Lab/CS-781-Capstone`. Please follow the official course instructions and the dataset's terms. An LLM assistant (Claude) was used for understanding concepts, basic code and writing support, as declared in the report.
