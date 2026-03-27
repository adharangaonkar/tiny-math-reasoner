# tiny-math-reasoner

CoT distillation of DeepSeek-R1-14B into Qwen2.5-0.5B for mathematical reasoning.

## Overview

This project distills chain-of-thought reasoning from a large teacher model (DeepSeek-R1-14B) into a small student model (Qwen2.5-0.5B) using supervised fine-tuning on verified math traces.

The goal is to improve mathematical reasoning in a 500M parameter model using a dataset of ~577 high-quality CoT traces generated from the MATH dataset (Levels 1–3).

## Pipeline

```
MATH Dataset (Levels 1–3)
        ↓
Trace Generation        — DeepSeek-R1-14B via Ollama
        ↓
Answer Filtering        — Keep only traces with correct final answers
        ↓
Dataset Formatting      — Chat-formatted HF Dataset
        ↓
Fine-tuning             — Qwen2.5-0.5B via TRL SFTTrainer
        ↓
Evaluation              — MATH test split + GSM8K (base vs fine-tuned)
```

## Models

| Role | Model |
|---|---|
| Teacher | DeepSeek-R1-14B (Q4_K_M via Ollama) |
| Student (base) | Qwen2.5-0.5B |
| Student (fine-tuned) | Qwen2.5-0.5B + SFT |

## Dataset

- Source: `lighteval/MATH` — Levels 1, 2, and 3
- Problems: 910 total processed
- Clean traces kept: 577 (63.4% keep rate)
- Eval: `lighteval/MATH` test split + GSM8K

## Notebooks

| Notebook | Description |
|---|---|
| `01_trace_generation.ipynb` | Generate CoT traces from teacher model |
| `02_dataset_formatting.ipynb` | Format and split dataset for training |
| `03_training.ipynb` | Fine-tune student model with SFTTrainer |
| `04_evaluation.ipynb` | Evaluate fine-tuned model |
| `05_comparison.ipynb` | Side-by-side base vs fine-tuned comparison |

## Hardware

- GPU: NVIDIA RTX 5070 Ti (16GB VRAM)
- RAM: 64GB
- OS: Ubuntu 24.04

## Setup

```bash
git clone https://github.com/yourusername/tiny-math-reasoner
cd tiny-math-reasoner
python -m venv ml-env
source ml-env/bin/activate
pip install -r requirements.txt
```

## Requirements

See `requirements.txt`. Key dependencies:
- `torch` + CUDA 12.8
- `transformers`
- `trl`
- `datasets`
- `ollama`
- `wandb`

## Results

*To be updated after training and evaluation.*

| Model | MATH (test) | GSM8K |
|---|---|---|
| Qwen2.5-0.5B (base) | TBD | TBD |
| Qwen2.5-0.5B (fine-tuned) | TBD | TBD |

## License

MIT