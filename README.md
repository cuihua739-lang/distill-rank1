# Distill-Rank1: Knowledge Distillation for Rank1 Reranking

> MSc Data Science Project, University of Glasgow

## Overview

This project explores **knowledge distillation** for improving the efficiency of **Rank1**, a test-time compute reranking model for information retrieval. Rank1 leverages reasoning chains (thinking at inference time) to make relevance judgments, but this is computationally expensive. 

The goal of this project is to distill the knowledge from the reasoning-capable teacher model into a smaller, more efficient student model that makes **direct relevance predictions without generating reasoning chains** at inference time, significantly reducing latency and compute costs while preserving performance.

## Background

**Rank1** ([Weller et al., 2025](https://arxiv.org/abs/2502.18418)) is a reasoning reranker model that generates reasoning chains before making relevance judgments in information retrieval tasks. While effective, the test-time compute overhead makes it impractical for large-scale deployment.

**Knowledge Distillation** (Hinton et al., 2015) transfers knowledge from a large teacher model to a smaller student model. In this project, the teacher is the Rank1 model with full reasoning capabilities, and the student is a smaller model trained to directly output relevance labels.

## Methodology

### 1. Data Preparation
- Source dataset: [rank1-training-data](https://huggingface.co/datasets/jhu-clsp/rank1-training-data) (385,791 samples)
- Each training sample originally contains: query, passage, reasoning chain (``), and label (true/false)
- **Distillation step**: Extract query and passage as input, and the final label as target, removing the intermediate reasoning chain
- The student model learns to predict the same relevance labels that the teacher model produced through reasoning, effectively distilling the teacher's knowledge

### 2. Model Architecture
- **Base model**: Qwen2.5-1.5B (configurations also available for 3B, 7B variants)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- **Rank1 integration**: Uses Rank1's evaluation framework and prompts

### 3. Training Configuration
| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2.5-1.5B |
| Fine-tuning Method | LoRA (all-linear layers) |
| Batch Size | 16 (8 per device, 2 gradient accumulation) |
| Learning Rate | 1e-4 (cosine scheduler) |
| Epochs | 1 |
| Precision | FP16 |
| Trainable Params | 9,232,384 (0.59% of total) |
| Hardware | 1x NVIDIA RTX 3090 (24GB) |
| Training Time | ~5.7 hours (24,112 steps) |

### 4. Training Loss Curve

The training loss steadily decreased from 0.2521 (step 500) to 0.0335 (step 18,500), demonstrating effective knowledge transfer:

```
Step   | Loss
-------|-------
500    | 0.2521
1000   | 0.0765
5000   | 0.0498
10000  | 0.0420
15000  | 0.0363
18500  | 0.0335
```

### 5. Variants
- **Chainless (Distilled)**: Direct relevance prediction without reasoning - `main.ipynb`
- **AWQ Quantized**: Quantized version for deployment on smaller GPUs - `main-awq.ipynb`
- **Chain-of-Thought Training**: Alternative approach with reasoning supervision - `cot_train.ipynb`

## Repository Structure

```
distill-rank1/
|-- main.ipynb           # End-to-end pipeline: data prep, LoRA training, evaluation
|-- main-awq.ipynb       # AWQ-quantized model variant for smaller footprint
|-- cot_train.ipynb      # Chain-of-thought training pipeline
|-- Evaluate.ipynb       # Evaluation and benchmarking
|-- configs/             
|   |-- train_config.yaml   # Training hyperparameters
|   |-- dataset_info.json   # Dataset configuration
|-- requirements.txt     # Python dependencies
|-- README.md            # This file
```

## Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 24GB+ VRAM (for full model) or 8GB+ (for AWQ quantized)
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_HF_USERNAME-lang/distill-rank1
cd distill-rank1

# Install dependencies
uv venv env --python=3.10
source env/bin/activate
uv pip install -r requirements.txt

# Install LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && uv pip install . && cd ..

# Set up Rank1
git clone https://github.com/orionw/rank1.git && cd rank1
git submodule update --init --recursive
cd ..
```

### Training (Chainless/Distilled Model)

Open `main.ipynb` in Jupyter and run cells sequentially. The notebook handles:
1. Environment setup and dependency installation
2. Data preparation from rank1-training-data (385,791 samples)
3. LoRA fine-tuning via LLaMA-Factory with Qwen2.5-1.5B
4. MTEB evaluation of the distilled model

### Training (AWQ Quantized)

Use `main-awq.ipynb` for a quantized variant that fits on smaller GPUs.

### Evaluation

Use `Evaluate.ipynb` for benchmarking on MTEB tasks and comparison with the original Rank1 model.

## Key Files

- **`main.ipynb`**: The complete end-to-end pipeline. Includes data extraction (removing reasoning chains), LoRA training with LLaMA-Factory, and MTEB integration.
- **`main-awq.ipynb`**: AWQ quantization of the trained model for efficient deployment on resource-constrained GPUs.
- **`cot_train.ipynb`**: Chain-of-thought training variant that preserves reasoning supervision.
- **`Evaluate.ipynb`**: Evaluation pipeline for measuring NDCG@10, MRR, and other IR metrics.

## Dependencies

Key packages used in this project:
- `transformers` (>=4.49.0)
- `llamafactory` (>=0.9.4)
- `peft` (>=0.14.0)
- `trl` (>=0.8.6)
- `vllm` (>=0.7.2)
- `accelerate` (>=1.3.0)
- `datasets` (>=2.16.0)
- `mteb` (MTEB evaluation framework)
- `wandb` (experiment tracking)
- `codecarbon` (energy monitoring)

## Results

The distilled chainless model successfully learns to predict relevance labels from the teacher's outputs:
- Training converges to low loss (~0.0335) within ~18k steps
- The model retains the distillation knowledge for binary relevance classification
- Evaluation on MTEB benchmark suite validates reranking effectiveness

Detailed evaluation metrics and comparison with baseline Rank1 are available in `Evaluate.ipynb`.

## Acknowledgments

- **Rank1**: [Orion Weller et al.](https://arxiv.org/abs/2502.18418) — Original test-time compute reranking model
- **LLaMA-Factory**: [hiyouga](https://github.com/hiyouga/LLaMA-Factory) — Unified LLM fine-tuning framework
- **MTEB**: Embedding benchmarking framework

This work was conducted as part of the MSc Data Science program at the University of Glasgow.

## License

MIT
