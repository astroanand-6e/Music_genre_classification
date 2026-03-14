# Music Genre Classification with Pre-trained Audio Models

**CS5100 Foundations of AI — Northeastern University, Spring 2026**

Comparing discriminative (MERT, AST), multimodal (CLAP), and generative (MusicLDM-VAE) pre-trained audio representations for music genre classification on the FMA-Small dataset.

## Project Structure

```
CS5100_FAI/
├── data/
│   └── data_utils.py           # Shared: metadata, audio I/O, dataset, splits, result logging
│
├── models/
│   ├── mert/finetune.py        # MERT-v1 (95M / 330M) — zero-shot & fine-tune
│   ├── clap/finetune.py        # CLAP (LAION / Microsoft) — zero-shot & fine-tune
│   ├── ast/finetune.py         # Audio Spectrogram Transformer — zero-shot & fine-tune
│   └── musicldm/finetune.py    # MusicLDM VAE encoder — zero-shot & fine-tune
│
├── evaluate.py                 # Unified multi-model comparison
│
├── notebooks/
│   ├── progress_report.ipynb   # Living progress report (auto-loads results)
│   └── baseline.ipynb          # Original MERT baseline notebook
│
└── results/
    ├── runs/                   # JSON summaries of every experiment
    ├── figures/                # Training curves, confusion matrices, F1 charts
    └── checkpoints/            # (gitignored) model checkpoint .pt files
```

## Dataset

**FMA-Small** — 8,000 tracks (30s each), 8 balanced genres:
Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock.

Download from [FMA](https://github.com/mdeff/fma) and place as:
- `fma_small/` — audio files
- `fma_metadata/` — `tracks.csv`, `genres.csv`

## Models

| Model | Type | HuggingFace ID | Sample Rate |
|-------|------|---------------|-------------|
| MERT-v1-95M | Music-specific discriminative | `m-a-p/MERT-v1-95M` | 24 kHz |
| MERT-v1-330M | Music-specific discriminative | `m-a-p/MERT-v1-330M` | 24 kHz |
| CLAP (LAION) | Audio-text contrastive | `laion/clap-htsat-fused` | 48 kHz |
| CLAP (Microsoft) | Audio-text contrastive | `microsoft/msclap` | 48 kHz |
| AST | General audio discriminative | `MIT/ast-finetuned-audioset-10-10-0.4593` | 16 kHz |
| MusicLDM-VAE | Generative (music) | `ucsd-reach/musicldm` | 16 kHz |

## Quick Start

### Zero-shot linear probe

```bash
# Single model
python models/mert/finetune.py --mode zero_shot --model_size 95m
python models/ast/finetune.py  --mode zero_shot
python models/clap/finetune.py --mode zero_shot --variant laion
python models/musicldm/finetune.py --mode zero_shot

# Multi-model comparison
python evaluate.py --model mert-95m --model mert-330m --model ast --model musicldm
```

### Fine-tuning

```bash
python models/mert/finetune.py     --mode finetune --model_size 330m --epochs 15
python models/ast/finetune.py      --mode finetune --epochs 20
python models/clap/finetune.py     --mode finetune --variant laion --epochs 20
python models/musicldm/finetune.py --mode finetune --epochs 20
```

### Results

Every run saves a structured JSON to `results/runs/`. Open `notebooks/progress_report.ipynb` and re-run the aggregation cell to see all results in a single comparison table and chart.

Checkpoints save to `results/checkpoints/{model}/{model}_{variant}_{YYYYMMDD_HHMMSS}.pt`.

## Requirements

```
torch
torchaudio
transformers
diffusers          # for MusicLDM
accelerate
librosa
scikit-learn
pandas
numpy
matplotlib
seaborn
tqdm
```

## Results

### Overall Accuracy

| Model | Zero-shot | Fine-tuned (head-only) | Fine-tuned (end-to-end) |
|-------|:---------:|:----------------------:|:-----------------------:|
| MERT-v1-95M | 58.41% | 57.54% | 63.12% |
| MERT-v1-330M | 62.41% | — | **66.06%** ← best |
| CLAP (LAION) | 12.50% | — | 63.75% |
| AST | 55.25% | — | 64.94% |

> MERT head-only fine-tune trained the classification head with the encoder frozen (100 epochs). End-to-end fine-tune updated all parameters jointly.

### Per-class F1 (Fine-tuned, end-to-end)

| Genre | MERT-330M | AST | CLAP LAION | MERT-95M (head-only) |
|-------|:---------:|:---:|:----------:|:--------------------:|
| Electronic | **0.722** | 0.677 | 0.641 | 0.590 |
| Experimental | 0.563 | 0.546 | **0.571** | 0.507 |
| Folk | 0.672 | **0.711** | 0.657 | 0.644 |
| Hip-Hop | **0.802** | 0.765 | 0.750 | 0.690 |
| Instrumental | 0.611 | **0.637** | 0.623 | 0.595 |
| International | **0.801** | 0.774 | 0.752 | 0.587 |
| Pop | **0.451** | 0.430 | 0.437 | 0.302 |
| Rock | **0.667** | 0.633 | 0.659 | 0.638 |

### Key Observations

- **MERT-330M fine-tuned (66.06%)** is the best result overall — music-domain pre-training at scale wins both zero-shot and fine-tuned settings.
- **AST** (64.94%) and **CLAP LAION** (63.75%) are competitive after fine-tuning despite very different pre-training objectives.
- **CLAP LAION** shows the largest gain (+51.3 pp) from its near-random zero-shot — its audio encoder requires supervised fine-tuning to be useful for closed-set classification.
- **MERT-95M head-only** (57.54%) actually dropped below zero-shot (58.41%), confirming frozen MERT representations are best used with a simple logistic regression probe, not a trained MLP head.
- **Pop** is the hardest genre for all models (F1 ≤ 0.45), due to high acoustic overlap with other genres.
- **Experimental** is consistently the second-hardest (F1 ≈ 0.51–0.57), consistent with its loose genre definition.

Full run JSONs are in `results/runs/` and figures in `results/figures/` and `results/evaluations/`.
