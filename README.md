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
└── results/                    # (gitignored) checkpoints, runs, figures, reports
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

## Results (Initial)

| Model | Mode | Accuracy |
|-------|------|----------|
| MERT-v1-95M | Zero-shot linear probe | 58.41% |
| MERT-v1-330M | Zero-shot linear probe | 62.41% |
