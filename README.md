# Music Genre Classification with Pre-trained Audio Models and Multimodal Fusion

**CS 5100: Foundations of Artificial Intelligence — Northeastern University, Spring 2026**

**Author:** Anand Thakkar

Systematic comparison of six pre-trained audio encoders for music genre classification, plus **CALM** (Conformer Audio-Language Model) — a multimodal architecture that fuses frozen audio and text representations via cross-modal attention, achieving **83.9% accuracy on FMA-Medium (16 genres)** with only 4M trainable parameters.

## Results

### FMA-Medium (16 genres)

| Model | Type | Zero-shot | Fine-tuned |
|-------|------|:---------:|:----------:|
| **CALM** | Audio + Text | 0.78% | **83.90%** |
| CLAP (Microsoft) | Audio-text contrastive | — | 74.78% |
| CLAP (LAION) | Audio-text contrastive | 32.02% | 73.32% |
| MERT-330M | Music SSL | 70.26% | 73.26% |
| AST | Audio spectrogram | 67.08% | 71.72% |
| MERT-95M | Music SSL | 67.74% | 70.92% |
| MusicLDM-VAE | Generative encoder | — | 70.30% |
| Conformer | Speech SSL | 60.84% | 65.63% |

### FMA-Small (8 genres)

| Model | Zero-shot | Fine-tuned |
|-------|:---------:|:----------:|
| **CALM** | — | **81.44%** |
| MERT-330M | 62.41% | 66.06% |
| AST | 55.25% | 64.94% |
| CLAP (LAION) | 12.50% | 63.75% |
| MERT-95M | 58.41% | 63.12% |
| MusicLDM-VAE | — | 56.87% |

## Project Structure

```
CS5100_FAI/
├── data/
│   ├── data_utils.py              # Dataset, audio I/O, metadata, splits, logging
│   ├── preprocess_audio.py        # MP3 → 16kHz mono WAV conversion
│   ├── build_text_cache.py        # Text metadata caching for CALM
│   ├── build_lyrics_cache.py      # Lyrics extraction via APIs
│   └── build_bollywood_metadata.py
│
├── models/
│   ├── calm/finetune.py           # CALM — multimodal audio-language fusion
│   ├── mert/finetune.py           # MERT-v1 (95M / 330M)
│   ├── clap/finetune.py           # CLAP (LAION / Microsoft)
│   ├── ast/finetune.py            # Audio Spectrogram Transformer
│   ├── musicldm/finetune.py       # MusicLDM VAE encoder
│   ├── conformer/finetune.py      # Wav2Vec2-Conformer
│   └── lyrics/multimodal_fusion.py
│
├── evaluate.py                    # Unified multi-model comparison
│
└── results/
    ├── runs/                      # JSON summaries per experiment
    ├── logs/finetune/             # Epoch-level training CSVs
    ├── figures/                   # Training curves, confusion matrices
    ├── reports/                   # Paper drafts and figures
    └── checkpoints/               # Model .pt files (gitignored)
```

## Datasets

- **FMA-Small** — 8,000 tracks, 8 balanced genres
- **FMA-Medium** — 25,000 tracks, 16 genres (imbalanced)
- **Bollywood** — 153 tracks, 6 sub-genres (out-of-distribution eval)

Download FMA from [github.com/mdeff/fma](https://github.com/mdeff/fma) and place as `fma_small/`, `fma_medium/`, `fma_metadata/`.

## Models

| Model | HuggingFace ID | Sample Rate |
|-------|---------------|:-----------:|
| MERT-v1-95M | `m-a-p/MERT-v1-95M` | 24 kHz |
| MERT-v1-330M | `m-a-p/MERT-v1-330M` | 24 kHz |
| CLAP (LAION) | `laion/clap-htsat-fused` | 48 kHz |
| CLAP (Microsoft) | `microsoft/msclap` | 44.1 kHz |
| AST | `MIT/ast-finetuned-audioset` | 16 kHz |
| MusicLDM-VAE | `ucsd-reach/musicldm` | 16 kHz |
| Conformer | `facebook/wav2vec2-conformer-rel-pos-large` | 16 kHz |
| CALM | Gemma 4 Conformer + Gemma 4 Text Encoder | 16 kHz |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Zero-shot
python models/mert/finetune.py --mode zero_shot --model_size 330m

# Fine-tune
python models/mert/finetune.py --mode finetune --model_size 330m --epochs 15
python models/calm/finetune.py --dataset fma_medium --epochs 22

# Multi-model evaluation
python evaluate.py --model mert-330m --model ast --model clap-laion
```

## Requirements

See `requirements.txt`. Core dependencies: `torch`, `torchaudio`, `transformers`, `diffusers`, `librosa`, `scikit-learn`, `pandas`, `matplotlib`.
