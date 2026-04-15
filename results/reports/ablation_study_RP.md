# CALM Ablation Study — Research Plan

**Model:** CALM (Conformer Audio-Language Model)
**Dataset:** FMA-Medium (16 genres, 25k tracks)
**Stage 1 Baseline:** Variant B (no_tags) — 83.25% val / 83.60% test
**Date:** 2026-04-14

---

## 0. Completed Results (before ablation)

### Stage 1 — FMA-Medium (15 epochs, cosine LR 5e-4 → 1e-6)

| Epoch | Train Loss | Val Acc | Notes |
|-------|-----------|---------|-------|
| 1 | 1.376 | 67.65% | |
| 5 | 0.640 | 77.48% | |
| 7 | 0.506 | 80.97% | |
| 11 | 0.286 | 83.18% | |
| 15 | 0.154 | **83.25%** | Test: 83.60% |

### Stage 1 — FMA-Small (15 epochs)

| Epoch | Train Loss | Val Acc | Notes |
|-------|-----------|---------|-------|
| 1 | 1.598 | 58.13% | |
| 5 | 0.613 | 76.56% | |
| 10 | 0.253 | 80.00% | |
| 13 | 0.080 | **81.72%** | Test: 81.44% |

### Cosine-Restart Continuation (Epochs 16+, LR 1e-4 → 1e-6, label_smoothing=0.1)

Training in progress. Resumed from `calm_ET_83.25_medium.pt` and `calm_ET_81.72_small.pt`.

| Run | Best Val (so far) | Starting Point | Status |
|-----|:-----------------:|:--------------:|--------|
| Medium restart | 84.13% (Epoch 17) | 83.25% | Running |
| Small restart | — | 81.72% | Running |

---

## 1. Study Design

Five variants, each changing only the text input to CALM. Same frozen Gemma 4 E2B encoders, same cross-attention architecture, same hyperparameters.

| Variant | Text Input | Cache File | What It Tests |
|---------|-----------|-----------|--------------|
| **A. audio-only** | Empty string (zeros) | `text_cache_medium_audio_only.json` | Conformer baseline without any text signal |
| **B. no_tags** (current) | Artist name + bio + title | `text_cache_medium_no_tags.json` | Core metadata value |
| **C. with_tags** | B + artist tags | `text_cache_medium_with_tags.json` | Genre leakage from tags |
| **D. lyrics-only** | Song lyrics (any script) | `text_cache_medium_lyrics.json` | Pure content, no artist identity |
| **E. metadata+lyrics** | Concatenate B + D | `text_cache_medium_combined.json` | Full information fusion |

### Exact text prompts per variant

```
A: ""  (zero-vector embedding)

B: "Artist: Arijit Singh. Bio: Indian playback singer known for
    romantic melodies... Track: Gerua."

C: "Artist: Arijit Singh. Bio: Indian playback singer known for
    romantic melodies... Track: Gerua. Tags: bollywood, romantic, hindi."

D: "Lyrics: dhoop se nikhal ke chaaon se phisal ke hum mile
    jahan par lamha tham gaya..."

E: "Artist: Arijit Singh. Track: Gerua. Lyrics: dhoop se nikhal ke
    chaaon se phisal ke hum mile jahan par lamha tham gaya..."
```

### Controls held constant

- Gemma 4 E2B Conformer (305M, frozen, bfloat16)
- Gemma 4 E2B text backbone for embedding generation (4.6B, frozen)
- FUSION_D = 512, N_HEADS = 8, n_cross_layers = 2
- Trainable params: ~4M (projections + cross-attention + head)
- AdamW, lr_fusion = 5e-4, cosine schedule, weight_decay = 1e-2
- Batch size = 8, grad_accum = 8 (effective batch = 64)
- 15 epochs, early stop patience = 5
- clip_secs = 5.0, 16 kHz mono

---

## 2. Implementation

### No architecture changes needed

CALM's text stream handles any text content through the same embedding → projection → cross-attention pipeline. Swapping `--text_cache` is the only change between variants.

```python
# Forward pass (identical for all variants):
def forward(self, input_features, text_embs, attention_mask=None):
    # Audio stream (same for all variants)
    audio_hidden = self.encode_audio(input_features, attention_mask)
    audio_tokens = self.audio_proj(audio_hidden.float())     # [B, 125, 512]

    # Text stream (content varies by ablation variant)
    # A: text_embs ~ zeros → cross-attn learns nothing from text
    # B: encodes "Artist: X. Bio: Y. Track: Z."
    # D: encodes "Lyrics: dhoop se nikhal ke..."
    # E: encodes "Artist: X. Track: Z. Lyrics: dhoop se..."
    text_token = self.text_proj(text_embs).unsqueeze(1)      # [B, 1, 512]

    # Cross-modal attention
    for ca_audio, ca_text in zip(self.cross_audio, self.cross_text):
        audio_tokens = ca_audio(audio_tokens, text_token)
        text_token   = ca_text(text_token, audio_tokens)

    fused = torch.cat([audio_tokens.mean(1), text_token.squeeze(1)], dim=1)
    return self.head(fused)
```

### Optional future extension: three-stream fusion

If Variant E shows lyrics add value beyond metadata, a dedicated third stream could be explored:

```python
# Two text tokens instead of one:
# text_proj_meta  → [B, 1, 512]  (artist/bio/title)
# text_proj_lyric → [B, 1, 512]  (lyrics)
# Concat → [B, 2, 512] as key_value for cross-attention
# Audio attends to 2 text tokens instead of 1
# Requires separate projection layer + separate embedding caches.
```

---

## 3. Predicted Results

| Variant | Predicted Val Acc | Reasoning |
|---------|:-----------------:|-----------|
| A. audio-only | 60-65% | Similar to Wav2Vec2-Conformer (65.63%); Gemma Conformer may be slightly better due to YouTube pretraining |
| B. no_tags | **82%** (measured) | Metadata provides strong artist/genre correlation |
| C. with_tags | **85-88%** | Tags like "hip-hop" directly encode genre — near-ceiling, but this is "cheating" |
| D. lyrics-only | **72-76%** | Lyrics have genre signal (party words → hip-hop, devotional → folk) but weaker than artist identity |
| E. metadata+lyrics | **83-85%** | Small lift over B — lyrics help where metadata is sparse (tracks without bio) |

### Genres expected to benefit most from lyrics (D vs B)

- **Hip-Hop** — distinctive vocabulary (slang, flow patterns, explicit content)
- **Folk/International** — language itself is a signal (Hindi, French, etc.)
- **Spoken** — lyrics ARE the content
- **Electronic** — might get HURT (lyrics are sparse/repetitive, metadata is more informative)

### Genres expected to benefit most from tags (C vs B)

- All genres uniformly — tags directly name the genre. This variant quantifies leakage, not real performance.

---

## 4. Training Tricks

Recommended additions for runs at 82%+ accuracy:

| Trick | Where | Expected Gain | Notes |
|-------|-------|:------------:|-------|
| **Label smoothing** | `CrossEntropyLoss(label_smoothing=0.1)` | +0.5-1% | Prevents overconfidence on ambiguous genres (Pop/Rock, Electronic/Experimental) |
| **Test-time augmentation** | Average predictions over 3-5 random 5s crops | +1-2% | Free accuracy boost at eval time, no training change |
| **Mixup on embeddings** | `emb = λ*emb_a + (1-λ)*emb_b` in training loop | +0.5-1% | Cheap regularization; λ ~ Beta(0.2, 0.2) |
| **Cosine warmup** | Linear warmup for first 2 epochs, then cosine decay | +0.3-0.5% | Stabilizes early training |

**Do NOT change** (until all ablations complete):
- Do not unfreeze the Conformer (Stage 2) — invalidates A-E comparisons
- Do not increase model capacity — 4M params is sufficient, more will overfit
- Do not change audio clip length or sampling rate

---

## 5. Run Commands

```bash
# Prerequisites
export HF_HUB_OFFLINE=1
conda activate torch

# ── Build caches ──

# Audio-only + combined caches:
python data/build_combined_cache.py --dataset medium

# FMA-Medium lyrics cache (~4 hours for 25k tracks):
python data/build_lyrics_cache.py --dataset medium --resume

# ── Run ablations ──

# A. Audio-only baseline:
python models/calm/finetune.py --dataset medium --stage 1 \
    --text_cache data/text_cache_medium_audio_only.json \
    --run_tag audio_only --epochs 15 --batch_size 8 --grad_accum 8

# B. No tags (DONE — baseline):
# python models/calm/finetune.py --dataset medium --stage 1 \
#     --text_cache data/text_cache_medium_no_tags.json \
#     --run_tag no_tags --epochs 15 --batch_size 8 --grad_accum 8

# C. With tags (leakage ablation):
python models/calm/finetune.py --dataset medium --stage 1 \
    --text_cache data/text_cache_medium_with_tags.json \
    --run_tag with_tags --epochs 15 --batch_size 8 --grad_accum 8

# D. Lyrics only:
python models/calm/finetune.py --dataset medium --stage 1 \
    --text_cache data/text_cache_medium_lyrics.json \
    --run_tag lyrics_only --epochs 15 --batch_size 8 --grad_accum 8

# E. Combined (metadata + lyrics):
python models/calm/finetune.py --dataset medium --stage 1 \
    --text_cache data/text_cache_medium_combined.json \
    --run_tag combined --epochs 15 --batch_size 8 --grad_accum 8

# ── Evaluate all ──

# Use best checkpoint from each run:
for tag in audio_only no_tags with_tags lyrics_only combined; do
    python models/calm/zero_shot.py --dataset medium \
        --calm_ckpt results/checkpoints/calm/calm_${tag}_*_medium.pt \
        --text_cache data/text_cache_medium_${tag}.json
done
```

---

## 6. Results Table (to be filled after runs)

### Overall accuracy

| Variant | Text Input | Best Val | Test Acc | Delta vs B | Epochs |
|---------|-----------|:--------:|:--------:|:----------:|:------:|
| A. audio-only | (none) | % | % | | |
| **B. no_tags** | **artist+bio+title** | **83.25%** | **83.60%** | **(ref)** | **15** |
| C. with_tags | B + artist tags | % | % | | |
| D. lyrics-only | song lyrics | % | % | | |
| E. metadata+lyrics | artist+title+lyrics | % | % | | |

### Per-class F1 (selected genres)

| Genre | A | B | C | D | E |
|-------|:---:|:---:|:---:|:---:|:---:|
| Electronic | | 0.83 | | | |
| Hip-Hop | | 0.88 | | | |
| Rock | | 0.87 | | | |
| International | | 0.78 | | | |
| Folk | | 0.79 | | | |
| Experimental | | 0.65 | | | |
| Pop | | 0.61 | | | |
| Classical | | 0.89 | | | |
| Instrumental | | 0.78 | | | |
| Old-Time/Historic | | 0.92 | | | |
| Spoken | | 0.57 | | | |
| Jazz | | 0.69 | | | |
| Soul-RnB | | 0.74 | | | |

### Key findings (to be written)

- Audio-only (A) vs no_tags (B): Delta = ___ pp → quantifies metadata value
- with_tags (C) vs no_tags (B): Delta = ___ pp → quantifies genre leakage from artist tags
- lyrics (D) vs no_tags (B): Delta = ___ pp → lyrics vs metadata as text source
- combined (E) vs no_tags (B): Delta = ___ pp → whether lyrics complement metadata
- Which genres benefit most from lyrics: ___
- Which genres benefit most from tags: ___

---

## 7. Timeline

| Task | Est. Time | Dependency |
|------|----------|------------|
| Build audio-only + combined caches | 1 min | None |
| Build FMA-Medium lyrics cache | ~4 hrs | lyrics.ovh + LRCLIB APIs |
| Run Variant A (audio-only) | ~1.5 hrs | Cache ready |
| Run Variant C (with_tags) | ~1.5 hrs | Cache exists |
| Run Variant D (lyrics-only) | ~1.5 hrs | Lyrics cache |
| Run Variant E (combined) | ~1.5 hrs | Combined cache |
| Evaluate all checkpoints | ~20 min | All runs done |
| Write up findings | 30 min | All results in |

**Critical path:** lyrics cache (4 hrs) → Variant D + E training (~3 hrs) → eval (20 min).
Variants A and C can run immediately in parallel with lyrics cache building.
