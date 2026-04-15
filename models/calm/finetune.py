"""
CALM — Conformer Audio-Language Model for music genre classification.

Architecture (inspired by Gemma 4's multimodal design):
  - Audio stream  : frozen/fine-tuned Gemma 4 E2B Conformer → [B, T_a, 1536]
  - Text stream   : frozen Gemma 4 E2B text encoder on lyrics / artist bio + name + title
                    → [B, D_text]  (mean-pooled, 1536-d)
  - Both projected into a shared 512-d space
  - Cross-modal attention: audio tokens attend to text, text attends to audio
  - Mean-pool each enriched stream → concat [1024] → MLP → 16 genres

Ablation:
  --text_cache  data/text_cache_medium_no_tags.json    (default, no leakage)
  --text_cache  data/text_cache_medium_with_tags.json  (ablation: +artist tags)

Usage
-----
# Build text cache first (if not done):
#   python data/build_text_cache.py --dataset medium --both

# Stage 1 — train only projections + cross-attn + head (frozen encoders):
python models/calm/finetune.py --dataset medium --stage 1 --epochs 15

# Stage 2 — unfreeze top Conformer layers:
python models/calm/finetune.py --dataset medium --stage 2 --epochs 10 \
    --conformer_ckpt results/checkpoints/calm/<stage1_ckpt>

# Ablation — same but with tags:
python models/calm/finetune.py --dataset medium --stage 1 \
    --text_cache data/text_cache_medium_with_tags.json --run_tag with_tags
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

_here = Path(__file__).resolve().parent
_project_root = _here.parent.parent
sys.path.insert(0, str(_project_root))

from data.data_utils import (
    load_fma_metadata, get_splits, get_label_maps,
    resolve_audio_path, fma_audio_dir,
    save_run_results, compute_per_class_f1,
    append_finetune_epoch_log, checkpoint_path,
)

TARGET_SR   = 16_000
AUDIO_D     = 1536   # Gemma 4 E2B audio tower output dim
TEXT_D      = 1536   # google/gemma-4-E2B-it text encoder hidden dim
FUSION_D    = 512    # shared projection dim
N_HEADS     = 8      # cross-attention heads


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_audio(path: str, clip_secs: float, sr: int = TARGET_SR) -> np.ndarray:
    import librosa
    wav, _ = librosa.load(path, sr=sr, mono=True, duration=clip_secs)
    n = int(clip_secs * sr)
    if len(wav) < n:
        wav = np.pad(wav, (0, n - len(wav)))
    return wav[:n].astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

_audio_feature_extractor = None

def get_audio_feature_extractor():
    """Lazy-load Gemma 4's audio feature extractor (mel spectrogram)."""
    global _audio_feature_extractor
    if _audio_feature_extractor is None:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(GEMMA_MODEL_NAME)
        _audio_feature_extractor = proc.feature_extractor
    return _audio_feature_extractor


class AudioTextDataset(Dataset):
    """Returns (mel_features, text_embedding, label) per track."""

    def __init__(self, df, text_cache: dict, text_embeddings: dict,
                 audio_dir: str, clip_secs: float = 5.0, split: str = "train"):
        self.df = df
        self.text_cache = text_cache
        self.text_embeddings = text_embeddings  # track_id -> np.ndarray [TEXT_D]
        self.audio_dir = audio_dir
        self.clip_secs = clip_secs
        self.split = split
        self.ids = list(df.index)
        self.fe = get_audio_feature_extractor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        track_id = self.ids[idx]
        row = self.df.loc[track_id]
        audio_path = resolve_audio_path(track_id, row, audio_dir=self.audio_dir)

        try:
            wav = load_audio(audio_path, self.clip_secs)
        except Exception:
            wav = np.zeros(int(self.clip_secs * TARGET_SR), dtype=np.float32)

        # Convert to mel spectrogram via Gemma's feature extractor
        features = self.fe([wav], sampling_rate=TARGET_SR, return_tensors="pt")
        mel = features.input_features.squeeze(0)  # [T_mel, 128]

        text_emb = self.text_embeddings.get(int(track_id))
        if text_emb is None:
            text_emb = np.zeros(TEXT_D, dtype=np.float32)

        return {
            "input_features": mel,
            "text_emb":       torch.from_numpy(text_emb.astype(np.float32)),
            "label":          torch.tensor(row["label"], dtype=torch.long),
            "track_id":       track_id,
        }


# ── CALM Model ────────────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Single cross-attention block: query from one stream, key/value from other.
    Inspired by Gemma 4's shared token space — both modalities project to the
    same FUSION_D dimension before interaction.
    """

    def __init__(self, d_model: int = FUSION_D, n_heads: int = N_HEADS,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        query    : [B, T_q, D]
        key_value: [B, T_kv, D]
        Returns  : [B, T_q, D]  (query enriched with key_value context)
        """
        out, _ = self.attn(query, key_value, key_value)
        return self.norm(query + self.drop(out))


class CALM(nn.Module):
    """
    Conformer Audio-Language Model.

    Audio stream  : Wav2Vec2-Conformer → [B, T_a, AUDIO_D]
                    → project → [B, T_a, FUSION_D]
    Text stream   : pre-computed sentence-transformer embedding [B, TEXT_D]
                    → project → [B, 1, FUSION_D]   (unsqueeze as 1 token)
    Cross-attention (×n_cross_layers):
        audio tokens attend to text token  → enriched_audio [B, T_a, FUSION_D]
        text token  attends to audio tokens → enriched_text  [B, 1,   FUSION_D]
    Pool each → concat [B, 2*FUSION_D] → MLP → num_classes
    """

    def __init__(self, num_classes: int, n_cross_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        # Audio encoder (loaded separately, weights injected via load_conformer)
        self.audio_encoder = None  # set after init

        # Projection layers (always trainable)
        self.audio_proj = nn.Sequential(
            nn.Linear(AUDIO_D, FUSION_D),
            nn.LayerNorm(FUSION_D),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(TEXT_D, FUSION_D),
            nn.LayerNorm(FUSION_D),
        )

        # Cross-modal attention layers (trainable)
        self.cross_audio = nn.ModuleList(
            [CrossModalAttention(FUSION_D, N_HEADS, dropout)
             for _ in range(n_cross_layers)]
        )
        self.cross_text = nn.ModuleList(
            [CrossModalAttention(FUSION_D, N_HEADS, dropout)
             for _ in range(n_cross_layers)]
        )

        # Classifier head (trainable)
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_D * 2),
            nn.Linear(FUSION_D * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def encode_audio(self, input_features, attention_mask=None):
        """Run Gemma 4 audio tower (Conformer), return [B, T, AUDIO_D]."""
        out = self.audio_encoder(
            input_features.bfloat16(),
            attention_mask=attention_mask,
        )
        return out.last_hidden_state  # [B, T_a, 1536] in bfloat16

    def forward(self, input_features, text_embs, attention_mask=None):
        # Audio: [B, T_a, AUDIO_D] → [B, T_a, FUSION_D]
        audio_hidden = self.encode_audio(input_features, attention_mask)
        audio_tokens = self.audio_proj(audio_hidden.float())   # [B, T_a, FUSION_D]

        # Text: [B, TEXT_D] → [B, 1, FUSION_D]
        text_token = self.text_proj(text_embs).unsqueeze(1)  # [B, 1, FUSION_D]

        # Cross-modal attention
        for ca_audio, ca_text in zip(self.cross_audio, self.cross_text):
            audio_tokens = ca_audio(audio_tokens, text_token)
            text_token   = ca_text(text_token, audio_tokens)

        # Pool each stream and concat
        audio_pooled = audio_tokens.mean(dim=1)   # [B, FUSION_D]
        text_pooled  = text_token.squeeze(1)      # [B, FUSION_D]
        fused = torch.cat([audio_pooled, text_pooled], dim=1)  # [B, 2*FUSION_D]

        return self.head(fused)


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_conformer_backbone():
    """Load Gemma 4 E2B's Conformer audio tower (305M params, 1536-d, 12 layers)."""
    from transformers import AutoModelForMultimodalLM
    print(f"Loading {GEMMA_MODEL_NAME} audio tower (Conformer)...")
    full_model = AutoModelForMultimodalLM.from_pretrained(
        GEMMA_MODEL_NAME, dtype=torch.bfloat16,  # bfloat16 — same range as fp32, no overflow
    )
    audio_tower = full_model.model.audio_tower
    n_params = sum(p.numel() for p in audio_tower.parameters()) / 1e6
    print(f"  Audio Conformer extracted ({n_params:.0f}M params, hidden={AUDIO_D})")
    del full_model
    torch.cuda.empty_cache()
    return audio_tower


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_top_layers(backbone, n_layers: int):
    """Unfreeze the top n transformer layers of the Conformer audio tower."""
    total = len(backbone.layers)
    for i, layer in enumerate(backbone.layers):
        if i >= total - n_layers:
            for p in layer.parameters():
                p.requires_grad = True
    print(f"  Unfrozen top {n_layers}/{total} conformer layers")


GEMMA_MODEL_NAME = "google/gemma-4-E2B-it"


def build_text_embeddings(text_cache: dict, batch_size: int = 16,
                          device: str = "cpu") -> dict:
    """
    Encode all text descriptions with the Gemma 4 E2B text encoder (frozen).
    Uses only the text backbone from the multimodal model.
    Returns dict: track_id (int) -> np.ndarray [TEXT_D=1536]
    """
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    print(f"Loading {GEMMA_MODEL_NAME} tokenizer...")
    processor = AutoProcessor.from_pretrained(GEMMA_MODEL_NAME)
    tokenizer = processor.tokenizer
    print(f"  Tokenizer loaded (vocab={tokenizer.vocab_size})")

    print(f"Loading {GEMMA_MODEL_NAME} model weights (this may take 1-2 min)...")
    full_model = AutoModelForMultimodalLM.from_pretrained(
        GEMMA_MODEL_NAME,
        dtype=torch.bfloat16,
    )
    print(f"  Model loaded.")

    # Extract just the text backbone — ignore vision/audio towers
    text_model = full_model.model.language_model
    n_params = sum(p.numel() for p in text_model.parameters()) / 1e6
    print(f"  Text backbone extracted ({n_params:.0f}M params, hidden={TEXT_D})")
    print(f"  Moving text backbone to {device}...")
    text_model = text_model.to(device)
    text_model.eval()

    # Free the rest (vision tower, audio tower, lm_head)
    del full_model
    torch.cuda.empty_cache()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ids   = list(text_cache.keys())
    texts = [text_cache[i] for i in ids]

    print(f"Encoding {len(texts)} texts with Gemma 4 E2B (hidden_dim={TEXT_D})...")
    all_embs = []

    from tqdm import tqdm
    for start in tqdm(range(0, len(texts), batch_size), desc="Gemma4 encode"):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = text_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
            )
            hidden = out.last_hidden_state            # [B, seq_len, 1536]
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)  # mean pool → [B, 1536]
            all_embs.append(pooled.cpu().float().numpy())

    embeddings = np.concatenate(all_embs, axis=0)     # [N, 1536]

    del text_model
    torch.cuda.empty_cache()
    print(f"  Done. Embedding shape: {embeddings.shape}")

    return {int(tid): emb for tid, emb in zip(ids, embeddings)}


def collate_fn(batch):
    # Mel features may vary in length — pad to longest in batch
    mels = [b["input_features"] for b in batch]
    max_len = max(m.shape[0] for m in mels)
    padded = []
    masks  = []
    for m in mels:
        orig_len = m.shape[0]
        pad_len  = max_len - orig_len
        if pad_len > 0:
            m = torch.nn.functional.pad(m, (0, 0, 0, pad_len))
        padded.append(m)
        mask = torch.ones(max_len, dtype=torch.long)
        mask[orig_len:] = 0
        masks.append(mask)
    input_features = torch.stack(padded)        # [B, T_mel, 128]
    attention_mask = torch.stack(masks)          # [B, T_mel]
    text_embs = torch.stack([b["text_emb"] for b in batch])
    labels    = torch.stack([b["label"]    for b in batch])
    return {"input_features": input_features, "attention_mask": attention_mask,
            "text_emb": text_embs, "label": labels}


def max_optimizer_lr(opt):
    return max(pg["lr"] for pg in opt.param_groups)


# ── Training ──────────────────────────────────────────────────────────────────

def run_finetune(args):
    print("=" * 60)
    print("CALM — Conformer Audio-Language Model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  {torch.cuda.get_device_name(0) if device.type=='cuda' else ''}")

    # ── Data ──
    print("\nLoading FMA metadata...")
    df = load_fma_metadata(subset=args.dataset)
    train_df, val_df, test_df = get_splits(df)
    label2id, id2label = get_label_maps(df)
    num_classes = len(label2id)
    audio_dir = fma_audio_dir(args.dataset)

    print(f"Dataset : FMA-{args.dataset.upper()} ({num_classes} genres)")
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    # ── Text cache ──
    text_cache_path = Path(args.text_cache)
    if not text_cache_path.is_absolute():
        text_cache_path = _project_root / text_cache_path

    if not text_cache_path.exists():
        print(f"\nText cache not found: {text_cache_path}")
        print("Run: python data/build_text_cache.py --dataset medium --both")
        sys.exit(1)

    print(f"\nLoading text cache: {text_cache_path.name}")
    with open(text_cache_path) as f:
        text_cache = {int(k): v for k, v in json.load(f).items()}

    # ── Text embeddings ──
    emb_cache_path = text_cache_path.with_suffix(".embeddings.npy")
    emb_ids_path   = text_cache_path.with_suffix(".embeddings_ids.npy")

    if emb_cache_path.exists() and emb_ids_path.exists():
        print("Loading cached text embeddings...")
        raw_embs = np.load(str(emb_cache_path))
        raw_ids  = np.load(str(emb_ids_path))
        text_embeddings = {int(tid): emb for tid, emb in zip(raw_ids, raw_embs)}
    else:
        text_embeddings = build_text_embeddings(
            text_cache,
            device="cuda" if device.type == "cuda" else "cpu",
        )
        # Save for reuse
        ids_arr  = np.array(list(text_embeddings.keys()), dtype=np.int64)
        embs_arr = np.stack(list(text_embeddings.values()))
        np.save(str(emb_cache_path), embs_arr)
        np.save(str(emb_ids_path),   ids_arr)
        print(f"  Saved embeddings cache → {emb_cache_path.name}")

    # ── Datasets ──
    make_ds = lambda split_df, split: AudioTextDataset(
        split_df, text_cache, text_embeddings,
        audio_dir=audio_dir, clip_secs=args.clip_secs, split=split,
    )
    train_ds = make_ds(train_df, "train")
    val_ds   = make_ds(val_df,   "val")
    test_ds  = make_ds(test_df,  "test")

    loader_kw = dict(num_workers=4, pin_memory=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2,
                              shuffle=False, **loader_kw)

    # ── Model ──
    print("\nBuilding CALM model...")
    model = CALM(num_classes=num_classes, n_cross_layers=args.n_cross_layers)

    # Load Conformer backbone
    backbone = load_conformer_backbone()
    freeze_module(backbone)

    if args.stage == 2:
        unfreeze_top_layers(backbone, n_layers=args.unfreeze_layers)

    model.audio_encoder = backbone
    model = model.to(device)

    # Load from checkpoint if resuming
    if args.calm_ckpt:
        print(f"Loading CALM checkpoint: {args.calm_ckpt}")
        state = torch.load(args.calm_ckpt, map_location=device)
        model.load_state_dict(state, strict=False)

    # ── Optimizer ──
    # Separate LRs: lower for backbone (if unfrozen), higher for fusion/head
    fusion_params = (
        list(model.audio_proj.parameters()) +
        list(model.text_proj.parameters()) +
        list(model.cross_audio.parameters()) +
        list(model.cross_text.parameters()) +
        list(model.head.parameters())
    )
    backbone_params = [p for p in model.audio_encoder.parameters()
                       if p.requires_grad]

    param_groups = [{"params": fusion_params, "lr": args.lr_fusion}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr_backbone})

    optimizer  = torch.optim.AdamW(param_groups, weight_decay=1e-2)

    if args.lr_schedule == "cosine_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.epochs, eta_min=1e-6,
        )
        print(f"  LR schedule: CosineAnnealingWarmRestarts (T_0={args.epochs}, peak={args.lr_fusion})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6,
        )
        print(f"  LR schedule: CosineAnnealingLR (T_max={args.epochs}, peak={args.lr_fusion})")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing,
    )
    if args.label_smoothing > 0:
        print(f"  Label smoothing: {args.label_smoothing}")

    # ── Run ID & logging ──
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag    = args.run_tag or ("stage2" if args.stage == 2 else "stage1")
    epoch_log_csv = (
        _project_root / "results" / "logs" / "finetune" /
        f"calm_{run_id}_{tag}_{args.dataset}.csv"
    )
    epoch_log_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nRun ID   : {run_id}")
    print(f"Tag      : {tag}")
    print(f"Epoch log: {epoch_log_csv}")

    # ── Sanity check before training ──
    print("\n[Sanity Check] Testing one batch...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        tb_mel  = test_batch["input_features"].to(device)
        tb_mask = test_batch["attention_mask"].to(device)
        tb_text = test_batch["text_emb"].to(device)

        # Check inputs
        print(f"  mel:  shape={list(tb_mel.shape)}, nan={tb_mel.isnan().any()}, range=[{tb_mel.min():.2f}, {tb_mel.max():.2f}]")
        print(f"  mask: shape={list(tb_mask.shape)}, sum={tb_mask.sum()}")
        print(f"  text: shape={list(tb_text.shape)}, nan={tb_text.isnan().any()}, range=[{tb_text.min():.2f}, {tb_text.max():.2f}]")

        # Check conformer output
        audio_hidden = model.encode_audio(tb_mel, tb_mask)
        print(f"  conformer out: shape={list(audio_hidden.shape)}, nan={audio_hidden.isnan().any()}, dtype={audio_hidden.dtype}, range=[{audio_hidden.min():.2f}, {audio_hidden.max():.2f}]")

        # Check projection
        audio_proj = model.audio_proj(audio_hidden.float())
        print(f"  audio_proj: nan={audio_proj.isnan().any()}, range=[{audio_proj.min():.2f}, {audio_proj.max():.2f}]")

        text_proj = model.text_proj(tb_text)
        print(f"  text_proj:  nan={text_proj.isnan().any()}, range=[{text_proj.min():.2f}, {text_proj.max():.2f}]")

        # Full forward
        logits = model(tb_mel, tb_text, attention_mask=tb_mask)
        print(f"  logits: nan={logits.isnan().any()}, range=[{logits.min():.2f}, {logits.max():.2f}]")
    print("[Sanity Check] Done.\n")

    # ── Training loop ──
    best_val_acc = args.resume_best_val
    epochs_no_improve = 0
    best_ckpt_path = None
    end_epoch = args.start_epoch + args.epochs - 1

    if args.resume_best_val > 0:
        print(f"  Resuming from best val: {args.resume_best_val:.4f}")

    for epoch in range(args.start_epoch, end_epoch + 1):
        # Train
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d} train",
                          leave=False, dynamic_ncols=True)
        for step, batch in enumerate(train_pbar):
            waveforms = batch["input_features"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            text_embs = batch["text_emb"].to(device)
            labels    = batch["label"].to(device)

            logits = model(waveforms, text_embs, attention_mask=attn_mask)
            loss   = criterion(logits, labels) / args.grad_accum

            loss.backward()
            train_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_pbar.set_postfix(loss=f"{loss.item()*args.grad_accum:.3f}")

        # Validate
        model.eval()
        val_loss  = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:2d} val  ",
                              leave=False, dynamic_ncols=True):
                waveforms = batch["input_features"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                text_embs = batch["text_emb"].to(device)
                labels_b  = batch["label"].to(device)

                logits = model(waveforms, text_embs, attention_mask=attn_mask)
                val_loss += criterion(logits, labels_b).item()

                val_preds.extend(logits.argmax(dim=1).cpu().tolist())
                val_labels.extend(labels_b.cpu().tolist())

        val_acc  = accuracy_score(val_labels, val_preds)
        scheduler.step()

        append_finetune_epoch_log(epoch_log_csv, {
            "run_id":        run_id,
            "epoch":         epoch,
            "train_loss":    train_loss / len(train_loader),
            "val_loss":      val_loss   / len(val_loader),
            "val_acc":       val_acc,
            "best_val_acc":  max(best_val_acc, val_acc),
            "lr_max":        max_optimizer_lr(optimizer),
        })

        if val_acc > best_val_acc + args.early_stop_min_delta:
            best_val_acc = val_acc
            epochs_no_improve = 0
            ckpt = checkpoint_path("calm", tag, args.dataset)
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            best_ckpt_path = ckpt
            print(f"Epoch {epoch:2d} | Loss {train_loss/len(train_loader):.3f} "
                  f"| Val {val_acc:.4f}* (saved)")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch:2d} | Loss {train_loss/len(train_loader):.3f} "
                  f"| Val {val_acc:.4f}  (no improve {epochs_no_improve}/{args.early_stop_patience})")
            if (args.early_stop_patience > 0 and
                    epochs_no_improve >= args.early_stop_patience):
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # ── Test ──
    print("\n" + "=" * 60)
    print("Test evaluation")
    print("=" * 60)

    if best_ckpt_path:
        print(f"Loading best checkpoint: {best_ckpt_path}")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    model.eval()
    test_preds, test_labels_list = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", dynamic_ncols=True):
            waveforms = batch["input_features"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            text_embs = batch["text_emb"].to(device)
            labels_b  = batch["label"].to(device)

            logits = model(waveforms, text_embs, attention_mask=attn_mask)

            test_preds.extend(logits.argmax(dim=1).cpu().tolist())
            test_labels_list.extend(labels_b.cpu().tolist())

    test_acc = accuracy_score(test_labels_list, test_preds)
    per_class_f1 = compute_per_class_f1(test_labels_list, test_preds, id2label)

    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Best Val Acc  : {best_val_acc:.4f}")
    print("\nPer-class F1:")
    for genre, f1 in sorted(per_class_f1.items()):
        print(f"  {genre:25s}: {f1:.4f}")

    save_run_results(
        model="calm",
        variant=f"{tag}_stage{args.stage}",
        mode="finetune",
        test_accuracy=test_acc,
        config={
            "stage":          args.stage,
            "run_tag":        tag,
            "text_cache":     str(text_cache_path.name),
            "clip_secs":      args.clip_secs,
            "n_cross_layers": args.n_cross_layers,
            "fusion_d":       FUSION_D,
            "audio_d":        AUDIO_D,
            "text_d":         TEXT_D,
            "batch_size":     args.batch_size,
            "grad_accum":     args.grad_accum,
            "lr_fusion":      args.lr_fusion,
            "lr_backbone":    args.lr_backbone,
            "epochs_run":     epoch,
            "dataset":        args.dataset,
            "best_val_acc":   best_val_acc,
        },
        per_class_f1=per_class_f1,
        dataset=args.dataset,
    )
    print(f"\nResults saved to results/runs/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CALM: Conformer Audio-Language Model for genre classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",    default="medium", choices=["small", "medium"])
    parser.add_argument("--stage",      type=int, default=1, choices=[1, 2],
        help="1=frozen encoders (fast), 2=unfreeze top Conformer layers")
    parser.add_argument("--text_cache", default="data/text_cache_medium_no_tags.json",
        help="Path to text cache JSON (use with_tags variant for ablation)")
    parser.add_argument("--run_tag",    default=None,
        help="Tag appended to checkpoint/log filenames (e.g. 'no_tags', 'with_tags')")
    parser.add_argument("--calm_ckpt",  default=None,
        help="Path to existing CALM checkpoint to resume from")

    # Architecture
    parser.add_argument("--n_cross_layers", type=int, default=2,
        help="Number of cross-modal attention layers")
    parser.add_argument("--clip_secs",      type=float, default=5.0)

    # Training
    parser.add_argument("--epochs",          type=int,   default=15)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--grad_accum",      type=int,   default=8)
    parser.add_argument("--lr_fusion",       type=float, default=5e-4,
        help="LR for projection + cross-attn + head")
    parser.add_argument("--lr_backbone",     type=float, default=1e-5,
        help="LR for unfrozen Conformer layers (Stage 2 only)")
    parser.add_argument("--unfreeze_layers", type=int,   default=4,
        help="Number of top Conformer layers to unfreeze in Stage 2")
    parser.add_argument("--early_stop_patience",  type=int,   default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    # Resume / cosine restart
    parser.add_argument("--resume_best_val", type=float, default=0.0,
        help="Best val acc from previous run (sets early stop baseline)")
    parser.add_argument("--start_epoch", type=int, default=1,
        help="Starting epoch number (for logging continuity)")
    parser.add_argument("--lr_schedule", default="cosine",
        choices=["cosine", "cosine_restart"],
        help="LR schedule: 'cosine' (default) or 'cosine_restart' (warm restarts every T_max epochs)")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
        help="Label smoothing for CrossEntropyLoss (e.g. 0.1)")

    args = parser.parse_args()
    run_finetune(args)


if __name__ == "__main__":
    main()
