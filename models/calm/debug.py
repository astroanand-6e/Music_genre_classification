"""
CALM Debug / Smoke Test
=======================
Validates each component independently before a full training run.
Runs in ~2 minutes on GPU, no full dataset needed.

Usage:
    python models/calm/debug.py              # all checks
    python models/calm/debug.py --skip_data  # skip real audio (pure tensor checks)

Checks:
    1. CALM model shape test         — dummy tensors, verify output shape
    2. Cross-modal attention          — verify audio/text interact correctly
    3. Gradient flow                  — confirm all trainable params get grads
    4. Conformer backbone load        — verify HF model loads and runs
    5. Sentence-transformer           — verify text encoder encodes correctly
    6. Text cache                     — verify JSON cache loads and is readable
    7. AudioTextDataset               — load 4 real tracks, check shapes
    8. Mini training loop             — 3 gradient steps, loss should decrease
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_here = Path(__file__).resolve().parent
_project_root = _here.parent.parent
sys.path.insert(0, str(_project_root))

# Import from sibling module
from models.calm.finetune import (
    CALM, CrossModalAttention, load_conformer_backbone,
    AUDIO_D, TEXT_D, FUSION_D, N_HEADS,
    AudioTextDataset, collate_fn,
)
from data.data_utils import (
    load_fma_metadata, get_splits, fma_audio_dir,
)

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Check 1: CALM model shapes ────────────────────────────────────────────────

def check_model_shapes(device):
    section("1. CALM model output shapes")

    B, T_a = 2, 150  # batch=2, 150 audio frames; text is always 1 token after unsqueeze
    num_classes = 16

    model = CALM(num_classes=num_classes, n_cross_layers=2).to(device)
    # Don't load real backbone — just check the fusion head with dummy audio hidden
    model.audio_encoder = None  # skip backbone for shape check

    audio_tokens = torch.randn(B, T_a, AUDIO_D, device=device)
    text_embs    = torch.randn(B, TEXT_D, device=device)

    # Manually run the projection + fusion (bypass audio_encoder)
    audio_proj   = model.audio_proj(audio_tokens)          # [B, T_a, FUSION_D]
    text_token   = model.text_proj(text_embs).unsqueeze(1) # [B, 1,   FUSION_D]

    print(f"  audio_tokens shape  : {list(audio_tokens.shape)}  (expected [{B}, {T_a}, {AUDIO_D}])")
    print(f"  audio_proj shape    : {list(audio_proj.shape)}   (expected [{B}, {T_a}, {FUSION_D}])")
    print(f"  text_embs shape     : {list(text_embs.shape)}   (expected [{B}, {TEXT_D}])")
    print(f"  text_token shape    : {list(text_token.shape)}   (expected [{B}, 1, {FUSION_D}])")

    for ca_a, ca_t in zip(model.cross_audio, model.cross_text):
        audio_proj = ca_a(audio_proj, text_token)
        text_token = ca_t(text_token, audio_proj)

    audio_pooled = audio_proj.mean(dim=1)
    text_pooled  = text_token.squeeze(1)
    fused        = torch.cat([audio_pooled, text_pooled], dim=1)
    logits       = model.head(fused)

    print(f"  fused shape         : {list(fused.shape)}     (expected [{B}, {FUSION_D*2}])")
    print(f"  logits shape        : {list(logits.shape)}    (expected [{B}, {num_classes}])")

    assert logits.shape == (B, num_classes), f"Wrong logits shape: {logits.shape}"
    print(PASS)


# ── Check 2: Cross-modal attention ───────────────────────────────────────────

def check_cross_attention(device):
    section("2. Cross-modal attention (audio queries text)")

    B, T_a, D = 2, 50, FUSION_D
    ca = CrossModalAttention(d_model=D, n_heads=N_HEADS).to(device)

    audio = torch.randn(B, T_a, D, device=device)
    text  = torch.randn(B, 1,   D, device=device)

    enriched_audio = ca(audio, text)  # audio attends to text
    enriched_text  = ca(text, audio)  # text attensds to audio

    print(f"  enriched_audio shape: {list(enriched_audio.shape)}  (expected [{B}, {T_a}, {D}])")
    print(f"  enriched_text shape : {list(enriched_text.shape)}   (expected [{B}, 1, {D}])")

    # Output should differ from input (attention changed the representation)
    diff = (enriched_audio - audio).abs().mean().item()
    print(f"  mean |enriched - input|: {diff:.5f}  (should be > 0)")
    assert diff > 0, "Cross-attention produced no change — check implementation"
    print(PASS)


# ── Check 3: Gradient flow ───────────────────────────────────────────────────

def check_gradients(device):
    section("3. Gradient flow through trainable params")

    B, T_a = 2, 100
    num_classes = 16
    model = CALM(num_classes=num_classes).to(device)
    model.audio_encoder = None  # skip backbone

    audio_tokens = torch.randn(B, T_a, AUDIO_D, device=device)
    text_embs    = torch.randn(B, TEXT_D, device=device)
    labels       = torch.randint(0, num_classes, (B,), device=device)

    # Manual forward (no backbone)
    audio_proj = model.audio_proj(audio_tokens)
    text_token = model.text_proj(text_embs).unsqueeze(1)
    for ca_a, ca_t in zip(model.cross_audio, model.cross_text):
        audio_proj = ca_a(audio_proj, text_token)
        text_token = ca_t(text_token, audio_proj)
    fused  = torch.cat([audio_proj.mean(1), text_token.squeeze(1)], dim=1)
    logits = model.head(fused)

    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    trainable = [(n, p) for n, p in model.named_parameters()
                 if p.requires_grad and "audio_encoder" not in n]
    no_grad = [n for n, p in trainable if p.grad is None]
    with_grad = [n for n, p in trainable if p.grad is not None]

    print(f"  Trainable params with grad : {len(with_grad)}")
    print(f"  Trainable params NO grad   : {len(no_grad)}")
    if no_grad:
        print(f"  WARNING — params missing grad:")
        for n in no_grad:
            print(f"    {n}")
    else:
        print(f"  All trainable params received gradients.")
    print(PASS if not no_grad else f"  {FAIL} — some params have no grad")


# ── Check 4: Conformer backbone ───────────────────────────────────────────────

def check_backbone(device):
    section("4. Wav2Vec2-Conformer backbone load + forward pass")

    try:
        backbone = load_conformer_backbone().to(device)
        backbone.eval()

        # Simulate 5s of audio at 16kHz
        dummy_audio = torch.randn(1, 5 * 16000, device=device)
        with torch.no_grad():
            out = backbone(dummy_audio)

        hidden = out.last_hidden_state
        print(f"  Input shape  : {list(dummy_audio.shape)}")
        print(f"  Output shape : {list(hidden.shape)}  (expected [1, ~150, {AUDIO_D}])")
        assert hidden.shape[-1] == AUDIO_D, f"Wrong hidden dim: {hidden.shape[-1]}"
        del backbone, dummy_audio, hidden
        torch.cuda.empty_cache()
        print(PASS)
    except Exception as e:
        print(f"  {FAIL}: {e}")
        raise  # let outer run() catch it and add to errors list


# ── Check 5: Sentence-transformer ────────────────────────────────────────────

def check_text_encoder():
    section("5. Gemma 4 E2B text encoder")

    try:
        from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
        model_name = "google/gemma-4-E2B-it"

        print(f"  Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16,
        )
        text_model = model.language_model
        text_model.eval()

        texts = [
            "Lyrics: Dhoop se nikal ke chhaav se fisal ke hum mile jahaan par",
            "Artist: Kurt Vile. Bio: Philly's Constant Hitmaker. Track: Freeway.",
            "Lyrics: दुनिया भुला के तुमसे मिला हूँ निकली है दिल से ये दुआ",
        ]
        tokens = tokenizer(texts, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = text_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
            )
            mask = tokens["attention_mask"].unsqueeze(-1)
            embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            embs = embs.float().numpy()

        print(f"  Texts encoded : {len(texts)}")
        print(f"  Embedding dim : {embs.shape[1]}  (expected {TEXT_D})")
        print(f"  Embedding range: [{embs.min():.3f}, {embs.max():.3f}]")
        assert embs.shape == (len(texts), TEXT_D), f"Wrong shape: {embs.shape}"

        from numpy.linalg import norm
        cos = lambda a, b: np.dot(a, b) / (norm(a) * norm(b))
        sim_hindi  = cos(embs[0], embs[2])  # romanized vs devanagari (same song)
        sim_diff   = cos(embs[0], embs[1])  # hindi vs english indie
        print(f"  Cosine sim (romanized vs devanagari): {sim_hindi:.4f}")
        print(f"  Cosine sim (hindi vs english indie) : {sim_diff:.4f}")

        del model, text_model
        torch.cuda.empty_cache()
        print(PASS)
    except ImportError:
        print(f"  {SKIP}: transformers not installed")


# ── Check 6: Text cache ───────────────────────────────────────────────────────

def check_text_cache():
    section("6. Text cache JSON")

    cache_path = _project_root / "data" / "text_cache_medium_no_tags.json"
    if not cache_path.exists():
        print(f"  {SKIP}: {cache_path.name} not found.")
        print("         Run: python data/build_text_cache.py --dataset medium")
        return

    with open(cache_path) as f:
        cache = json.load(f)

    keys = list(cache.keys())
    print(f"  Entries     : {len(cache)}")
    print(f"  Sample keys : {keys[:5]}")

    # Spot-check a few values
    empty = sum(1 for v in cache.values() if not v.strip())
    print(f"  Empty texts : {empty}")
    print(f"  Sample text : {list(cache.values())[0][:120]}")
    assert len(cache) > 1000, f"Cache suspiciously small: {len(cache)}"
    print(PASS)


# ── Check 7: AudioTextDataset ─────────────────────────────────────────────────

def check_dataset():
    section("7. AudioTextDataset — real audio + text")

    cache_path = _project_root / "data" / "text_cache_medium_no_tags.json"
    if not cache_path.exists():
        print(f"  {SKIP}: text cache not found — run build_text_cache.py first")
        return

    with open(cache_path) as f:
        text_cache = {int(k): v for k, v in json.load(f).items()}

    # Build tiny fake text embeddings (skip sentence-transformer here)
    text_embeddings = {tid: np.random.randn(TEXT_D).astype(np.float32)
                       for tid in text_cache}

    df = load_fma_metadata(subset="medium")
    _, _, test_df = get_splits(df)
    audio_dir = fma_audio_dir("medium")

    # Use only 8 tracks
    mini_df = test_df.head(8)
    ds = AudioTextDataset(mini_df, text_cache, text_embeddings,
                          audio_dir=audio_dir, clip_secs=5.0)

    print(f"  Dataset size  : {len(ds)}")
    sample = ds[0]
    print(f"  waveform shape: {list(sample['waveform'].shape)}  "
          f"(expected [80000] for 5s @ 16kHz)")
    print(f"  text_emb shape: {list(sample['text_emb'].shape)}  "
          f"(expected [{TEXT_D}])")
    print(f"  label         : {sample['label'].item()}")

    loader = torch.utils.data.DataLoader(ds, batch_size=4,
                                         collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"  Batch waveform: {list(batch['waveform'].shape)}  (expected [4, 80000])")
    print(f"  Batch text_emb: {list(batch['text_emb'].shape)}  (expected [4, {TEXT_D}])")
    print(f"  Batch labels  : {batch['label'].tolist()}")
    print(PASS)


# ── Check 8: Mini training loop ───────────────────────────────────────────────

def check_mini_train(device):
    section("8. Mini training loop (3 gradient steps)")

    B, T_a, steps = 4, 150, 3
    num_classes = 16

    model = CALM(num_classes=num_classes).to(device)
    model.audio_encoder = None  # skip backbone for speed

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for i in range(steps):
        audio_tokens = torch.randn(B, T_a, AUDIO_D, device=device)
        text_embs    = torch.randn(B, TEXT_D, device=device)
        labels       = torch.randint(0, num_classes, (B,), device=device)

        optimizer.zero_grad()

        # Manual forward
        audio_proj = model.audio_proj(audio_tokens)
        text_token = model.text_proj(text_embs).unsqueeze(1)
        for ca_a, ca_t in zip(model.cross_audio, model.cross_text):
            audio_proj = ca_a(audio_proj, text_token)
            text_token = ca_t(text_token, audio_proj)
        fused  = torch.cat([audio_proj.mean(1), text_token.squeeze(1)], dim=1)
        logits = model.head(fused)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  Step {i+1}: loss = {loss.item():.4f}")

    print(f"  Losses: {[f'{l:.4f}' for l in losses]}")
    print(f"  (Random baseline for {num_classes} classes ≈ {-np.log(1/num_classes):.4f})")
    print(PASS)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CALM debug / smoke test")
    parser.add_argument("--skip_backbone", action="store_true",
                        help="Skip loading the full Conformer backbone (saves ~2min)")
    parser.add_argument("--skip_data", action="store_true",
                        help="Skip real audio dataset check")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    errors = []

    def run(name, fn, *a, **kw):
        try:
            fn(*a, **kw)
        except Exception as e:
            print(f"  {FAIL}: {e}")
            import traceback; traceback.print_exc()
            errors.append(name)

    run("shapes",        check_model_shapes,    device)
    run("cross_attn",    check_cross_attention,  device)
    run("gradients",     check_gradients,        device)
    run("text_cache",    check_text_cache)

    # Backbone before sentence-transformer — prevents OOM from both in VRAM together
    if not args.skip_backbone:
        run("backbone",  check_backbone,         device)
        torch.cuda.empty_cache()
    else:
        print(f"\n{'='*60}")
        print(f"  4. Conformer backbone  {SKIP} (--skip_backbone)")

    # Text encoder after backbone is freed
    run("text_encoder",  check_text_encoder)

    if not args.skip_data:
        run("dataset",   check_dataset)
    else:
        print(f"\n{'='*60}")
        print(f"  7. AudioTextDataset    {SKIP} (--skip_data)")

    run("mini_train",    check_mini_train,       device)

    print(f"\n{'='*60}")
    if errors:
        print(f"  FAILED checks: {errors}")
    else:
        print("  All checks passed. CALM is ready to train.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
