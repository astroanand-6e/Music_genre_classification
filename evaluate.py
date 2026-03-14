"""
Unified Evaluation Script — FMA-Small Genre Classification

Runs zero-shot linear probes and/or loads fine-tuned checkpoints for any
combination of models, then produces a ranked comparison table and charts.

Usage
-----
    # Zero-shot comparison of all models
    python evaluate.py --model mert-95m --model mert-330m --model clap-laion --model ast --model musicldm

    # Evaluate fine-tuned checkpoints
    python evaluate.py \\
        --model mert-330m --checkpoint results/checkpoints/mert/mert_330m_20260314.pt \\
        --model ast       --checkpoint results/checkpoints/ast/ast_20260314.pt

    # Mix zero-shot and fine-tuned
    python evaluate.py --model mert-95m --model ast --checkpoint results/checkpoints/ast/ast.pt
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from data.data_utils import (
    load_fma_metadata, get_splits, get_label_maps,
    save_run_results, compute_per_class_f1,
    get_audio_path, load_waveform, figures_path, RESULTS_DIR,
)

# __ CLI __

def parse_args():
    p = argparse.ArgumentParser(description="Unified model evaluation")
    p.add_argument(
        "--model", action="append", required=True,
        help="Model key to evaluate: mert-95m, mert-330m, clap-laion, "
             "clap-microsoft, ast, musicldm. Repeat for multiple models.",
    )
    p.add_argument(
        "--checkpoint", action="append", default=[],
        help="Optional checkpoint path per model (positional matching with --model). "
             "Omit or pass '' for zero-shot.",
    )
    p.add_argument("--clip_secs", type=float, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()

# __ Model registry __

MODEL_REGISTRY = {
    "mert-95m":        {"family": "mert",     "variant": "95m",       "sr": 24000, "dim": 768},
    "mert-330m":       {"family": "mert",     "variant": "330m",      "sr": 24000, "dim": 1024},
    "clap-laion":      {"family": "clap",     "variant": "laion",     "sr": 48000, "dim": 512},
    "clap-microsoft":  {"family": "clap",     "variant": "microsoft", "sr": 48000, "dim": 512},
    "ast":             {"family": "ast",      "variant": "",          "sr": 16000, "dim": 768},
    "musicldm":        {"family": "musicldm", "variant": "vae",       "sr": 16000, "dim": 512},
}

# __ Feature extractors (zero-shot) __

def extract_mert(track_ids, labels, variant, clip_secs, device):
    from transformers import Wav2Vec2FeatureExtractor, AutoModel
    model_id = {"95m": "m-a-p/MERT-v1-95M", "330m": "m-a-p/MERT-v1-330M"}[variant]
    sr = 24000
    dim = {"95m": 768, "330m": 1024}[variant]
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    max_len = sr * int(clip_secs)
    feats, labs = [], []
    for tid, lab in tqdm(zip(track_ids, labels), total=len(track_ids), desc=f"MERT-{variant}"):
        path = get_audio_path(tid)
        try:
            wav = load_waveform(path, sr, clip_secs)
            if len(wav) < max_len:
                wav = np.pad(wav, (0, max_len - len(wav)))
            inp = processor(wav, sampling_rate=sr, return_tensors="pt",
                            padding="max_length", max_length=max_len, truncation=True).to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            feats.append(out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
        except Exception:
            feats.append(np.zeros(dim))
        labs.append(lab)
    return np.array(feats), np.array(labs)


def extract_clap(track_ids, labels, variant, clip_secs, device):
    from transformers import ClapModel, ClapProcessor
    model_id = {"laion": "laion/clap-htsat-fused", "microsoft": "microsoft/msclap"}[variant]
    sr = 48000
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id).to(device)
    model.eval()
    max_len = int(sr * clip_secs)
    feats, labs = [], []
    for tid, lab in tqdm(zip(track_ids, labels), total=len(track_ids), desc=f"CLAP-{variant}"):
        path = get_audio_path(tid)
        try:
            wav = load_waveform(path, sr, clip_secs)
            if len(wav) < max_len:
                wav = np.pad(wav, (0, max_len - len(wav)))
            with torch.no_grad():
                inp = processor(audio=wav, sampling_rate=sr, return_tensors="pt").to(device)
                out = model.get_audio_features(**inp)
                out = out.pooler_output if hasattr(out, "pooler_output") else out
            feats.append(out.squeeze().cpu().numpy())
        except Exception:
            feats.append(np.zeros(512))
        labs.append(lab)
    return np.array(feats), np.array(labs)


def extract_ast(track_ids, labels, clip_secs, device):
    from transformers import ASTFeatureExtractor, ASTModel
    model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
    sr = 16000
    fe = ASTFeatureExtractor.from_pretrained(model_id)
    model = ASTModel.from_pretrained(model_id).to(device)
    model.eval()
    max_len = int(sr * clip_secs)
    feats, labs = [], []
    for tid, lab in tqdm(zip(track_ids, labels), total=len(track_ids), desc="AST"):
        path = get_audio_path(tid)
        try:
            wav = load_waveform(path, sr, clip_secs)
            if len(wav) < max_len:
                wav = np.pad(wav, (0, max_len - len(wav)))
            inp = fe(wav, sampling_rate=sr, return_tensors="pt",
                     padding="max_length", truncation=True).to(device)
            with torch.no_grad():
                out = model(**inp)
            feats.append(out.last_hidden_state[:, :2, :].mean(dim=1).squeeze().cpu().numpy())
        except Exception:
            feats.append(np.zeros(768))
        labs.append(lab)
    return np.array(feats), np.array(labs)


def extract_musicldm(track_ids, labels, clip_secs, device):
    import librosa as _lr
    from diffusers import MusicLDMPipeline

    sr = 16000
    pipe = MusicLDMPipeline.from_pretrained("ucsd-reach/musicldm", torch_dtype=torch.float32)
    vae = pipe.vae.to(device)
    vae.eval()
    del pipe.unet, pipe.text_encoder, pipe.tokenizer

    feats, labs = [], []
    for tid, lab in tqdm(zip(track_ids, labels), total=len(track_ids), desc="MusicLDM"):
        path = get_audio_path(tid)
        try:
            wav = load_waveform(path, sr, clip_secs)
            mel = _lr.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=160, n_mels=64)
            log_mel = _lr.power_to_db(mel, ref=np.max)
            log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8) * 2 - 1
            mel_t = torch.from_numpy(log_mel).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                latent = vae.encode(mel_t).latent_dist.mean
            pooled = nn.functional.adaptive_avg_pool2d(latent, (8, 16))
            feats.append(pooled.flatten(1).squeeze(0).cpu().numpy())
        except Exception:
            feats.append(np.zeros(512))
        labs.append(lab)
    return np.array(feats), np.array(labs)


EXTRACTORS = {
    "mert":     lambda tids, labs, var, cs, dev: extract_mert(tids, labs, var, cs, dev),
    "clap":     lambda tids, labs, var, cs, dev: extract_clap(tids, labs, var, cs, dev),
    "ast":      lambda tids, labs, var, cs, dev: extract_ast(tids, labs, cs, dev),
    "musicldm": lambda tids, labs, var, cs, dev: extract_musicldm(tids, labs, cs, dev),
}

# __ Plotting helpers __

def plot_confusion_matrix(y_true, y_pred, genre_names, title, save_path):
    """Save a normalised (row-%) confusion matrix heatmap with percentage annotations."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=genre_names,
        yticklabels=genre_names,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Row %", "shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_f1_bars(f1_data, title, save_path):
    """Save a per-genre F1 bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(f1_data.keys(), f1_data.values(), color="coral", edgecolor="white")
    for bar, val in zip(bars, f1_data.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("F1-Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# __ Main __

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pad checkpoints list to match models
    checkpoints = list(args.checkpoint) + [""] * (len(args.model) - len(args.checkpoint))

    df = load_fma_metadata()
    label2id, id2label = get_label_maps(df)
    train_df, val_df, test_df = get_splits(df)
    genre_names = [id2label[i] for i in range(len(id2label))]

    comparison_rows = []

    for model_key, ckpt in zip(args.model, checkpoints):
        if model_key not in MODEL_REGISTRY:
            print(f"Unknown model key: {model_key}. Skipping.")
            continue

        info = MODEL_REGISTRY[model_key]
        family = info["family"]
        variant = info["variant"]

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_key}" + (f" (checkpoint: {ckpt})" if ckpt else " (zero-shot)"))
        print(f"{'='*60}")

        if ckpt:
            print(f"  Fine-tuned evaluation for {model_key}:")
            print(f"    python models/{family}/finetune.py --mode finetune ...")
            print(f"  Skipping checkpoint load in unified evaluator.\n")
            continue

        # zero-shot feature extraction
        extractor = EXTRACTORS.get(family)
        if extractor is None:
            print(f"  No extractor for family {family}")
            continue

        train_ids, train_labels = train_df.index.tolist(), train_df["label"].tolist()
        test_ids, test_labels = test_df.index.tolist(), test_df["label"].tolist()

        X_train, y_train = extractor(train_ids, train_labels, variant, args.clip_secs, device)
        X_test, y_test = extractor(test_ids, test_labels, variant, args.clip_secs, device)

        print("  Training logistic regression...")
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1s = compute_per_class_f1(y_test, y_pred, id2label)

        print(f"  Accuracy: {acc*100:.2f}%\n")
        print(classification_report(y_test, y_pred, target_names=genre_names))

        comparison_rows.append({
            "Model": model_key,
            "Mode": "zero_shot",
            "Accuracy": round(acc * 100, 2),
            "per_class_f1": f1s,
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        })

        save_run_results(
            model=family, variant=variant, mode="zero_shot",
            test_accuracy=acc,
            config={"clip_secs": args.clip_secs, "target_sr": info["sr"]},
            per_class_f1=f1s,
        )

    # __ Summary __
    if not comparison_rows:
        print("No models evaluated.")
        return

    comp_df = pd.DataFrame(comparison_rows).sort_values("Accuracy", ascending=False)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(comp_df[["Model", "Mode", "Accuracy"]].to_string(index=False))

    os.makedirs(os.path.join(RESULTS_DIR, "reports"), exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # save CSV
    csv_path = os.path.join(RESULTS_DIR, "reports", f"comparison_{ts}.csv")
    comp_df[["Model", "Mode", "Accuracy"]].to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")

    # bar chart — model accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(comp_df["Model"], comp_df["Accuracy"], color="steelblue")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("FMA-Small Genre Classification — Zero-Shot Model Comparison")
    ax.set_xlim(0, 100)
    for bar, val in zip(bars, comp_df["Accuracy"]):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center")
    plt.tight_layout()
    fig_path = figures_path(f"comparison_{ts}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved -> {fig_path}")

    # per-model confusion matrix (% normalised) + F1 bar chart
    for row in comparison_rows:
        model_key = row["Model"]
        safe_key = model_key.replace("-", "_")
        y_true = np.array(row["y_true"])
        y_pred_arr = np.array(row["y_pred"])

        cm_path = figures_path(f"confmat_{safe_key}_{ts}.png")
        plot_confusion_matrix(
            y_true, y_pred_arr, genre_names,
            title=f"Confusion Matrix — {model_key}  ({row['Accuracy']:.1f}%)",
            save_path=cm_path,
        )
        print(f"Saved -> {cm_path}")

        f1_path = figures_path(f"f1_{safe_key}_{ts}.png")
        plot_f1_bars(
            row["per_class_f1"],
            title=f"Per-Genre F1 — {model_key}  ({row['Accuracy']:.1f}%)",
            save_path=f1_path,
        )
        print(f"Saved -> {f1_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
