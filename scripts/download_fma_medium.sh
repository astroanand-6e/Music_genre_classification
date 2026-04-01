#!/usr/bin/env bash
# Download FMA-Medium dataset (25k tracks, 16 genres, ~22 GB)
# Source: https://github.com/mdeff/fma
#
# Usage:
#   bash scripts/download_fma_medium.sh
#   bash scripts/download_fma_medium.sh --output_dir /path/to/dir

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${BASE_DIR}"

# Parse optional --output_dir flag
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "=== FMA-Medium Download ==="
echo "Output: ${OUTPUT_DIR}"
echo ""

# 1. Audio (fma_medium.zip, ~22 GB)
AUDIO_ZIP="${OUTPUT_DIR}/fma_medium.zip"
if [ ! -f "${OUTPUT_DIR}/fma_medium/.done" ]; then
  echo "[1/3] Downloading fma_medium.zip (~22 GB)..."
  wget -c -O "${AUDIO_ZIP}" \
    "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
  echo "[1/3] Extracting..."
  unzip -q "${AUDIO_ZIP}" -d "${OUTPUT_DIR}/"
  touch "${OUTPUT_DIR}/fma_medium/.done"
  rm -f "${AUDIO_ZIP}"
  echo "[1/3] Done."
else
  echo "[1/3] fma_medium already extracted, skipping."
fi

# 2. Metadata (fma_metadata.zip, ~342 MB)
if [ ! -f "${OUTPUT_DIR}/fma_metadata/tracks.csv" ]; then
  echo "[2/3] Downloading fma_metadata.zip..."
  META_ZIP="${OUTPUT_DIR}/fma_metadata.zip"
  wget -c -O "${META_ZIP}" \
    "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
  echo "[2/3] Extracting..."
  unzip -q "${META_ZIP}" -d "${OUTPUT_DIR}/"
  rm -f "${META_ZIP}"
  echo "[2/3] Done."
else
  echo "[2/3] fma_metadata already present, skipping."
fi

# 3. Verify track count
TRACK_COUNT=$(find "${OUTPUT_DIR}/fma_medium" -name "*.mp3" | wc -l)
echo "[3/3] Track count: ${TRACK_COUNT} (expected ~25,000)"

echo ""
echo "=== Done! ==="
echo "Audio:    ${OUTPUT_DIR}/fma_medium/"
echo "Metadata: ${OUTPUT_DIR}/fma_metadata/"
