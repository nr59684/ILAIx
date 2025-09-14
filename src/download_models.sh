#!/usr/bin/env bash
# -------------------------------------------------------------
# download_models.sh  —  Fetches *model.zip* and unpacks it.
# -------------------------------------------------------------
# ZIP layout expected:
#   model.zip
#   ├── modernBERT/
#   │   ├── model.safetensors / model.pt / config.json / …
#   ├── LegalBERT/
#   ├── RoBERTa/
#   └── ELECTRA/
# After extraction the folder tree will be under ./model/
# so you can reference e.g. MODEL_PATH=model/modernBERT/model.pt
# -------------------------------------------------------------
set -euo pipefail

# Google Drive ID for *model.zip* — replace once uploaded
FILE_ID="MODEL_ZIP_FILE_ID"
ZIP_NAME="model.zip"

# Destination folder
MODEL_DIR="$(dirname "$0")/../model"
mkdir -p "$MODEL_DIR"

# Helper: install gdown only if missing
command -v gdown >/dev/null 2>&1 || {
  echo "Installing gdown…" >&2;
  pip install --quiet --no-cache-dir gdown;
}

# 1. Download ZIP if not present
if [[ -f "$ZIP_NAME" ]]; then
  echo "✓ $ZIP_NAME already exists — skipping download"
else
  echo "↓ Downloading $ZIP_NAME from Google Drive…"
  gdown --id "$FILE_ID" -O "$ZIP_NAME"
fi

# 2. Extract — only if at least one subfolder is missing
if [[ -d "$MODEL_DIR/modernBERT" && -d "$MODEL_DIR/LegalBERT" && -d "$MODEL_DIR/RoBERTa" ]]; then
  echo "✓ Model folders already present — skipping unzip"
else
  echo "⇡ Extracting checkpoints into $MODEL_DIR…"
  unzip -q -o "$ZIP_NAME" -d "$MODEL_DIR"
fi

echo "Done. Directory tree under $MODEL_DIR:"
find "$MODEL_DIR" -maxdepth 2 -type f | head
