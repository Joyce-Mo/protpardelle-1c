#!/bin/bash
set -euo pipefail

date
hostname

# ---------- environment ----------
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "protpardelle" ]; then
    conda activate protpardelle
fi

# ---------- paths (edit these) ----------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Directory of CATH-20 PDB files
PDB_DIR="${PDB_DIR:-/Users/joycemo/Documents/PhD/Rotation3/dataset/cath20/cath20-filtered-foldseek}"

# Where to save output representations
OUTPUT_BASE="${OUTPUT_BASE:-/Users/joycemo/Documents/PhD/Rotation3/dataset/cath20/cath20_protpardelle_reps}"

# Model config and checkpoint
CONFIG="${REPO_ROOT}/model_params/configs/cc89.yaml"
CHECKPOINT="${REPO_ROOT}/model_params/weights/cc89_epoch415.pth"

# ---------- device ----------
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE=cuda
elif python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE=mps
else
    DEVICE=cpu
fi
echo "Using device: ${DEVICE}"

# ---------- run ----------
echo "========== Extracting Protpardelle representations =========="
python "${REPO_ROOT}/representation_extraction/extract_representations.py" \
    --config-path "${CONFIG}" \
    --checkpoint-path "${CHECKPOINT}" \
    --pdb-dir "${PDB_DIR}" \
    --output-dir "${OUTPUT_BASE}/protpardelle" \
    --hook-targets transformer_output,patch_embedding,noise_conditioning \
    --save-tensors \
    --num-steps 100 \
    --device "${DEVICE}"
