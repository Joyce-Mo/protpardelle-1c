#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l mem_free=4G
#$ -l scratch=2G
#$ -l h_rt=24:00:00
#$ -r y
#$ -m bea
#$ -M joyce.mo@ucsf.edu
#$ -q gpu.q
#$ -l gpu_mem=16G

date
hostname

# ---------- modules ----------
if command -v module >/dev/null 2>&1; then
    ## module load gcc/12.4.0
    module load cuda
fi

# ---------- environment ----------
if [ -z "$ENV_DIR" ]; then
    conda activate protpardelle
else
    source "$ENV_DIR"/protpardelle/bin/activate
fi

# ---------- paths (edit these) ----------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Directory of CATH-20 PDB files
PDB_DIR="/wynton/home/rotation/jqmo/rotation3/datasets/cath20-filtered-foldseek"

# Where to save output representations
OUTPUT_BASE="/wynton/home/rotation/jqmo/rotation3/datasets/cath20_partprodelle_initial_reps"

# Model config and checkpoint
CONFIG="${REPO_ROOT}/model_params/configs/cc89.yaml"
CHECKPOINT="${REPO_ROOT}/model_params/weights/cc89_epoch415.pth"

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
    --device cuda

# ---------- end-of-job ----------
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
