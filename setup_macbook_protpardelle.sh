#!/usr/bin/env bash
set -euo pipefail

# MacBook setup script for Protpardelle following the upstream README flow.

usage() {
  cat <<'EOF'
Usage: setup_macbook_protpardelle.sh [options]

Options:
  --repo PATH               Path to protpardelle-1c repo
                            (default: /Users/joycemo/Documents/GitHub/protpardelle-1c)
  --mode MODE               standard | lock (default: standard)
  --env NAME                Conda env name for standard mode (default: protpardelle)
  --python VERSION          Python version for standard mode (default: 3.12)
  --env-dir DIR             ENV_DIR for lock mode (default: envs)
  --lock-python VERSION     Python for uv venv in lock mode (default: 3.10)
  --download-model-params   Run download_model_params.sh after install
  -h, --help                Show this help and exit
EOF
}

REPO="/Users/joycemo/Documents/GitHub/protpardelle-1c"
MODE="standard"
ENV_NAME="protpardelle"
PYTHON_VERSION="3.12"
ENV_DIR="envs"
LOCK_PYTHON="3.10"
DOWNLOAD_MODEL_PARAMS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --env-dir)
      ENV_DIR="$2"
      shift 2
      ;;
    --lock-python)
      LOCK_PYTHON="$2"
      shift 2
      ;;
    --download-model-params)
      DOWNLOAD_MODEL_PARAMS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$REPO" ]]; then
  echo "Repo not found: $REPO" >&2
  exit 1
fi

if [[ "$MODE" != "standard" && "$MODE" != "lock" ]]; then
  echo "--mode must be one of: standard, lock" >&2
  exit 1
fi

if [[ "$MODE" == "standard" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH." >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"

  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists; reusing it."
  else
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" --yes
  fi

  conda activate "$ENV_NAME"

  (
    cd "$REPO"
    bash setup.sh
    if [[ "$DOWNLOAD_MODEL_PARAMS" -eq 1 ]]; then
      bash download_model_params.sh
    fi
  )
else
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found in PATH. Install uv first for --mode lock." >&2
    exit 1
  fi

  (
    cd "$REPO"
    mkdir -p "$ENV_DIR"
    uv venv "$ENV_DIR/protpardelle" -p "python${LOCK_PYTHON}"
    # shellcheck disable=SC1091
    source "$ENV_DIR/protpardelle/bin/activate"
    uv pip sync uv_indexes.txt uv.lock --index-strategy=unsafe-best-match
    uv pip install -e . --no-deps
    if [[ "$DOWNLOAD_MODEL_PARAMS" -eq 1 ]]; then
      bash download_model_params.sh
    fi
  )
fi

echo
echo "Setup complete."
if [[ "$MODE" == "standard" ]]; then
  echo "Activate with: conda activate $ENV_NAME"
else
  echo "Activate with: source $REPO/$ENV_DIR/protpardelle/bin/activate"
fi
echo "Then run: python -c 'import protpardelle; print(\"ok\")'"
echo "If needed, run download_model_params.sh to fetch model weights/configs."
echo "Also install Foldseek separately and add it to PATH."
