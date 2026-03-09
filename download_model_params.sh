#!/bin/bash

set -euo pipefail

for cmd in wget curl aria2c hf; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: $cmd is required but not installed." >&2
        exit 1
    fi
done

mkdir -p model_params

# Download Protpadelle model params
echo "Downloading Protpadelle model parameters..."
zenodo_url="https://zenodo.org/records/16817230/files/protpardelle-1c.tar.gz?download=1"
out_tar="protpardelle-1c.tar.gz"

if ! aria2c \
    --max-connection-per-server=1 \
    --split=1 \
    --min-split-size=1G \
    --header='User-Agent: Mozilla/5.0' \
    --header='Referer: https://zenodo.org/records/16817230' \
    --auto-file-renaming=false \
    -o "$out_tar" "$zenodo_url"; then
    echo "aria2 download failed, retrying with wget..."
    if ! wget -O "$out_tar" --referer='https://zenodo.org/records/16817230' "$zenodo_url"; then
        echo "wget failed, retrying with curl..."
        curl -fL -A 'Mozilla/5.0' -e 'https://zenodo.org/records/16817230' -o "$out_tar" "$zenodo_url"
    fi
fi

tar -xzvf "$out_tar" --strip-components=1 # there will be a model_params/ directory
rm "$out_tar"
echo "Protpadelle model parameters downloaded."

# Download ESMFold model
echo "Downloading ESMFold model..."
mkdir -p model_params/ESMFold
hf download facebook/esmfold_v1 --local-dir model_params/ESMFold
echo "ESMFold model downloaded."

# Download ProteinMPNN weights
echo "Downloading ProteinMPNN weights..."
mkdir -p model_params/ProteinMPNN
tmp="$(mktemp -d)"
repo_url="https://github.com/dauparas/ProteinMPNN.git"
branch="main"
folder="vanilla_model_weights"

git_ver="$(git --version | awk '{print $3}')"
IFS=. read -r M m p <<<"$git_ver"
: "${m:=0}"
: "${p:=0}"
# Check Git version >= 2.25.0
if ((M > 2 || (M == 2 && (m > 25 || (m == 25 && p >= 0))))); then
    git clone --depth=1 --filter=tree:0 --sparse "$repo_url" "$tmp"
    git -C "$tmp" sparse-checkout set "$folder"
    git -C "$tmp" checkout "$branch"
else
    git clone --depth=1 --single-branch --branch "$branch" "$repo_url" "$tmp" \
        || git clone --depth=1 "$repo_url" "$tmp"
fi

mv "$tmp/$folder" model_params/ProteinMPNN/
rm -rf "$tmp"
echo "ProteinMPNN weights downloaded."

# Download LigandMPNN weights
echo "Downloading LigandMPNN weights..."
mkdir -p model_params/LigandMPNN
curl -fsSL https://raw.githubusercontent.com/dauparas/LigandMPNN/main/get_model_params.sh | bash -s -- ./model_params/LigandMPNN
echo "LigandMPNN weights downloaded."
