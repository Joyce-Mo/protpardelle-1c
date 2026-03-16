# representation_extraction

`representation_extraction` is a toolkit for extracting intermediate neural network representations from Protpardelle at different modules and across the denoising schedule, with support for multiple stochastic samplers, PDB-conditioned encoding, per-residue structure alignment, and full-resolution tensor export.

## Purpose

Score-based diffusion models for protein structure generation are typically treated as black boxes: noisy coordinates go in, clean coordinates come out. But the internal representations computed at each denoising step contain rich information about what the model has learned about protein geometry, and how that understanding evolves as noise decreases. This toolkit makes those representations accessible for analysis.

During sampling, Protpardelle runs a U-ViT denoiser iteratively: at each denoising step the model processes progressively less-noisy coordinates through convolutional layers, a transformer, and back. `representation_extraction` uses PyTorch forward hooks to non-invasively capture hidden states from any combination of these internal modules at every step, producing a structured dataset that maps **(module, denoising step, sample)** to a feature vector.

The toolkit supports two operating modes:

1. **De novo mode** (default): Generate proteins from random noise and capture how representations form during the generative process.
2. **PDB mode** (`--pdb-dir`): Encode real protein structures by noising them to each sigma level and running the denoiser, producing per-residue representations aligned to the input structure.

Typical use cases:
- Visualize how representations change across the denoising trajectory (UMAP colored by step).
- Compare what different modules "see" at the same noise level.
- Quantify representation similarity between Protpardelle and other protein models (ESM, AlphaFold, Boltz).
- Study the effect of noise schedule, step scale, or sampler choice on internal features.
- Compare representation dynamics across samplers (ODE vs Langevin vs predictor-corrector).
- Map learned features back to protein structure for per-residue analysis.
- Probe which residues or structural motifs the model attends to at different noise levels.

## How It Works

### Architecture context

Protpardelle's structure denoiser (`model.struct_model`, a `CoordinateDenoiser`) wraps a `TimeCondUViT` — a U-Net with a Vision Transformer bottleneck, where every layer receives time/noise conditioning. The data flow is:

```
noisy_coords [B, L, 37, 3]
    |
    v
Karras preconditioning
    Normalize the noisy input by the noise variance so the network always
    sees inputs of roughly unit scale regardless of the current sigma.
    c_in = 1 / sqrt(sigma^2 + sigma_data^2)
    |
    v
down_conv blocks  (TimeCondResnetBlock layers, optional downsampling)
    Convolutional encoder. Each block applies:
      GroupNorm -> SiLU -> Conv1d -> + time_embedding -> GroupNorm -> SiLU -> Conv1d + residual
    The conv operates over the sequence dimension, treating each residue's
    atom features as channels: input shape [B, C_in, L*A] where C_in = 3 (xyz)
    and A = 37 (atom types). Produces feature maps [B, C, L*A] with C = model_dim.
    |
    v
to_patch_embedding  (rearrange + linear projection -> [B, N, D])
    Reshapes the conv output from [B, C, L*A] to per-residue tokens [B, L, C*A_patch]
    then linearly projects to the transformer dimension D (256 in cc89).
    This is where spatial (atom-level) features become token-level (residue-level).
    |
    v
TimeCondTransformer  (depth x [TimeCondAttention, TimeCondFeedForward])
    10 transformer layers (for cc58/cc89/cc91). Each layer consists of:
      1. TimeCondAttention: multi-head self-attention with time-conditioned
         LayerNorm. The noise level modulates the normalization statistics,
         so the same weights behave differently at high vs low noise.
      2. TimeCondFeedForward: MLP with time-conditioned LayerNorm.
    All layers share the same positional encoding (relative residue index).
    Output shape: [B, L, D] where D = 256.
    |
    v
from_patch  (linear + rearrange back to spatial)
    Projects [B, L, D] back to [B, C, L*A] for the convolutional decoder.
    |
    v
up_conv blocks  (with skip connections from down_conv)
    Convolutional decoder, symmetric to the encoder. Skip connections from
    corresponding down_conv blocks provide high-resolution spatial detail.
    |
    v
Karras postconditioning
    c_skip * input + c_out * network_output
    Blends the raw input with the network prediction using noise-level-dependent
    weights, so at very high noise the model defaults to the mean and at very
    low noise it trusts its prediction fully.
    -> denoised_coords [B, L, 37, 3]
```

**Noise schedule.** The noise level at each step is determined by a Karras-style schedule (Karras et al., NeurIPS 2022):
- `timestep` goes from 1 (maximum noise) to 0 (clean) over `num_steps` linearly spaced values.
- `sigma = sigma_data * (s_max^(1/rho) + (1-t) * (s_min^(1/rho) - s_max^(1/rho)))^rho`
- Default parameters: `sigma_data=10.0, s_min=0.001, s_max=80.0, rho=7.0`.
- At step 0, sigma ~ 800 Angstroms (pure noise). At step N-1, sigma ~ 0.01 (nearly clean).

**Score function.** The model predicts denoised coordinates x0 from noisy coordinates xt. The score (gradient of log probability) is derived as:
```
score = (xt - x0_predicted) / sigma
```
This score drives all sampler update rules.

### Operating modes

#### De novo mode (default)

The standard mode generates proteins from random Gaussian noise. Coordinates are initialized as `xt = randn * sigma_max` and iteratively denoised. At each step, hooks capture the model's internal state. This reveals how the model builds structure from nothing: what features emerge first, when secondary structure appears in the representations, etc.

#### PDB mode (`--pdb-dir`)

PDB mode encodes real protein structures. For each PDB file and each noise level in the schedule:

1. Load the structure via `load_feats_from_pdb()` to get atom coordinates, residue types, and chain information.
2. Add Gaussian noise: `xt = clean_coords + sigma * randn`.
3. Run one forward pass through the denoiser at that sigma.
4. Hooks capture per-residue representations aligned to the input structure.

This produces representations where **position i in the tensor corresponds exactly to residue i in the PDB**. The approach is analogous to masked language model encoding (like ESM) but in coordinate space: by corrupting the structure with calibrated noise and asking the model to denoise it, the model's internal activations reveal what it has learned about each residue's structural context.

The output includes:
- Per-residue tensors: `per_structure/<pdb_name>/<module>/step_NNNN.pt` with shape `[1, L, D]`.
- Metadata: `per_structure/<pdb_name>/metadata.pt` containing `aatype`, `residue_index`, `chain_index`, and `atom_positions`.
- B-factor PDBs: `bfactor_pdbs/<pdb>_<module>_step<N>.pdb` where the B-factor column encodes the L2 norm of each residue's representation vector, for visualization in PyMOL or ChimeraX.

### Sampler strategies

The script provides five denoising strategies via `--sampler`. All except `default` use a manual denoising loop (`run_manual_sampling()`) that calls `model.forward()` directly, giving full control over the update rule while still triggering all registered hooks. In PDB mode, only the forward pass is used (no iterative denoising between steps); each noise level is applied independently to the clean structure.

| Sampler | Update rule | Key parameters |
|---|---|---|
| `euler` | Deterministic ODE: `xt_next = xt + step_scale * (sigma_next - sigma) * score` | `--step-scale` |
| `stochastic` | Karras stochastic: inject noise proportional to gamma before denoising, then Euler step from inflated sigma | `--s-churn`, `--noise-scale`, `--s-t-min`, `--s-t-max` |
| `langevin` | Annealed Langevin dynamics: run K corrector MCMC steps at each noise level using the score, then one Euler predictor step | `--langevin-corrector-steps`, `--langevin-snr` |
| `predictor_corrector` | PC sampler: one Euler predictor step, then K Langevin corrector steps at the new noise level | `--langevin-corrector-steps`, `--langevin-snr` |
| `default` | Use `model.sample()` directly (the standard Protpardelle sampling path) | `--s-churn` via sampling config |

**Euler ODE** is the simplest and fastest sampler. It follows the probability flow ODE:
```
dx/dt = (x - D(x, sigma)) / sigma * d_sigma/dt
```
Discretized as: `xt_next = xt + step_scale * (sigma_next - sigma) * (xt - x0_pred) / sigma`. The `step_scale` parameter (default 1.2) controls how aggressively the score is followed; values > 1 reduce diversity but improve sample quality.

**Karras stochastic sampler** adds controlled noise at each step before denoising, which improves sample diversity and can correct accumulated errors. At each step:
```
gamma = s_churn / num_steps  (if s_t_min <= sigma <= s_t_max, else 0)
sigma_hat = sigma + gamma * sigma
xt_hat = xt + sqrt(sigma_hat^2 - sigma^2) * noise_scale * z,  z ~ N(0,I)
x0_pred = model(xt_hat, sigma_hat)
xt_next = xt_hat + step_scale * (sigma_next - sigma_hat) * score
```
This is equivalent to `model.sample()` with `s_churn > 0`.

**Annealed Langevin dynamics** (Song & Ermon, NeurIPS 2019; Song et al., ICLR 2021) performs score-based MCMC at each noise level to refine the sample before stepping to the next noise level:
```
for each corrector step k = 1..K:
    step_size = snr * sigma^2
    z ~ N(0, I)
    xt = xt - step_size * score(xt, sigma) + sqrt(2 * step_size) * z
# then one Euler predictor step to sigma_next
```
This refines the sample at the current noise level before stepping to the next. Higher `--langevin-snr` and more `--langevin-corrector-steps` produce more refined samples but require proportionally more forward passes. With K corrector steps, each denoising step requires K+1 forward passes (K corrector + 1 predictor), so 5 corrector steps with 100 denoising steps = 600 total forward passes.

**Predictor-corrector** reverses the order: first move to the next noise level (Euler predictor), then refine there (Langevin corrector). This is generally more stable for large step sizes because the corrector operates at the target noise level rather than the source.

### Hook mechanism

The extraction script uses PyTorch's `register_forward_hook()` and `register_forward_pre_hook()` to non-invasively intercept activations during the model's forward pass. This requires no modification to model code.

For each hooked module, the callback:
1. Detaches the activation from the computation graph and moves it to CPU.
2. Stores the full-resolution tensor (if `--save-tensors`) for per-residue analysis.
3. Mean-pools spatial dimensions to produce a single `[B, D]` vector for UMAP/PCA.

The `pool_representation()` function handles different tensor shapes:
- `[B, L, D]` (transformer, patch_embedding): mean over L (sequence length) -> `[B, D]`
- `[B, C, L, A]` (conv blocks): mean over L and A -> `[B, C]`
- `[B, D]` (noise_conditioning): identity (already pooled)

### Hook targets

| Target name | Module hooked | Tensor shape | What it captures |
|---|---|---|---|
| `transformer_output` | `struct_model.net.transformer` | `[B, L, 256]` | Final transformer hidden state after all 10 layers of attention and feedforward. This is the model's highest-level "understanding" of the structure at the current noise level. |
| `per_layer_attn` | `transformer.layers[i][0]` | `[B, L, 256]` | Per-layer attention output. Use with `--layers 0,4,9` to compare early, middle, and late layers. Early layers capture local patterns; late layers capture global structure. |
| `per_layer_ff` | `transformer.layers[i][1]` | `[B, L, 256]` | Per-layer feed-forward output. The FF block applies a nonlinear transformation after attention; comparing attn vs FF at the same layer shows what the MLP adds. |
| `patch_embedding` | `struct_model.net.to_patch_embedding` | `[B, L, 256]` | Input to the transformer, after conv encoding and patching. This is what the convolutional encoder extracts before any global attention. |
| `from_patch` | `struct_model.net.from_patch` | `[B, L, 256]` | Transformer output projected back to spatial representation for the conv decoder. |
| `down_conv` | `struct_model.net.down_conv[i][j]` | `[B, C, L*A]` | Each downsampling ResNet block. Multiple blocks are registered (one per conv layer). |
| `up_conv` | `struct_model.net.up_conv[i][j]` | `[B, C, L*A]` | Each upsampling ResNet block. Receives skip connections from corresponding down_conv. |
| `noise_conditioning` | `struct_model.noise_block` | `[B, 1, 1024]` | Time/noise embedding vector. A single vector per sample encoding the current sigma through sinusoidal positional encoding + MLP. This is structure-independent: it only encodes "how noisy is this input." |
| `coord_denoiser_input` | `struct_model` (pre-hook) | `[B, L, 37, 3]` | Raw noisy coordinates entering the denoiser, before Karras preconditioning. |
| `coord_denoiser_output` | `struct_model` | `[B, L, 37, 3]` | Denoised coordinate prediction (x0_hat), after Karras postconditioning. |

Multiple targets can be combined: `--hook-targets transformer_output,patch_embedding,noise_conditioning`.

### Call-to-step alignment

The model may be called more than once per denoising step (e.g. self-conditioning passes, Langevin corrector steps). The script aligns hook calls to denoising steps by splitting calls into `num_steps` groups and taking the last call in each group (which corresponds to the final state at that noise level).

## Interpreting the Output

### UMAP plots

Each hooked module produces its own UMAP visualization under `umap/`. UMAP (Uniform Manifold Approximation and Projection) is a nonlinear dimensionality reduction that projects the high-dimensional feature vectors (e.g. 256D) down to 2D for visualization.

**How to read UMAP plots:**
- The axes (UMAP-1, UMAP-2) are abstract coordinates with no direct physical meaning. UMAP is a nonlinear projection — it preserves neighborhood structure from the high-dimensional space, not distances or directions.
- **Distances between points**: nearby points have similar representations in the original high-dimensional space. Distant points are dissimilar.
- **Cluster structure**: groups of nearby points indicate the model encodes those inputs similarly.
- **Color gradients**: a smooth color gradient (e.g. step 0 to step 49) along a trajectory means the representation changes continuously.
- The specific x/y values and orientation are arbitrary — UMAP could rotate, mirror, or non-linearly warp the plot and the information content would be identical. Only relative positions matter.

**What each plot type reveals:**

**`umap_<module>_by_step.png`** — Points colored by denoising step (viridis: dark purple = step 0 / high noise, yellow = final step / low noise).

- A smooth arc ordered by step (as in `transformer_output`) means the representation changes monotonically and continuously as noise decreases. This is the expected behavior for a well-trained denoiser: the model's internal state should evolve smoothly from "I see noise" to "I see a protein."
- Clustered early steps that spread apart at late steps indicate that the model can only discriminate between different structures at low noise levels.
- A compact blob at early steps means all inputs look the same to the model when heavily noised.

**`umap_<module>_by_structure.png`** (PDB mode only) — Points colored by input PDB structure.

- Structures that overlap at early steps but separate at late steps confirm that structural identity emerges gradually during denoising.
- Structures that separate even at early steps (as in `patch_embedding`) indicate that the convolutional encoder extracts structure-specific features before global attention.
- The degree of separation between structures at a given noise level indicates how much structural information the model retains from the noisy input at that noise level.

### What different modules reveal

**`transformer_output`**: The model's highest-level representation. In practice, this shows a smooth denoising trajectory where all structures cluster together at high noise and separate at low noise. This is the most informative target for comparing what the model has learned overall.

**`patch_embedding`**: The input to the transformer, after convolutional encoding. This reveals what information is available to the transformer before any global attention. If structures separate in patch_embedding but not in transformer_output, the transformer may be discarding or averaging out local features. If they separate in transformer_output but not in patch_embedding, the transformer is learning global structure from local features.

**`noise_conditioning`**: The time/noise embedding. This should form a perfect 1D curve ordered by step (since it's a deterministic function of sigma), with all structures overlapping exactly (since it doesn't depend on the input). If you see deviations from this, something unexpected is happening in the noise embedding.

**`per_layer_attn` / `per_layer_ff`**: Comparing across layers shows how representations develop through the transformer depth. Early layers (0-2) typically capture local geometric patterns; middle layers (3-6) build medium-range contacts; late layers (7-9) refine global structure.

**`down_conv` / `up_conv`**: The convolutional encoder/decoder representations operate at the atom level rather than the residue level. Useful for studying how local chemical environment is encoded.

### Aligning representations to protein structure

In PDB mode, representations are naturally aligned to the input structure: position `i` in the tensor corresponds to residue `i` in the PDB. There are three ways to map representations back to structure:

**1. Direct tensor indexing:**
```python
import torch

# Load per-residue representations
rep = torch.load("per_structure/1SMG/transformer_output/step_0025.pt")
meta = torch.load("per_structure/1SMG/metadata.pt")

# rep.shape: [1, 90, 256] — 90 residues, 256D representation each
# rep[0, i, :] is the 256D representation of residue i
# meta["aatype"][i] is the amino acid type (0=ALA, 1=ARG, ..., 19=VAL)
# meta["residue_index"][i] is the PDB residue number
# meta["chain_index"][i] is the chain index (0, 1, 2, ...)
```

**2. B-factor PDB visualization:**

The `bfactor_pdbs/` directory contains PDB files where the B-factor column encodes the L2 norm of each residue's representation vector, normalized to 0-100. Open in PyMOL:
```
load bfactor_pdbs/1SMG_transformer_output_step0025.pdb
spectrum b, blue_white_red
```
Or in ChimeraX:
```
open bfactor_pdbs/1SMG_transformer_output_step0025.pdb
color bfactor palette blue:white:red
```
Residues with high B-factor have large representation norms — these are residues where the model's internal activation is strongest, which may indicate structurally important or information-rich positions.

**3. Downstream analysis (probing, regression):**
```python
import torch
import numpy as np

# Collect representations across noise levels for one structure
reps = []
for step in range(50):
    r = torch.load(f"per_structure/1SMG/transformer_output/step_{step:04d}.pt")
    reps.append(r[0])  # [L, D]
reps = torch.stack(reps)  # [50, L, D]

# Now reps[t, i, :] is residue i's representation at noise level t.
# You can:
# - Train a linear probe to predict secondary structure from reps
# - Compute cosine similarity between residue pairs across steps
# - Regress structural properties (SASA, contacts) from representations
# - Compare with ESM or AlphaFold representations of the same structure
```

## Directory Contents

| File | Description |
|---|---|
| `extract_representations.py` | Main extraction script with multi-module hooks, stochastic samplers, PDB encoding, tensor export, per-module UMAP. |
| `extract_podelle_representations.py` | Original single-target extraction script (kept for reference). |
| `README.md` | This file. |

## Quick Start

### De novo mode (generate from noise)

```bash
# Basic: extract transformer output using Euler ODE sampler
python representation_extraction/extract_representations.py \
    --config-path model_params/configs/cc58.yaml \
    --checkpoint-path model_params/weights/cc58_epoch416.pth \
    --output-dir results/representations \
    --num-samples 8 --length 128 --num-steps 100 --device cuda

# Multi-module with full tensor export
python representation_extraction/extract_representations.py \
    --config-path model_params/configs/cc58.yaml \
    --checkpoint-path model_params/weights/cc58_epoch416.pth \
    --output-dir results/representations_tensors \
    --hook-targets transformer_output,patch_embedding,noise_conditioning \
    --save-tensors \
    --num-samples 8 --length 128 --num-steps 100 --device cuda

# Annealed Langevin dynamics with 5 corrector steps
python representation_extraction/extract_representations.py \
    --config-path model_params/configs/cc58.yaml \
    --checkpoint-path model_params/weights/cc58_epoch416.pth \
    --output-dir results/representations_langevin \
    --sampler langevin --langevin-corrector-steps 5 --langevin-snr 0.16 \
    --save-tensors \
    --num-samples 8 --length 128 --num-steps 100 --device cuda

# Predictor-corrector sampler
python representation_extraction/extract_representations.py \
    --config-path model_params/configs/cc58.yaml \
    --checkpoint-path model_params/weights/cc58_epoch416.pth \
    --output-dir results/representations_pc \
    --sampler predictor_corrector --langevin-corrector-steps 3 --langevin-snr 0.1 \
    --num-samples 8 --length 128 --num-steps 100 --device cuda

# Stochastic Karras sampler (equivalent to model.sample with s_churn > 0)
python representation_extraction/extract_representations.py \
    --config-path model_params/configs/cc58.yaml \
    --checkpoint-path model_params/weights/cc58_epoch416.pth \
    --output-dir results/representations_stochastic \
    --sampler stochastic --s-churn 200 \
    --num-samples 8 --length 128 --num-steps 100 --device cuda

# Per-layer attention analysis at selected transformer depths
python representation_extraction/extract_representations.py \
    --config-path model_params/configs/cc58.yaml \
    --checkpoint-path model_params/weights/cc58_epoch416.pth \
    --output-dir results/representations_layers \
    --hook-targets per_layer_attn --layers 0,4,9 \
    --num-steps 100 --device cuda

# All-atom model (cc89/cc91):
python representation_extraction/extract_representations.py \
    --config-path model_params/configs/cc89.yaml \
    --checkpoint-path model_params/weights/cc89_epoch415.pth \
    --output-dir results/representations_allatom \
    --hook-targets transformer_output,coord_denoiser_output \
    --save-tensors \
    --num-steps 100 --device cuda

# Use model.sample() directly (standard Protpardelle sampling path, no custom sampler)
python representation_extraction/extract_representations.py \
    --config-path model_params/configs/cc58.yaml \
    --checkpoint-path model_params/weights/cc58_epoch416.pth \
    --output-dir results/representations_default \
    --sampler default --s-churn 50 \
    --num-steps 100 --device cuda
```

### PDB mode (encode real structures)

```bash
# Encode PDB structures with per-residue representations
python representation_extraction/extract_representations.py \
    --config-path model_params/configs/cc89.yaml \
    --checkpoint-path model_params/weights/cc89_epoch415.pth \
    --output-dir results/march_rep_test \
    --pdb-dir examples/march_rep_test \
    --hook-targets transformer_output,patch_embedding,noise_conditioning \
    --save-tensors \
    --num-steps 50 --device mps
```

This loads all `.pdb` files from the directory, noises each one at 50 sigma levels, and saves:
- Per-residue tensors aligned to each input structure
- B-factor PDBs for visualization
- Per-module UMAP plots colored by denoising step and by structure

## Output Format

### Overview

A typical run produces this directory structure:

```
results/
├── architecture_summary.txt           # Model tree with [HOOK] annotations
├── representations.npz                # Combined pooled features (zero-padded to max dim)
├── representations_<module>.npz       # Per-module pooled features at native dim
├── representations_index.csv          # Human-readable index
├── embeddings/                        # (--save-tensors) Full-resolution tensors
│   ├── manifest.pt
│   ├── trajectories/
│   │   ├── xt_traj.pt
│   │   └── x0_traj.pt
│   └── <module>/
│       ├── step_0000.pt
│       └── ...
├── per_structure/                     # (PDB mode + --save-tensors) Per-residue aligned tensors
│   └── <pdb_name>/
│       ├── metadata.pt
│       └── <module>/
│           ├── step_0000.pt
│           └── ...
├── bfactor_pdbs/                      # (PDB mode + --save-tensors) B-factor colored PDBs
│   └── <pdb>_<module>_step<N>.pdb
└── umap/                              # Per-module UMAP plots and CSVs
    ├── umap_<module>_by_step.png
    ├── umap_<module>_by_structure.png # (PDB mode only)
    └── umap_<module>.csv
```

### `representations.npz`

```python
data = np.load("representations.npz", allow_pickle=True)
data["features"]           # [N_total, D_max] float32 — pooled features, zero-padded to max dim
data["step_idx"]           # [N_total] int32 — denoising step (0 = noisiest)
data["sample_idx"]         # [N_total] int32 — sample/PDB index within batch
data["module_idx"]         # [N_total] int32 — module index
data["module_names"]       # [num_modules] str — maps module_idx -> name
data["module_native_dims"] # [num_modules] int32 — native dimension per module (for unpadding)
data["num_steps"]          # [1] int32
data["num_samples"]        # [1] int32
data["sampler"]            # [1] str — sampler strategy used
data["pdb_names"]          # [num_pdbs] str — (PDB mode only) input PDB names
```

Where `N_total = num_modules * num_steps * num_samples`. Different modules may have different native embedding dimensions (e.g. `noise_conditioning` = 1024D, `transformer_output` = 256D). The combined file zero-pads all features to the maximum dimension; use `module_native_dims` to recover the original sizes.

### `representations_<module>.npz`

Per-module files at native dimension (no zero-padding):

```python
data = np.load("representations_transformer_output.npz", allow_pickle=True)
data["features"]    # [num_steps * num_samples, 256] float32
data["step_idx"]    # [num_steps * num_samples] int32
data["sample_idx"]  # [num_steps * num_samples] int32
data["pdb_names"]   # [num_pdbs] str (PDB mode only)
```

### `representations_index.csv`

| module | step_idx | sample_idx |
|---|---|---|
| transformer_output | 0 | 0 |
| transformer_output | 0 | 1 |
| ... | ... | ... |
| patch_embedding | 99 | 7 |

### `embeddings/` directory (with `--save-tensors`)

```
embeddings/
├── manifest.pt                    # Dict: module -> [step_paths], + _meta, _xt_traj, _x0_traj
├── trajectories/
│   ├── xt_traj.pt                 # List[Tensor] — noisy coords at each step
│   └── x0_traj.pt                 # List[Tensor] — predicted clean coords at each step
├── transformer_output/
│   ├── step_0000.pt               # Full tensor [B, L, D] at step 0
│   ├── step_0001.pt
│   └── ...
├── patch_embedding/
│   ├── step_0000.pt
│   └── ...
└── noise_conditioning/
    ├── step_0000.pt               # [B, 1, D_time] noise embedding
    └── ...
```

**Loading saved tensors:**
```python
import torch

# Load manifest
manifest = torch.load("embeddings/manifest.pt", weights_only=False)
meta = manifest["_meta"]  # sampler, num_steps, seed, etc.

# Load a specific module at a specific step
rep = torch.load("embeddings/transformer_output/step_0050.pt")
# rep.shape: [B, L, D] — full token-level representations

# Load coordinate trajectories
xt_traj = torch.load("embeddings/trajectories/xt_traj.pt")  # list of [B, L, 37, 3]
x0_traj = torch.load("embeddings/trajectories/x0_traj.pt")
```

### `per_structure/` directory (PDB mode + `--save-tensors`)

```
per_structure/
└── 1SMG/
    ├── metadata.pt                          # {pdb_name, aatype, residue_index, chain_index, atom_positions, num_residues}
    ├── transformer_output/
    │   ├── step_0000.pt                     # [1, 90, 256] — 90 residues, 256D, at noise level 0
    │   ├── step_0001.pt
    │   └── ...
    ├── patch_embedding/
    │   └── ...
    └── noise_conditioning/
        └── ...
```

### `bfactor_pdbs/` directory (PDB mode + `--save-tensors`)

PDB files where the B-factor column encodes representation magnitude per residue. Saved at 5 representative denoising steps (start, 25%, 50%, 75%, end) for each (structure, module) pair.

The B-factor value is the L2 norm of the residue's representation vector, normalized to 0-100 across all residues in that structure. For modules with fewer tokens than residues (e.g. `noise_conditioning` which outputs one vector per sample), the B-factor is uniform across all residues.

### `umap/` directory (unless `--skip-umap`)

Per-module UMAP visualizations:

- `umap_<module>_by_step.png`: Scatter plot colored by denoising step (viridis colormap). Shows how representations evolve during denoising.
- `umap_<module>_by_structure.png` (PDB mode only): Scatter plot colored by input PDB. Shows when/where the model distinguishes different structures.
- `umap_<module>.csv`: UMAP coordinates with step, sample, and (in PDB mode) structure labels for custom plotting.

## Code Structure

### `extract_representations.py` — module-by-module walkthrough

The script is organized into these sections:

**Imports and hook target registry** (lines 141-230): Defines `HOOK_TARGET_REGISTRY`, a dictionary mapping target names (e.g. `"transformer_output"`) to resolver functions. Each resolver takes the model and optional layer indices and returns a list of `(label, nn.Module)` pairs to hook. This registry-based design makes it easy to add new hook targets without modifying the rest of the code.

**Architecture summary** (lines 233-389): `print_architecture_summary()` prints an annotated module tree with `[HOOK]` markers at extraction points. Tries `torchinfo` first (for detailed shape/param info), falls back to `torchsummary`, and always prints a custom recursive walk. The summary is saved to `architecture_summary.txt` for reference.

**Pooling** (lines 392-411): `pool_representation()` reduces tensors to `[B, D]` by mean-pooling spatial dimensions. Handles `[B, L, D]` (transformer), `[B, C, L, A]` (conv), and `[B, D]` (already pooled) shapes.

**Sampler functions** (lines 414-658):
- `_compute_score()`: Derives the score function from the model's x0 prediction.
- `_euler_step()`: Standard Euler ODE update.
- `_stochastic_step()`: Karras stochastic update with noise injection.
- `_langevin_corrector_step()`: One Langevin MCMC step at the current noise level.
- `run_manual_sampling()`: The main denoising loop. Initializes from Gaussian noise, iterates through the noise schedule, applies the selected sampler's update rule at each step, and triggers hooks on every `model.forward()` call.

**PDB extraction** (lines 662-768): `run_pdb_extraction()` implements PDB mode. For each noise level in the schedule and each PDB, it loads the structure, adds calibrated noise, and runs one denoiser forward pass. The function handles variable-length proteins (each PDB is processed individually with batch size 1) and atom dimension mismatches between the PDB (37 atoms) and the model's expected number of atoms.

**CLI argument parsing** (lines 774-845): `parse_args()` defines all command-line arguments including the new `--pdb-dir` flag.

**Main function** (lines 907-end):
1. Loads the model and resolves hook targets.
2. Prints the architecture summary.
3. Registers forward hooks on selected modules. Each hook stores both the pooled `[B, D]` representation and (optionally) the full-resolution tensor.
4. Runs either PDB extraction or de novo sampling.
5. Removes hooks.
6. Saves PDB-specific outputs (per-structure tensors, B-factor PDBs).
7. Saves raw torch tensors to `embeddings/`.
8. Saves pooled numpy representations to `.npz` files (per-module at native dim + combined with zero-padding).
9. Computes per-module UMAP projections and plots.

### Key design decisions

**Forward hooks vs. model modification**: We use PyTorch's hook API rather than modifying model code. This means the extraction script works with any Protpardelle checkpoint without requiring code changes to the model itself.

**Manual denoising loop**: Rather than using `model.sample()` (which encapsulates the entire sampling process), we reimplement the denoising loop in `run_manual_sampling()`. This is necessary to support custom samplers (Langevin, predictor-corrector) while still triggering hooks on every forward pass.

**Per-module UMAPs**: Each module gets its own UMAP rather than one combined UMAP. This avoids the problem of zero-padded features (when modules have different dimensions) distorting the embedding, and makes it easier to compare dynamics within a single module across the denoising trajectory.

**Zero-padded combined file**: The combined `representations.npz` zero-pads all modules to the maximum dimension for convenience in cross-module analysis. The `module_native_dims` array records each module's original dimension so users can unpad.

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--config-path` | (required) | Path to model config YAML |
| `--checkpoint-path` | (required) | Path to model weights `.pth` |
| `--output-dir` | (required) | Directory for output files |
| `--pdb-dir` | `None` | Directory of `.pdb` files to encode (enables PDB mode) |
| `--sampling-config` | `None` | Override default sampling YAML (only for `--sampler=default`) |
| `--num-samples` | `8` | Number of proteins to sample (de novo mode only) |
| `--length` | `128` | Protein length in residues (de novo mode only) |
| `--num-steps` | `100` | Number of denoising steps / noise levels |
| `--step-scale` | `1.2` | Score scaling factor |
| `--seed` | `0` | Random seed |
| `--device` | auto | `cpu`, `cuda`, or `mps` |
| **Sampler** | | |
| `--sampler` | `euler` | `default`, `euler`, `stochastic`, `langevin`, `predictor_corrector` |
| `--s-churn` | `0.0` | Stochastic churn for `stochastic` sampler (gamma = s_churn / num_steps) |
| `--noise-scale` | `1.0` | Noise magnitude scaling for `stochastic` sampler |
| `--s-t-min` | `0.01` | Min sigma for stochastic noise injection (pre sigma_data scaling) |
| `--s-t-max` | `50.0` | Max sigma for stochastic noise injection (pre sigma_data scaling) |
| `--langevin-corrector-steps` | `1` | Corrector steps per noise level (`langevin`, `predictor_corrector`) |
| `--langevin-snr` | `0.16` | Signal-to-noise ratio for Langevin step size = snr * sigma^2 |
| **Hooks** | | |
| `--hook-targets` | `transformer_output` | Comma-separated module names (see Hook targets table) |
| `--layers` | `all` | Layer indices for `per_layer_attn` / `per_layer_ff`: `all` or comma-separated ints (e.g. `0,4,9`). All models have 10 layers (indices 0-9). |
| **Output** | | |
| `--save-tensors` | `False` | Save full-resolution torch `.pt` tensors (not just pooled numpy). Required for per-residue analysis and B-factor PDBs. |
| `--save-embeddings` | `False` | Alias for `--save-tensors` |
| `--skip-umap` | `False` | Skip UMAP computation |
| `--umap-neighbors` | `30` | UMAP n_neighbors |
| `--umap-min-dist` | `0.1` | UMAP min_dist |
| `--umap-metric` | `euclidean` | UMAP distance metric |

## Extending

**Add a new hook target:**
1. Write a resolver function `_resolve_my_target(model, layers) -> list[tuple[str, nn.Module]]`.
2. Add it to `HOOK_TARGET_REGISTRY` in `extract_representations.py`.

**Add a new sampler:**
1. Write a step function (see `_euler_step`, `_langevin_corrector_step` as examples).
2. Add a branch in `run_manual_sampling()` for the new sampler name.
3. Add the name to the `--sampler` choices in `parse_args()`.

**Change pooling strategy:**
Edit `pool_representation()`. For example, use max-pooling, CLS-token selection, or per-residue output (skip pooling and save full tensors with `--save-tensors`).

**Custom noise schedule or conditional sampling:**
Pass `--sampling-config` pointing to a sampling YAML with `--sampler=default`, or modify the noise schedule function in `run_manual_sampling()` for custom schedules.

## Relationship to Boltz masking_code

This toolkit is inspired by the `masking_code` directory in [Boltz](https://github.com/jwohlwend/boltz), which provides step-by-step diffusion control and representation intervention for Boltz-1. Key differences:

| | Boltz `masking_code` | Protpardelle `representation_extraction` |
|---|---|---|
| **Goal** | Intervene on representations (mask/edit `z`) then run diffusion | Extract and analyze representations across modules and steps |
| **Diffusion control** | Manual step loop via `DiffusionStepper` class | Manual loop with pluggable samplers (Euler, Langevin, PC) |
| **Architecture** | Pairformer trunk + structure diffusion module | U-ViT (conv + transformer) coordinate denoiser |
| **Representations** | Pair rep `z` [B,N,N,C] and single rep `s` [B,N,C] | Transformer hidden states, conv features, noise embeddings |
| **Input modes** | FASTA/YAML + MSA server | De novo generation or PDB encoding |
| **Samplers** | Single stepper (Euler + optional noise injection) | Euler, Karras stochastic, annealed Langevin, predictor-corrector |
| **Output** | mmCIF structures | numpy arrays, torch tensors, UMAP plots, B-factor PDBs |
| **Structure alignment** | N/A (generates structures) | Per-residue representations aligned to input PDB |
