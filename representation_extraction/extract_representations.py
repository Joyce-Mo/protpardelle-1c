#!/usr/bin/env python3
"""Extract representations from different Protpardelle modules across the denoising schedule.

This script hooks into various internal modules of Protpardelle during sampling and
captures intermediate hidden states at every denoising step. It supports multiple
denoising strategies (ODE, stochastic Karras, annealed Langevin dynamics, predictor-
corrector) via a manual sampling loop, and can save both pooled numpy arrays and
full-resolution torch tensors.

Hookable modules (selected via --hook-targets):
    transformer_output   : Output of the full TimeCondTransformer block.
    per_layer_attn       : Output of TimeCondAttention in selected transformer layers.
    per_layer_ff         : Output of TimeCondFeedForward in selected transformer layers.
    patch_embedding      : Output of to_patch_embedding (input to transformer).
    from_patch           : Output of from_patch (transformer -> conv decoder).
    down_conv            : Output(s) of downsampling convolutional blocks.
    up_conv              : Output(s) of upsampling convolutional blocks.
    noise_conditioning   : Output of the NoiseConditioningBlock (time embedding).
    coord_denoiser_input : Input tensor to the CoordinateDenoiser forward pass.
    coord_denoiser_output: Output tensor from the CoordinateDenoiser (denoised coords).

Sampler strategies (selected via --sampler):
    euler                : Deterministic ODE (Euler method). Default Protpardelle sampler.
    stochastic           : Karras stochastic sampler (adds noise proportional to gamma
                           at each step, controlled by --s-churn).
    langevin             : Annealed Langevin dynamics. Runs --langevin-corrector-steps
                           corrector updates per noise level using the score function,
                           with step size --langevin-snr.
    predictor_corrector  : Predictor-corrector (PC) sampler. Alternates one Euler
                           predictor step with --langevin-corrector-steps Langevin
                           corrector steps.

Outputs:
    representations.npz            : Pooled feature vectors with step/sample/module indices.
    representations_index.csv      : Human-readable index.
    embeddings/                    : (--save-tensors) Per-step raw torch tensors.
    embeddings/manifest.pt         : (--save-tensors) Dict mapping module -> list of step paths.
    representations_umap.csv/png   : (unless --skip-umap) UMAP visualizations.

Example:
    python representation_extraction/extract_representations.py \
        --config-path model_params/configs/cc89.yaml \
        --checkpoint-path model_params/weights/cc89_epoch415.pth \
      --output-dir results/march_rep_test \
      --pdb-dir examples/march_rep_test \
      --hook-targets transformer_output,patch_embedding,noise_conditioning \
      --save-tensors \
      --num-steps 50 --device mps


All models have 10 transformer layers (indices 0-9). So --layers 0,5,11 in your example
  would fail on index 11. Good choices:                                                 
  - --layers 0,4,9 — first, middle, last                                                   
  - --layers 0,2,4,6,9 — broader sweep         
  - --layers all — everything (more data, slower)                                          
                                                                                           
  Sampler (--sampler)                                                                      
                                                                                           
  ┌─────────────────────┬───────────────────────┬────────────┬────────────────────────┐    
  │       Sampler       │     What it does      │   Speed    │      When to use       │  
  ├─────────────────────┼───────────────────────┼────────────┼────────────────────────┤
  │                     │ Deterministic ODE.    │ Fast (1    │ Start here. Baseline   │
  │ euler               │ Exactly what the      │ fwd/step)  │ representations.       │
  │                     │ model learned.        │            │                        │
  ├─────────────────────┼───────────────────────┼────────────┼────────────────────────┤
  │ stochastic          │ Adds noise at each    │ Fast (1    │ Study effect of        │
  │                     │ step (Karras).        │ fwd/step)  │ stochasticity.         │
  ├─────────────────────┼───────────────────────┼────────────┼────────────────────────┤
  │                     │ MCMC corrector at     │ Slow (K+1  │ Study refined          │
  │ langevin            │ each noise level,     │ fwd/step)  │ representations at     │
  │                     │ then Euler predictor. │            │ each noise level.      │
  ├─────────────────────┼───────────────────────┼────────────┼────────────────────────┤
  │                     │ Euler first, then     │ Slow (K+1  │ More stable variant of │
  │ predictor_corrector │ Langevin at new noise │ fwd/step)  │  Langevin.             │
  │                     │  level.               │            │                        │
  └─────────────────────┴───────────────────────┴────────────┴────────────────────────┘

  For a first run, use euler. Langevin with 5 corrector steps means 6x more forward passes
  (600 instead of 100).

  Langevin parameters

  Only relevant for langevin and predictor_corrector:
  - --langevin-corrector-steps: 1-5. More = better refinement, slower. Start with 1.
  - --langevin-snr: Step size = snr * σ². Range 0.05-0.5. Too high → unstable. 0.16 is a
  safe default.

  Sampling parameters

  - --num-steps: Denoising discretization. 50-200. More = smoother trajectory, slower. 100
  is a good balance.
  - --step-scale: Score scaling. 1.0-1.6. Higher = lower diversity, sharper structures. 1.2
   is the default.
  - --num-samples: Batch size. How many proteins generated in parallel. 8 is reasonable for
   128-length on a GPU with 24GB. Reduce to 4 if you OOM.
  - --length: Protein length in residues. VRAM scales ~linearly. 128 is moderate. 64 for
  fast tests, 256 for larger.

  Device

  - cuda — GPU (required for practical speed)
  - mps — Apple Silicon GPU 
  - cpu — Very slow, only for debugging

  Recommended first run

  python representation_extraction/extract_representations.py \
      --config-path model_params/configs/cc58.yaml \
      --checkpoint-path model_params/weights/cc58_epoch416.pth \
      --output-dir results/representations \
      --hook-targets transformer_output \
      --sampler euler \
      --save-tensors \
      --num-samples 4 --length 64 --num-steps 50 --device mps

  This is fast and will confirm everything works. Then scale up:

  # Full multi-module extraction
  python representation_extraction/extract_representations.py \
      --config-path model_params/configs/cc58.yaml \
      --checkpoint-path model_params/weights/cc58_epoch416.pth \
      --output-dir results/repr_multi \
      --hook-targets transformer_output,patch_embedding,noise_conditioning,per_layer_attn \
      --layers 0,4,9 \
      --sampler euler \
      --save-tensors \
      --num-samples 8 --length 128 --num-steps 100 --device mps

  # Then compare with Langevin
  python representation_extraction/extract_representations.py \
      --config-path model_params/configs/cc58.yaml \
      --checkpoint-path model_params/weights/cc58_epoch416.pth \
      --output-dir results/repr_langevin \
      --hook-targets transformer_output \
      --sampler langevin --langevin-corrector-steps 3 --langevin-snr 0.16 \
      --save-tensors \
      --num-samples 8 --length 128 --num-steps 100 --device mps
"""

from __future__ import annotations

import argparse
import inspect
import io
import sys
from collections import defaultdict
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from protpardelle.core.models import load_model
from protpardelle.data.atom import atom37_mask_from_aatype
from protpardelle.data.pdb_io import feats_to_pdb_str, load_feats_from_pdb
from protpardelle.utils import seed_everything, unsqueeze_trailing_dims


# ---------------------------------------------------------------------------
# Hookable target registry
# ---------------------------------------------------------------------------

def _resolve_transformer_output(model, _layers):
    return [("transformer_output", model.struct_model.net.transformer)]


def _resolve_per_layer_attn(model, layers):
    transformer_layers = model.struct_model.net.transformer.layers
    idxs = layers if layers else list(range(len(transformer_layers)))
    return [(f"attn_layer{i}", transformer_layers[i][0]) for i in idxs]


def _resolve_per_layer_ff(model, layers):
    transformer_layers = model.struct_model.net.transformer.layers
    idxs = layers if layers else list(range(len(transformer_layers)))
    return [(f"ff_layer{i}", transformer_layers[i][1]) for i in idxs]


def _resolve_patch_embedding(model, _layers):
    return [("patch_embedding", model.struct_model.net.to_patch_embedding)]


def _resolve_from_patch(model, _layers):
    return [("from_patch", model.struct_model.net.from_patch)]


def _resolve_down_conv(model, _layers):
    pairs = []
    for i, layer in enumerate(model.struct_model.net.down_conv):
        for j, block in enumerate(layer):
            pairs.append((f"down_conv_{i}_{j}", block))
    return pairs


def _resolve_up_conv(model, _layers):
    pairs = []
    for i, layer in enumerate(model.struct_model.net.up_conv):
        for j, block in enumerate(layer):
            pairs.append((f"up_conv_{i}_{j}", block))
    return pairs


def _resolve_noise_conditioning(model, _layers):
    return [("noise_conditioning", model.struct_model.noise_block)]


def _resolve_coord_denoiser_input(model, _layers):
    return [("coord_denoiser_input", model.struct_model)]


def _resolve_coord_denoiser_output(model, _layers):
    return [("coord_denoiser_output", model.struct_model)]


HOOK_TARGET_REGISTRY: dict[str, callable] = {
    "transformer_output": _resolve_transformer_output,
    "per_layer_attn": _resolve_per_layer_attn,
    "per_layer_ff": _resolve_per_layer_ff,
    "patch_embedding": _resolve_patch_embedding,
    "from_patch": _resolve_from_patch,
    "down_conv": _resolve_down_conv,
    "up_conv": _resolve_up_conv,
    "noise_conditioning": _resolve_noise_conditioning,
    "coord_denoiser_input": _resolve_coord_denoiser_input,
    "coord_denoiser_output": _resolve_coord_denoiser_output,
}


# ---------------------------------------------------------------------------
# Architecture summary
# ---------------------------------------------------------------------------

def _format_params(n: int) -> str:
    """Format a parameter count as a human-readable string."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def print_architecture_summary(
    model: torch.nn.Module,
    hooked_modules: set[int],
    hook_label_map: dict[int, str],
    output_path: Path | None = None,
    seq_len: int = 128,
) -> str:
    """Print model architecture tree annotated with hook locations.

    Uses torchsummary/torchinfo if available for detailed shape/param info,
    then always prints a module tree with [HOOK: <target>] markers showing
    exactly where representations are extracted from.

    Args:
        model: The Protpardelle model.
        hooked_modules: Set of id(module) for hooked modules.
        hook_label_map: Mapping id(module) -> hook target label.
        output_path: If provided, write summary to this file.
        seq_len: Sequence length for dummy input (used by torchsummary).

    Returns:
        The summary string.
    """
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("PROTPARDELLE ARCHITECTURE SUMMARY")
    lines.append("Modules marked with [HOOK] are captured during extraction.")
    lines.append("=" * 80)

    # --- Try torchsummary / torchinfo for detailed summary ---
    detailed_summary = None

    # Try torchinfo first (modern, handles complex models)
    try:
        from torchinfo import summary as torchinfo_summary

        # Build dummy inputs for the CoordinateDenoiser (struct_model)
        device = next(model.parameters()).device
        num_atoms = model.num_atoms
        dummy_coords = torch.randn(1, seq_len, num_atoms, 3, device=device)
        dummy_noise = torch.ones(1, seq_len, device=device)
        dummy_mask = torch.ones(1, seq_len, device=device)
        dummy_residx = torch.arange(seq_len, device=device).unsqueeze(0)

        # torchinfo returns a ModelStatistics object whose __str__ is the summary
        info = torchinfo_summary(
            model.struct_model,
            input_data={
                "noisy_coords": dummy_coords,
                "noise_level": dummy_noise,
                "seq_mask": dummy_mask,
                "residue_index": dummy_residx,
            },
            col_names=["input_size", "output_size", "num_params"],
            depth=5,
            verbose=0,
            device=device,
        )
        detailed_summary = str(info)
    except Exception:
        pass

    # Fall back to torchsummary (older API)
    if detailed_summary is None:
        try:
            from torchsummary import summary as torch_summary

            # torchsummary prints directly to stdout; capture it
            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()
            try:
                torch_summary(model.struct_model.net, input_size=(seq_len, model.num_atoms, 3))
            except Exception:
                pass
            finally:
                sys.stdout = old_stdout
            captured = buf.getvalue()
            if captured.strip():
                detailed_summary = captured
        except ImportError:
            pass

    if detailed_summary:
        lines.append("")
        lines.append("--- Detailed parameter summary (struct_model / CoordinateDenoiser) ---")
        lines.append(detailed_summary)

    # --- Custom annotated module tree (always printed) ---
    lines.append("")
    lines.append("--- Module tree with hook annotations ---")
    lines.append("")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"Total parameters:     {_format_params(total_params)}")
    lines.append(f"Trainable parameters: {_format_params(trainable_params)}")
    lines.append("")

    def _walk(module: torch.nn.Module, prefix: str, depth: int):
        mod_id = id(module)
        is_hooked = mod_id in hooked_modules
        hook_tag = f"  [HOOK: {hook_label_map[mod_id]}]" if is_hooked else ""

        # Module class name and parameter count
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        param_str = f" ({_format_params(n_params)} params)" if n_params > 0 else ""

        # Get first parameter shape as a hint
        shape_hint = ""
        own_params = list(module.parameters(recurse=False))
        if own_params:
            p0 = own_params[0]
            shape_hint = f" weight={list(p0.shape)}"

        indent = "  " * depth
        connector = "├─ " if depth > 0 else ""
        line = f"{indent}{connector}{prefix}{module.__class__.__name__}{param_str}{shape_hint}{hook_tag}"
        if is_hooked:
            line = f"\033[1;32m{line}\033[0m"  # green bold for terminal
        lines.append(line)

        children = list(module.named_children())
        for i, (child_name, child_mod) in enumerate(children):
            child_prefix = f"{child_name}: " if child_name else ""
            _walk(child_mod, child_prefix, depth + 1)

    _walk(model, "Protpardelle.", 0)

    lines.append("")
    lines.append("=" * 80)

    summary_text = "\n".join(lines)

    # Print to console (strip ANSI for file output)
    print(summary_text)

    if output_path is not None:
        # Strip ANSI escape codes for the file version
        import re
        clean_text = re.sub(r"\033\[[0-9;]*m", "", summary_text)
        output_path.write_text(clean_text, encoding="utf-8")
        print(f"Architecture summary -> {output_path}")

    return summary_text


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def pool_representation(x: torch.Tensor) -> torch.Tensor:
    """Reduce a tensor to [B, D] by mean-pooling spatial dimensions.

    Handles common shapes:
        [B, L, D]      -> mean over L
        [B, C, L, A]   -> mean over L and A (conv feature maps)
        [B, D]          -> identity
        other           -> flatten to [B, -1]
    """
    if x.ndim == 3:
        return x.mean(dim=1)
    if x.ndim == 4:
        return x.mean(dim=(2, 3))
    if x.ndim == 2:
        return x
    return x.flatten(start_dim=1)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def _compute_score(xt: torch.Tensor, x0: torch.Tensor, sigma: torch.Tensor,
                   seq_mask: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """Compute the score: (xt - x0) / sigma, masked by seq_mask."""
    score = (xt - x0) / unsqueeze_trailing_dims(sigma.clamp(min=tol), xt)
    return score * unsqueeze_trailing_dims(seq_mask, score)


def _euler_step(xt: torch.Tensor, x0: torch.Tensor,
                sigma: torch.Tensor, sigma_next: torch.Tensor,
                seq_mask: torch.Tensor, step_scale: float) -> torch.Tensor:
    """Standard Euler ODE step: xt_next = xt + step_scale * (sigma_next - sigma) * score."""
    score = _compute_score(xt, x0, sigma, seq_mask)
    step = score * step_scale * unsqueeze_trailing_dims(sigma_next - sigma, score)
    return xt + step


def _stochastic_step(xt: torch.Tensor, x0: torch.Tensor,
                     sigma: torch.Tensor, sigma_next: torch.Tensor,
                     seq_mask: torch.Tensor, step_scale: float,
                     gamma: float, noise_scale: float) -> torch.Tensor:
    """Karras stochastic sampler: inject noise then Euler step from inflated sigma."""
    sigma_hat = sigma + gamma * sigma
    sigma_delta = torch.sqrt(sigma_hat ** 2 - sigma ** 2)
    xt_hat = xt + unsqueeze_trailing_dims(sigma_delta, xt) * noise_scale * torch.randn_like(xt)
    xt_hat = xt_hat * unsqueeze_trailing_dims(seq_mask, xt_hat)
    # Euler step from sigma_hat -> sigma_next
    score = _compute_score(xt_hat, x0, sigma_hat, seq_mask)
    step = score * step_scale * unsqueeze_trailing_dims(sigma_next - sigma_hat, score)
    return xt_hat + step


def _langevin_corrector_step(xt: torch.Tensor, x0: torch.Tensor,
                             sigma: torch.Tensor, seq_mask: torch.Tensor,
                             snr: float) -> torch.Tensor:
    """One Langevin MCMC corrector step at current noise level.

    Following Song et al. (2021) "Score-Based Generative Modeling through SDEs":
        step_size = snr * sigma^2
        xt_corrected = xt + step_size * score + sqrt(2 * step_size) * z
    """
    score = _compute_score(xt, x0, sigma, seq_mask)
    # score points from xt toward data; gradient of log p_sigma is -score in Protpardelle's
    # convention (score = (xt - x0)/sigma), so the Langevin update *subtracts* score.
    step_size = snr * unsqueeze_trailing_dims(sigma ** 2, xt)
    noise = torch.randn_like(xt)
    xt_corrected = xt - step_size * score + torch.sqrt(2 * step_size) * noise
    return xt_corrected * unsqueeze_trailing_dims(seq_mask, xt_corrected)


def run_manual_sampling(
    model,
    seq_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_index: torch.Tensor,
    num_steps: int,
    step_scale: float,
    sampler: Literal["euler", "stochastic", "langevin", "predictor_corrector"],
    s_churn: float = 0.0,
    noise_scale: float = 1.0,
    s_t_min: float = 0.01,
    s_t_max: float = 50.0,
    langevin_corrector_steps: int = 1,
    langevin_snr: float = 0.16,
    dummy_fill_mode: str = "zero",
) -> dict[str, Any]:
    """Manual denoising loop with pluggable sampler strategy.

    This replaces model.sample() to give full control over the update rule
    while still triggering all registered hooks on each model.forward() call.

    Returns the same dict structure as model.sample() for compatibility.
    """
    device = model.device
    sigma_data = model.sigma_data
    noise_schedule_fn = model.sampling_noise_schedule_default

    # Scale thresholds by sigma_data (matching model.sample)
    s_t_min = s_t_min * sigma_data
    s_t_max = s_t_max * sigma_data

    batch_size = seq_mask.shape[0]
    num_atoms = model.num_atoms
    coords_shape = (batch_size, seq_mask.shape[1], num_atoms, 3)

    # Noise schedule
    timesteps = torch.linspace(1, 0, num_steps + 1)
    sigma_init = noise_schedule_fn(timesteps[0])

    # Initialize from Gaussian noise
    xt = torch.randn(*coords_shape, device=device) * sigma_init
    xt = xt * unsqueeze_trailing_dims(seq_mask, xt)

    # Backbone atom mask for masking
    bb_seq = (seq_mask * 7).long()  # GLY
    bb_atom_mask = atom37_mask_from_aatype(bb_seq, seq_mask)

    x0 = None
    x_self_cond = None
    s_logprobs = None
    s_self_cond = None
    s_hat = (seq_mask * 7).long()

    xt_traj, x0_traj = [], []

    sigma_float = sigma_init

    pbar = tqdm(total=num_steps, desc=f"Sampling ({sampler})")

    for i, t_next in enumerate(timesteps[1:]):
        sigma_next_float = noise_schedule_fn(t_next)
        if i == num_steps - 1:
            sigma_next_float = sigma_next_float * 0

        sigma = torch.full((batch_size,), sigma_float, device=device)
        sigma_next = torch.full((batch_size,), sigma_next_float, device=device)

        gamma = s_churn / num_steps if (sigma_float >= s_t_min and sigma_float <= s_t_max) else 0.0

        # Apply backbone mask
        if model.config.model.task == "backbone":
            xt = xt * bb_atom_mask.unsqueeze(-1)

        # --- Forward pass: get x0 prediction ---
        if sigma_float > 0:
            # For stochastic sampler, inflate sigma before forward pass
            if sampler == "stochastic" and gamma > 0:
                sigma_hat = sigma + gamma * sigma
                sigma_delta = torch.sqrt(sigma_hat ** 2 - sigma ** 2)
                xt_input = xt + unsqueeze_trailing_dims(sigma_delta, xt) * noise_scale * torch.randn_like(xt)
                xt_input = xt_input * unsqueeze_trailing_dims(seq_mask, xt_input)
                fwd_sigma = sigma_hat
            else:
                xt_input = xt
                fwd_sigma = sigma

            x0, s_logprobs, x_self_cond, s_self_cond = model.forward(
                noisy_coords=xt_input,
                noise_level=fwd_sigma,
                seq_mask=seq_mask,
                residue_index=residue_index,
                chain_index=chain_index,
                struct_self_cond=(
                    x_self_cond if model.config.train.self_cond_train_prob > 0.5 else None
                ),
                run_mpnn_model=False,
            )

            # --- Apply sampler update rule ---
            if sampler == "euler":
                xt = _euler_step(xt, x0, sigma, sigma_next, seq_mask, step_scale)

            elif sampler == "stochastic":
                if gamma > 0:
                    # We already computed x0 from xt_input at sigma_hat
                    score = _compute_score(xt_input, x0, fwd_sigma, seq_mask)
                    step = score * step_scale * unsqueeze_trailing_dims(sigma_next - fwd_sigma, score)
                    xt = xt_input + step
                else:
                    xt = _euler_step(xt, x0, sigma, sigma_next, seq_mask, step_scale)

            elif sampler == "langevin":
                # Corrector: Langevin steps at current noise level
                for _c in range(langevin_corrector_steps):
                    # Re-evaluate x0 for corrector (except first which reuses above)
                    if _c > 0:
                        x0, s_logprobs, x_self_cond, s_self_cond = model.forward(
                            noisy_coords=xt,
                            noise_level=sigma,
                            seq_mask=seq_mask,
                            residue_index=residue_index,
                            chain_index=chain_index,
                            struct_self_cond=(
                                x_self_cond if model.config.train.self_cond_train_prob > 0.5 else None
                            ),
                            run_mpnn_model=False,
                        )
                    xt = _langevin_corrector_step(xt, x0, sigma, seq_mask, langevin_snr)

                # Predictor: Euler step to next noise level
                # Re-evaluate x0 after corrector steps
                if langevin_corrector_steps > 0:
                    x0, s_logprobs, x_self_cond, s_self_cond = model.forward(
                        noisy_coords=xt,
                        noise_level=sigma,
                        seq_mask=seq_mask,
                        residue_index=residue_index,
                        chain_index=chain_index,
                        struct_self_cond=(
                            x_self_cond if model.config.train.self_cond_train_prob > 0.5 else None
                        ),
                        run_mpnn_model=False,
                    )
                xt = _euler_step(xt, x0, sigma, sigma_next, seq_mask, step_scale)

            elif sampler == "predictor_corrector":
                # Predictor: Euler step
                xt = _euler_step(xt, x0, sigma, sigma_next, seq_mask, step_scale)

                # Corrector: Langevin steps at sigma_next
                if sigma_next_float > 0:
                    for _c in range(langevin_corrector_steps):
                        x0_c, _, x_self_cond, _ = model.forward(
                            noisy_coords=xt,
                            noise_level=sigma_next,
                            seq_mask=seq_mask,
                            residue_index=residue_index,
                            chain_index=chain_index,
                            struct_self_cond=(
                                x_self_cond if model.config.train.self_cond_train_prob > 0.5 else None
                            ),
                            run_mpnn_model=False,
                        )
                        xt = _langevin_corrector_step(xt, x0_c, sigma_next, seq_mask, langevin_snr)
        else:
            xt = x0

        sigma_float = sigma_next_float

        # Logging (same as model.sample)
        xt_scale = sigma_data / unsqueeze_trailing_dims(
            torch.sqrt(sigma_next ** 2 + sigma_data ** 2), xt
        )
        xt_traj.append((xt * xt_scale).cpu())
        x0_traj.append(x0.cpu())

        pbar.update(1)
    pbar.close()

    atom_mask = atom37_mask_from_aatype(s_hat, seq_mask)
    return {
        "x": xt,
        "s": s_hat,
        "seq_mask": seq_mask,
        "atom_mask": atom_mask,
        "xt_traj": xt_traj,
        "x0_traj": x0_traj,
        "st_traj": [s_hat.cpu()] * len(xt_traj),
        "s0_traj": [s_logprobs.cpu() if s_logprobs is not None else torch.zeros(1)] * len(xt_traj),
        "residue_index": residue_index,
        "chain_index": chain_index,
    }


# ---------------------------------------------------------------------------
# PDB-conditioned extraction (encode real structures)
# ---------------------------------------------------------------------------

def run_pdb_extraction(
    model,
    pdb_paths: list[Path],
    num_steps: int,
    step_scale: float,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run the denoiser on real PDB structures at each noise level.

    For each PDB, noise the coordinates to each sigma level in the schedule,
    run one forward pass, and let hooks capture per-residue representations.
    This produces representations that are aligned to the input residues.

    Returns:
        Dict with xt_traj, x0_traj, pdb_names, feats_list, etc.
    """
    if device is None:
        device = model.device
    sigma_data = model.sigma_data
    noise_schedule_fn = model.sampling_noise_schedule_default
    num_atoms = model.num_atoms

    # Load all PDBs and collect features
    all_feats = []
    pdb_names = []
    for pdb_path in pdb_paths:
        feats, _hetero = load_feats_from_pdb(pdb_path, include_pos_feats=True)
        all_feats.append(feats)
        pdb_names.append(pdb_path.stem)
        print(f"  Loaded {pdb_path.name}: {feats['aatype'].shape[0]} residues")

    # Noise schedule: linearly spaced timesteps from 1 (max noise) to 0 (clean)
    timesteps = torch.linspace(1, 0, num_steps + 1)

    xt_traj, x0_traj = [], []

    pbar = tqdm(total=num_steps * len(all_feats), desc="PDB extraction")

    for step_i in range(num_steps):
        t = timesteps[step_i]
        t_next = timesteps[step_i + 1]
        sigma_float = float(noise_schedule_fn(t))
        sigma_next_float = float(noise_schedule_fn(t_next))
        if step_i == num_steps - 1:
            sigma_next_float = 0.0

        step_xt_list, step_x0_list = [], []

        for feats in all_feats:
            length = feats["aatype"].shape[0]
            seq_mask = torch.ones(1, length, device=device)
            residue_index = feats["residue_index"].unsqueeze(0).long().to(device)
            chain_index = feats["chain_index"].unsqueeze(0).long().to(device)

            # Get clean coordinates [1, L, 37, 3]
            clean_coords = feats["atom_positions"].unsqueeze(0).to(device)
            # Truncate or pad atom dim to match model
            if clean_coords.shape[2] > num_atoms:
                clean_coords = clean_coords[:, :, :num_atoms, :]
            elif clean_coords.shape[2] < num_atoms:
                pad = torch.zeros(
                    1, length, num_atoms - clean_coords.shape[2], 3, device=device
                )
                clean_coords = torch.cat([clean_coords, pad], dim=2)

            # Add noise: xt = clean + sigma * noise
            sigma = torch.full((1,), sigma_float, device=device)
            noise = torch.randn_like(clean_coords) * sigma_float
            xt = clean_coords + noise
            xt = xt * seq_mask.unsqueeze(-1).unsqueeze(-1)

            # Forward pass through denoiser (triggers hooks)
            if sigma_float > 0:
                x0_pred, _s_logprobs, _x_self_cond, _s_self_cond = model.forward(
                    noisy_coords=xt,
                    noise_level=sigma,
                    seq_mask=seq_mask,
                    residue_index=residue_index,
                    chain_index=chain_index,
                    run_mpnn_model=False,
                )
            else:
                x0_pred = clean_coords

            # Scale for trajectory logging
            xt_scale = sigma_data / torch.sqrt(sigma ** 2 + sigma_data ** 2)
            step_xt_list.append((xt * xt_scale).cpu())
            step_x0_list.append(x0_pred.cpu())

            pbar.update(1)

        xt_traj.append(step_xt_list)
        x0_traj.append(step_x0_list)

    pbar.close()

    return {
        "xt_traj": xt_traj,  # list[list[Tensor]] — [step][pdb]
        "x0_traj": x0_traj,
        "pdb_names": pdb_names,
        "feats_list": all_feats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-step representations from multiple Protpardelle modules."
    )
    p.add_argument("--config-path", type=Path, required=True)
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sampling-config", type=Path, default=None,
                    help="Override default sampling YAML (only used with --sampler=default).")
    p.add_argument("--pdb-dir", type=Path, default=None,
                    help="Directory of PDB files to encode. When set, representations are "
                         "extracted by noising each structure at each sigma level and running "
                         "the denoiser, producing per-residue features aligned to the input. "
                         "Overrides --num-samples and --length.")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--step-scale", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])

    # Sampler selection
    p.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["default", "euler", "stochastic", "langevin", "predictor_corrector"],
        help=(
            "Denoising strategy. 'default' uses model.sample() directly. "
            "Other choices use a manual loop with the specified update rule."
        ),
    )
    p.add_argument("--s-churn", type=float, default=0.0,
                    help="Stochastic churn for 'stochastic' sampler; gamma = s_churn / num_steps.")
    p.add_argument("--noise-scale", type=float, default=1.0,
                    help="Scale for injected noise in stochastic sampler.")
    p.add_argument("--s-t-min", type=float, default=0.01,
                    help="Don't apply s_churn below this noise level (pre sigma_data scaling).")
    p.add_argument("--s-t-max", type=float, default=50.0,
                    help="Don't apply s_churn above this noise level (pre sigma_data scaling).")
    p.add_argument("--langevin-corrector-steps", type=int, default=1,
                    help="Number of Langevin corrector steps per noise level (for 'langevin' and 'predictor_corrector').")
    p.add_argument("--langevin-snr", type=float, default=0.16,
                    help="Signal-to-noise ratio for Langevin step size: step_size = snr * sigma^2.")

    # Hook configuration
    p.add_argument(
        "--hook-targets",
        type=str,
        default="transformer_output",
        help=(
            "Comma-separated list of modules to hook. "
            f"Choices: {', '.join(HOOK_TARGET_REGISTRY.keys())}"
        ),
    )
    p.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layer indices for per_layer_attn / per_layer_ff: 'all' or comma-separated ints.",
    )

    # Output options
    p.add_argument("--save-tensors", action="store_true",
                    help="Save full-resolution torch tensors (not just pooled numpy).")
    p.add_argument("--save-embeddings", action="store_true",
                    help="Alias for --save-tensors.")
    p.add_argument("--skip-umap", action="store_true")
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-metric", type=str, default="euclidean")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_default_sampling_kwargs(cfg_override: Path | None) -> dict[str, Any]:
    if cfg_override is not None:
        cfg_path = cfg_override
    else:
        cfg_path = (
            resources.files("protpardelle")
            / "configs"
            / "running"
            / "sampling_unconditional.yaml"
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["sampling"]


def _filter_sampling_kwargs(model, sampling_kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(model.sample)
    valid = set(sig.parameters.keys())
    filtered = {k: v for k, v in sampling_kwargs.items() if k in valid}
    if "noise_schedule" in filtered and not callable(filtered["noise_schedule"]):
        filtered.pop("noise_schedule")
    return filtered


def _parse_layers(layers_arg: str, depth: int) -> list[int] | None:
    if layers_arg.strip().lower() == "all":
        return None
    idxs = []
    for tok in layers_arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        i = int(tok)
        if i < 0 or i >= depth:
            raise ValueError(f"Layer {i} out of range [0, {depth - 1}]")
        idxs.append(i)
    return sorted(set(idxs)) if idxs else None


def _align_calls_to_steps(
    call_reps: list[torch.Tensor], num_steps: int
) -> tuple[list[torch.Tensor], np.ndarray]:
    """Given N hook calls, select/downsample to match num_steps denoising steps."""
    n = len(call_reps)
    if n == num_steps:
        return call_reps, np.arange(num_steps)
    groups = np.array_split(np.arange(n), num_steps)
    idx = np.array([g[-1] for g in groups], dtype=np.int64)
    return [call_reps[j] for j in idx], idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    save_tensors = args.save_tensors or args.save_embeddings

    device = torch.device(args.device) if args.device is not None else None
    print(f"Loading model from {args.config_path} / {args.checkpoint_path} ...")
    model = load_model(args.config_path, args.checkpoint_path, device=device)

    # Resolve layer selection
    transformer_depth = len(model.struct_model.net.transformer.layers)
    layer_idxs = _parse_layers(args.layers, transformer_depth)

    # Parse hook targets
    target_names = [t.strip() for t in args.hook_targets.split(",") if t.strip()]
    for t in target_names:
        if t not in HOOK_TARGET_REGISTRY:
            raise ValueError(
                f"Unknown hook target '{t}'. Choose from: {list(HOOK_TARGET_REGISTRY.keys())}"
            )

    # Resolve all (label, module) pairs to hook
    hook_specs: list[tuple[str, torch.nn.Module]] = []
    for t in target_names:
        hook_specs.extend(HOOK_TARGET_REGISTRY[t](model, layer_idxs))

    print(f"Hooking {len(hook_specs)} module(s): {[s[0] for s in hook_specs]}")

    # Print architecture summary with hook annotations
    hooked_module_ids = {id(mod) for _, mod in hook_specs}
    hook_label_map = {id(mod): label for label, mod in hook_specs}
    print_architecture_summary(
        model,
        hooked_modules=hooked_module_ids,
        hook_label_map=hook_label_map,
        output_path=args.output_dir / "architecture_summary.txt",
        seq_len=args.length,
    )

    # Storage for pooled representations: label -> list of [B, D] tensors
    hook_records_pooled: dict[str, list[torch.Tensor]] = defaultdict(list)
    # Storage for raw tensors (only if --save-tensors): label -> list of tensors
    hook_records_raw: dict[str, list[torch.Tensor]] = defaultdict(list) if save_tensors else {}
    hook_handles = []

    # Register hooks
    for label, module in hook_specs:
        is_input_hook = label == "coord_denoiser_input"

        if is_input_hook:
            def _make_pre_hook(lbl):
                def _hook(_module, args, _kwargs=None):
                    inp = args[0] if isinstance(args, tuple) else args
                    inp_cpu = inp.detach().cpu()
                    hook_records_pooled[lbl].append(pool_representation(inp_cpu))
                    if save_tensors:
                        hook_records_raw[lbl].append(inp_cpu)
                return _hook
            hook_handles.append(module.register_forward_pre_hook(_make_pre_hook(label)))
        else:
            def _make_hook(lbl):
                def _hook(_module, _inputs, output):
                    out = output[0] if isinstance(output, tuple) else output
                    out_cpu = out.detach().cpu()
                    hook_records_pooled[lbl].append(pool_representation(out_cpu))
                    if save_tensors:
                        hook_records_raw[lbl].append(out_cpu)
                return _hook
            hook_handles.append(module.register_forward_hook(_make_hook(label)))

    # -----------------------------------------------------------------------
    # PDB mode vs. de novo sampling mode
    # -----------------------------------------------------------------------
    pdb_mode = args.pdb_dir is not None
    pdb_paths: list[Path] = []
    if pdb_mode:
        pdb_paths = sorted(args.pdb_dir.glob("*.pdb"))
        if not pdb_paths:
            raise FileNotFoundError(f"No .pdb files found in {args.pdb_dir}")
        print(f"PDB mode: encoding {len(pdb_paths)} structures from {args.pdb_dir}")

    try:
        with torch.no_grad():
            if pdb_mode:
                aux = run_pdb_extraction(
                    model=model,
                    pdb_paths=pdb_paths,
                    num_steps=args.num_steps,
                    step_scale=args.step_scale,
                    device=torch.device(args.device) if args.device else model.device,
                )
            else:
                # Prepare sampling inputs
                seq_mask, residue_index, chain_index = model.make_seq_mask_for_sampling(
                    length_ranges_per_chain=[(args.length, args.length)],
                    num_samples=args.num_samples,
                )

                sampler_desc = args.sampler
                if args.sampler == "stochastic":
                    sampler_desc += f" (s_churn={args.s_churn})"
                elif args.sampler in ("langevin", "predictor_corrector"):
                    sampler_desc += f" (corrector_steps={args.langevin_corrector_steps}, snr={args.langevin_snr})"

                print(f"Sampling: {args.num_samples} samples, {args.num_steps} steps, "
                      f"length {args.length}, sampler={sampler_desc}")

                if args.sampler == "default":
                    sampling_kwargs = _load_default_sampling_kwargs(args.sampling_config)
                    sampling_kwargs["num_steps"] = args.num_steps
                    sampling_kwargs["step_scale"] = args.step_scale
                    sampling_kwargs["s_churn"] = args.s_churn
                    if "conditional_cfg" in sampling_kwargs:
                        sampling_kwargs["conditional_cfg"]["num_recurrence_steps"] = 1
                    sampling_kwargs = _filter_sampling_kwargs(model, sampling_kwargs)

                    aux = model.sample(
                        seq_mask=seq_mask,
                        residue_index=residue_index,
                        chain_index=chain_index,
                        dummy_fill_mode=model.config.data.dummy_fill_mode,
                        **sampling_kwargs,
                    )
                else:
                    aux = run_manual_sampling(
                        model=model,
                        seq_mask=seq_mask,
                        residue_index=residue_index,
                        chain_index=chain_index,
                        num_steps=args.num_steps,
                        step_scale=args.step_scale,
                        sampler=args.sampler,
                        s_churn=args.s_churn,
                        noise_scale=args.noise_scale,
                        s_t_min=args.s_t_min,
                        s_t_max=args.s_t_max,
                        langevin_corrector_steps=args.langevin_corrector_steps,
                        langevin_snr=args.langevin_snr,
                        dummy_fill_mode=model.config.data.dummy_fill_mode,
                    )
    finally:
        for h in hook_handles:
            h.remove()

    if not hook_records_pooled:
        raise RuntimeError("No representations captured. Check hook targets.")

    num_steps_effective = len(aux["xt_traj"])
    print(f"Captured representations from {len(hook_records_pooled)} module(s) "
          f"across {num_steps_effective} effective steps.")

    # -----------------------------------------------------------------------
    # PDB mode: save per-residue representations aligned to structures
    # -----------------------------------------------------------------------
    if pdb_mode:
        pdb_out_dir = args.output_dir / "per_structure"
        pdb_out_dir.mkdir(parents=True, exist_ok=True)
        pdb_names = aux["pdb_names"]
        feats_list = aux["feats_list"]

        # In PDB mode, hooks fire once per (step, pdb) pair.
        # Total calls = num_steps * num_pdbs. Group by steps, then by pdbs.
        num_pdbs = len(pdb_names)

        for mod_name in sorted(hook_records_raw.keys() if save_tensors else hook_records_pooled.keys()):
            if save_tensors:
                raw_list = hook_records_raw[mod_name]
            else:
                raw_list = hook_records_pooled[mod_name]

            # raw_list has num_steps * num_pdbs entries (step-major order)
            for pdb_i, pdb_name in enumerate(pdb_names):
                struct_dir = pdb_out_dir / pdb_name / mod_name
                struct_dir.mkdir(parents=True, exist_ok=True)

                for step_j in range(num_steps_effective):
                    call_idx = step_j * num_pdbs + pdb_i
                    if call_idx < len(raw_list):
                        torch.save(raw_list[call_idx], struct_dir / f"step_{step_j:04d}.pt")

        # Save PDB metadata (residue names, indices, chain info)
        for pdb_i, (pdb_name, feats) in enumerate(zip(pdb_names, feats_list)):
            meta_path = pdb_out_dir / pdb_name / "metadata.pt"
            torch.save({
                "pdb_name": pdb_name,
                "aatype": feats["aatype"],
                "residue_index": feats["residue_index"],
                "chain_index": feats["chain_index"],
                "atom_positions": feats["atom_positions"],
                "num_residues": feats["aatype"].shape[0],
            }, meta_path)

        # Write per-residue B-factor PDBs: color each residue by representation norm
        # at selected denoising steps (useful for visualization in PyMOL/ChimeraX)
        if save_tensors:
            bfactor_dir = args.output_dir / "bfactor_pdbs"
            bfactor_dir.mkdir(parents=True, exist_ok=True)
            # Pick a few representative steps
            vis_steps = [0, num_steps_effective // 4, num_steps_effective // 2,
                         3 * num_steps_effective // 4, num_steps_effective - 1]
            vis_steps = sorted(set(s for s in vis_steps if 0 <= s < num_steps_effective))

            for mod_name in sorted(hook_records_raw.keys()):
                raw_list = hook_records_raw[mod_name]
                for pdb_i, (pdb_name, feats) in enumerate(zip(pdb_names, feats_list)):
                    for step_j in vis_steps:
                        call_idx = step_j * num_pdbs + pdb_i
                        if call_idx >= len(raw_list):
                            continue
                        rep = raw_list[call_idx]  # [1, L, D] or [1, C, L, A] etc.
                        length = feats["aatype"].shape[0]
                        num_atoms_pdb = feats["atom_positions"].shape[1]

                        # Compute per-residue norm as B-factor
                        if rep.ndim == 3 and rep.shape[1] >= length:
                            per_res = rep[0, :length].norm(dim=-1)  # [L]
                        elif rep.ndim == 4:
                            per_res = rep[0, :, :length].norm(dim=(0, 2))  # [L]
                        elif rep.ndim == 3 and rep.shape[1] < length:
                            # Module with fewer tokens (e.g. noise_conditioning [1,1,D])
                            # Broadcast a single scalar to all residues
                            val = rep[0].norm(dim=-1).mean()
                            per_res = val.expand(length)
                        else:
                            continue
                        # Normalize to 0-100 for B-factor column
                        if per_res.max() > 0:
                            per_res = per_res / per_res.max() * 100.0

                        bfactors = per_res.unsqueeze(-1).expand(-1, num_atoms_pdb)

                        pdb_str = feats_to_pdb_str(
                            atom_coords=feats["atom_positions"][:length],
                            aatype=feats["aatype"][:length].long(),
                            residue_index=feats["residue_index"][:length].long(),
                            chain_index=feats["chain_index"][:length].long(),
                            b_factors=bfactors,
                        )
                        out_name = f"{pdb_name}_{mod_name}_step{step_j:04d}.pdb"
                        (bfactor_dir / out_name).write_text(pdb_str)

            print(f"B-factor PDBs -> {bfactor_dir}/ (color by representation norm in PyMOL/ChimeraX)")

        print(f"Per-structure representations -> {pdb_out_dir}/")
        print(f"  Structure: per_structure/<pdb_name>/<module>/step_NNNN.pt")
        print(f"  Metadata:  per_structure/<pdb_name>/metadata.pt")

    # -----------------------------------------------------------------------
    # Save raw torch tensors / embeddings
    # -----------------------------------------------------------------------
    if save_tensors:
        emb_dir = args.output_dir / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, list[str]] = {}

        for mod_name in sorted(hook_records_raw.keys()):
            raw_list = hook_records_raw[mod_name]
            aligned, _ = _align_calls_to_steps(raw_list, num_steps_effective)

            mod_dir = emb_dir / mod_name
            mod_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for step_i, tensor in enumerate(aligned):
                fname = f"step_{step_i:04d}.pt"
                torch.save(tensor, mod_dir / fname)
                paths.append(str(mod_dir / fname))
            manifest[mod_name] = paths

        # Save x0 and xt trajectories as torch tensors
        traj_dir = emb_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        torch.save(aux["xt_traj"], traj_dir / "xt_traj.pt")
        torch.save(aux["x0_traj"], traj_dir / "x0_traj.pt")
        manifest["_xt_traj"] = [str(traj_dir / "xt_traj.pt")]
        manifest["_x0_traj"] = [str(traj_dir / "x0_traj.pt")]

        # Save sampling metadata alongside tensors
        meta = {
            "sampler": args.sampler,
            "num_steps": args.num_steps,
            "step_scale": args.step_scale,
            "s_churn": args.s_churn,
            "noise_scale": args.noise_scale,
            "langevin_corrector_steps": args.langevin_corrector_steps,
            "langevin_snr": args.langevin_snr,
            "num_samples": args.num_samples,
            "length": args.length,
            "seed": args.seed,
            "config_path": str(args.config_path),
            "checkpoint_path": str(args.checkpoint_path),
        }
        manifest["_meta"] = meta
        torch.save(manifest, emb_dir / "manifest.pt")
        print(f"Saved raw tensors -> {emb_dir}/ ({sum(len(v) for k, v in manifest.items() if not k.startswith('_'))} files)")

    # -----------------------------------------------------------------------
    # Save pooled numpy representations
    # -----------------------------------------------------------------------
    # Different modules may produce different embedding dimensions
    # (e.g. noise_conditioning=1024D, transformer_output=256D).
    # We save per-module npz files at native dims, and a combined file
    # zero-padded to max dim for UMAP / cross-module analysis.

    per_module_data: dict[str, dict[str, np.ndarray]] = {}
    all_features, all_step, all_sample, all_module, all_dim = [], [], [], [], []
    module_names_ordered = sorted(hook_records_pooled.keys())

    if pdb_mode:
        num_pdbs = len(aux["pdb_names"])
    else:
        num_pdbs = 0  # unused in de novo mode

    for mod_name in module_names_ordered:
        reps = hook_records_pooled[mod_name]

        if pdb_mode:
            # In PDB mode: hooks fire num_steps * num_pdbs times (step-major).
            # Each call is [1, D]. Reshape to [num_steps, num_pdbs, D].
            n_total = len(reps)
            expected = num_steps_effective * num_pdbs
            if n_total != expected:
                # Align to expected count
                reps, _ = _align_calls_to_steps(reps, expected)
            stacked = np.stack([r.numpy() for r in reps], axis=0)  # [n_total, 1, D]
            stacked = stacked.squeeze(1)  # [n_total, D]
            dim = stacked.shape[-1]
            flat = stacked.astype(np.float32)
            # Step and sample indices: step-major order
            steps = np.repeat(np.arange(num_steps_effective), num_pdbs).astype(np.int32)
            samples = np.tile(np.arange(num_pdbs), num_steps_effective).astype(np.int32)
        else:
            aligned, _ = _align_calls_to_steps(reps, num_steps_effective)
            stacked = np.stack([r.numpy() for r in aligned], axis=0)
            n_steps, batch_size, dim = stacked.shape
            flat = stacked.reshape(n_steps * batch_size, dim).astype(np.float32)
            steps = np.repeat(np.arange(n_steps), batch_size).astype(np.int32)
            samples = np.tile(np.arange(batch_size), n_steps).astype(np.int32)

        # Save per-module file at native dimension
        mod_npz = args.output_dir / f"representations_{mod_name}.npz"
        save_dict = {"features": flat, "step_idx": steps, "sample_idx": samples}
        if pdb_mode:
            save_dict["pdb_names"] = np.array(aux["pdb_names"], dtype=object)
        np.savez_compressed(mod_npz, **save_dict)
        print(f"  {mod_name}: {flat.shape[0]} vectors x {dim}D -> {mod_npz}")

        per_module_data[mod_name] = {"features": flat, "step_idx": steps, "sample_idx": samples}
        all_features.append(flat)
        all_step.append(steps)
        all_sample.append(samples)
        all_module.append(np.full(flat.shape[0], mod_name, dtype=object))
        all_dim.append(dim)

    # Zero-pad all features to max dimension for the combined file
    max_dim = max(all_dim)
    padded_features = []
    for feat, dim in zip(all_features, all_dim):
        if dim < max_dim:
            pad_width = ((0, 0), (0, max_dim - dim))
            padded_features.append(np.pad(feat, pad_width, mode="constant", constant_values=0.0))
        else:
            padded_features.append(feat)

    features = np.concatenate(padded_features, axis=0).astype(np.float32)
    step_idx = np.concatenate(all_step, axis=0)
    sample_idx = np.concatenate(all_sample, axis=0)
    module_labels = np.concatenate(all_module, axis=0)

    unique_modules = sorted(set(module_labels))
    mod_to_int = {m: i for i, m in enumerate(unique_modules)}
    module_idx = np.array([mod_to_int[m] for m in module_labels], dtype=np.int32)

    # Record native dimension per module so users can unpad
    native_dims = np.array([all_dim[module_names_ordered.index(m)] for m in unique_modules], dtype=np.int32)

    npz_path = args.output_dir / "representations.npz"
    npz_dict = {
        "features": features,
        "step_idx": step_idx,
        "sample_idx": sample_idx,
        "module_idx": module_idx,
        "module_names": np.array(unique_modules, dtype=object),
        "module_native_dims": native_dims,
        "num_steps": np.array([num_steps_effective], dtype=np.int32),
        "num_samples": np.array([num_pdbs if pdb_mode else args.num_samples], dtype=np.int32),
        "sampler": np.array([args.sampler], dtype=object),
    }
    if pdb_mode:
        npz_dict["pdb_names"] = np.array(aux["pdb_names"], dtype=object)
    np.savez_compressed(npz_path, **npz_dict)

    csv_path = args.output_dir / "representations_index.csv"
    pd.DataFrame({
        "module": module_labels,
        "step_idx": step_idx,
        "sample_idx": sample_idx,
    }).to_csv(csv_path, index=False)

    dim_summary = ", ".join(f"{m}={d}D" for m, d in zip(unique_modules, native_dims))
    print(f"Combined: {features.shape[0]} vectors, padded to {max_dim}D (native: {dim_summary})")
    print(f"  -> {npz_path}")
    print(f"  -> {csv_path}")

    # -----------------------------------------------------------------------
    # Per-module UMAP (one UMAP per hooked module, colored by step)
    # -----------------------------------------------------------------------
    if args.skip_umap:
        return

    try:
        import umap
    except ImportError:
        print("umap-learn not installed; skipping UMAP. Install with: pip install umap-learn")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping UMAP plots.")
        return

    umap_dir = args.output_dir / "umap"
    umap_dir.mkdir(parents=True, exist_ok=True)

    for mod_name in module_names_ordered:
        mod_data = per_module_data[mod_name]
        feat = mod_data["features"]
        steps = mod_data["step_idx"]
        samples = mod_data["sample_idx"]

        if feat.shape[0] < 15:
            print(f"  Skipping UMAP for {mod_name}: too few points ({feat.shape[0]})")
            continue

        n_neighbors = min(args.umap_neighbors, feat.shape[0] - 1)
        print(f"  Computing UMAP for {mod_name} ({feat.shape[0]} points, {feat.shape[1]}D) ...")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.seed,
        )
        emb = reducer.fit_transform(feat)

        # Save CSV
        mod_umap_df = pd.DataFrame({
            "step_idx": steps,
            "sample_idx": samples,
            "umap1": emb[:, 0],
            "umap2": emb[:, 1],
        })
        if pdb_mode:
            # In PDB mode, sample_idx maps to PDB index
            pdb_names = aux["pdb_names"]
            mod_umap_df["pdb_name"] = [pdb_names[s] for s in samples]
        mod_umap_df.to_csv(umap_dir / f"umap_{mod_name}.csv", index=False)

        # Plot colored by denoising step
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=steps, s=8, alpha=0.7, cmap="viridis")
        plt.colorbar(sc, ax=ax, label="Denoising step")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title(f"{mod_name} — {args.sampler} (colored by step)")
        fig.tight_layout()
        fig.savefig(umap_dir / f"umap_{mod_name}_by_step.png", dpi=200)
        plt.close(fig)

        # In PDB mode, also color by structure
        if pdb_mode:
            pdb_labels = mod_umap_df["pdb_name"].values
            fig, ax = plt.subplots(figsize=(8, 6))
            for pdb_name in sorted(set(pdb_labels)):
                mask = pdb_labels == pdb_name
                ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.7, label=pdb_name)
            ax.legend(fontsize=7, markerscale=3, loc="best")
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.set_title(f"{mod_name} — {args.sampler} (colored by structure)")
            fig.tight_layout()
            fig.savefig(umap_dir / f"umap_{mod_name}_by_structure.png", dpi=200)
            plt.close(fig)

    print(f"Per-module UMAP outputs -> {umap_dir}/")


if __name__ == "__main__":
    main()
