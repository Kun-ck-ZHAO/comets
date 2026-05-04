# Architecture

This document describes the internal architecture of the CoMET/CoMETs code in `comets/` and how notebooks use it.

## High-Level Design

CoMETs learns aligned embeddings from two synchronized time-series modalities using contrastive objectives.

Inputs:
- Modality A: `A` with shape `[T, C_A]`
- Modality B: `B` with shape `[T, C_B]`

Output:
- Two embedding trajectories `Z_A`, `Z_B` in a shared projection space

Core components:
1. Config layer (`configs.py`)
2. Encoders (`encoders.py`)
3. Projection heads (`heads.py`)
4. Sampling/windowing (`sampler.py`)
5. Contrastive loss (`infonce.py`)
6. Training loop and model wrapper (`core.py`)
7. Visualization helper (`viz.py`)

## Module Responsibilities

### `comets/configs.py`
Dataclasses used to wire experiments declaratively.

- `ModalityConfig`
  - `encoder_cls`: encoder class for a modality
  - `encoder_cfg`: encoder configuration object
  - `win_len`: temporal window length used by sampler/gathering

- `SamplingConfig`
  - `batch_size`: number of sampled anchors per step
  - `pos_offset`: positive index offset (`t -> t + offset`)
  - `negative_mask_dist`: temporal radius for excluding near-time negatives

- `TrainingConfig`
  - Optimization: `epochs`, `lr`, `weight_decay`
  - Contrastive: `temperature`
  - Projection: `proj_dim`, `head_cls`
  - Loss mixing: `beta`, `alpha_cross`

### `comets/encoders.py`
Families of modality encoders mapping windowed inputs `[B, W, C]` to normalized vectors `[B, D]`.

Implemented families:
- `Conv1DEncoder`
- `SkippedConvEncoder` (residual 1D conv blocks)
- `CEBRASKipEncoder` (CEBRA-like skip CNN variant)
- `Conv2DEncoder`
- `MLPEncoder`
- `GRUEncoder`

Each has a config dataclass carrying dimensions/hyperparameters.

### `comets/heads.py`
Projection heads applied after modality encoders.

- `IdentityHead`
- `LinearHead`
- `OneLayerMLP`
- `TwoLayerMLP`
- `make_head(in_dim, out_dim, head_cls)` for head construction

Heads output normalized projection vectors for contrastive learning.

### `comets/sampler.py`
Temporal sampling and window extraction.

- `InBatchSampler.sample(T)` returns
  - `ref_idx`: sampled anchor indices
  - `pos_idx = ref_idx + pos_offset`

- `_gather_windows(X, idx, win_len)`
  - converts point indices into clamped temporal windows
  - output shape `[B, win_len, C]` (or `[B,1,C]` when `win_len <= 1`)

Negatives are implicitly in-batch and filtered by temporal masking inside the loss.

### `comets/infonce.py`
`InfoNCELoss` implements contrastive objective with:
- batch-wise normalization
- temporal masking for invalid near-time negatives
- row-wise numerical stabilization
- negatives-only denominator formulation

Arguments include both vectors and their source indices so masking can be index-aware.

### `comets/core.py`
Contains two primary interfaces:

- `train(...) -> CoMETs`
  - builds encoders/projectors for both modalities
  - optimizer: `AdamW`
  - scheduler: cosine annealing over training steps
  - gradient clipping applied each step
  - computes four losses per step:
    - within-modality: `AA`, `BB`
    - cross-modality: `AB`, `BA`
  - combines with `beta` and `alpha_cross`

- `CoMETs` object
  - stores trained encoders/projectors
  - `embedA`, `embedB`, `embed_all` for batched embedding extraction

### `comets/viz.py`
`plotemb2` utility for side-by-side 3D scatter plots of two trajectories.

## Training Data Flow

Per training step:
1. Sample `ref_idx`, `pos_idx`.
2. Gather windows for both modalities at both index sets.
3. Encode + project windows into vectors.
4. Form in-batch negatives from current mini-batch anchors.
5. Compute `AA`, `BB`, `AB`, `BA` InfoNCE losses.
6. Mix losses:
   - `within = beta*AA + (1-beta)*BB`
   - `cross  = beta*AB + (1-beta)*BA`
   - `total  = alpha_cross*cross + (1-alpha_cross)*within`
7. Backprop + grad clip + optimizer step + LR scheduler step.

## Notebook Interaction Pattern

Most notebooks follow this pattern:
1. Build/normalize modality matrices `A`, `B`.
2. Instantiate modality configs with chosen encoder family.
3. Instantiate sampling + training configs.
4. Call `comets.train(...)`.
5. Extract `ZA`, `ZB` via `model.embedA/B`.
6. Run downstream analysis (visualization, decoding RMSE, shuffle controls, grid sweeps).

## Practical Boundaries

- Designed for synchronized dual-modality contrastive alignment.
- Windowing/sampling assumes temporal indexing on first dimension (`T`).
- No built-in checkpoint manager/CLI in core package; orchestration currently lives in notebooks.

## Directory Context Around Core

- `data_checkpoint/`: cached input/output artifacts used by notebooks
- `sfn_fig/`: generated figure assets
- `archive/`: historical implementations and earlier experiments

## Known Maintenance Notes

- Top-level metadata name is `comet`, while code imports use `comets`.
- `comets/__init__.py` should be reviewed for export list consistency.
- Test coverage is currently absent from repo root; adding unit tests for windowing/loss masking would reduce regression risk.
