# CoMETs Extension Plan: Latent Encoder + Unit Encoder

## Overview

Extend CoMETs into a two-stage self-supervised framework:

- **Stage 1 — Latent encoder**: learns a population-level latent `Z [T, D]` from synchronized LFP + spike data using temporal contrastive learning (InfoNCE over time windows).
- **Stage 2 — Unit encoder**: freezes the latent encoder, then learns a geometry over neuron identity by defining conditional similarity between neurons using their firing rates and the frozen latent as context.

The two stages are trained sequentially. This is a **clean rewrite**, not a patch on the original CoMETs. The original `comets/` is kept as reference; `comets_test1/comets/` is the new package.

---

## Rewrite principle

The original CoMETs code has implicit assumptions that need to be made explicit now that we have two encoders and two training stages. Specifically:

- **Neuron ordering is load-bearing.** The spike input matrix is `[T, N]` where column `j` always corresponds to neuron `j`. The latent encoder stacks all neurons in this fixed order to produce the population representation. This ordering must be consistent across training, inference, and Stage 2 sampling. The new code will make this explicit: the data contract is `spikes [T, N]` with a fixed neuron index set `{0, ..., N-1}`, and `unit_ids [N]` maps those indices to real neuron identifiers.
- **Two named encoders.** Code uses `latent_encoder` and `unit_encoder` consistently, never ambiguous "encoder" alone.
- **Two named training functions.** `train_latent()` and `train_unit()` are separate, clearly scoped.
- **Two named samplers.** `LatentSampler` for temporal sampling (Stage 1 and Stage 2 time groups), `UnitSampler` for neuron sampling within time groups (Stage 2 only).

---

## Motivation

CoMETs Stage 1 collapses all neurons into one population-level latent per time window. It does not preserve any notion of which neurons are more similar to each other. Stage 2 asks: after we have a working latent, can we also learn a space where neurons that tend to be active under similar conditions are geometrically closer? The conditional framing is key — two neurons are not similar just because they both fire. They are similar only when they fire at comparable levels under similar latent population states.

---

## Stage 1: Latent Encoder

### Input contract

```
spikes:   [T, N]   — binned spike counts, column j = neuron j (fixed order)
lfp:      [T, C]   — binned LFP channels
unit_ids: [N]      — real neuron identifiers for the N columns (for bookkeeping)
```

The latent encoder sees `spikes [T, N]` stacked as a flat population vector. Neuron ordering must match between training and any downstream use.

### What trains

- Spike encoder + projection head (modality A)
- LFP encoder + projection head (modality B)
- Loss: temporal InfoNCE — anchor and positive are nearby time windows, in-batch negatives

### Renamed API (relative to original CoMETs)

| Old | New |
|---|---|
| `train()` | `train_latent()` |
| `CoMETs` class | `CoMETsModel` |
| `embedA`, `embedB` | `embed_latent_spikes`, `embed_latent_lfp` |
| `embed_all` | `embed_latent` |
| `TrainingConfig` | `LatentTrainingConfig` |
| `InBatchSampler` | `LatentSampler` |

### Files (modified from original)

- `configs.py` — rename `TrainingConfig` → `LatentTrainingConfig`; keep `ModalityConfig`, `SamplingConfig`
- `core.py` — rename as above; `train_latent()` returns `CoMETsModel`
- `sampler.py` — rename `InBatchSampler` → `LatentSampler`; keep `_gather_windows`
- `encoders.py`, `heads.py`, `infonce.py`, `viz.py` — no renames needed, generic building blocks

---

## Stage 2: Unit Encoder

### Concept

Each neuron `i` has a learnable embedding `e_i ∈ R^{d_u}`. This embedding is **stationary** — it does not vary with time. The loss shapes the geometry of the embedding space by defining soft positive pairs between neurons based on their co-activity and shared latent context.

The embedding `e_i` is what gets compared in the loss and what gets extracted for evaluation. Firing rates `r_i,t` and frozen latent `z_t` appear only as gates in the loss weighting — they do not enter the representation.

```
representation:   e_i           [d_u]    — what we compare and evaluate
loss gates:       r_i,t         scalar   — firing rate of neuron i at time t
                  z_t           [D]      — frozen latent encoder output at time t
```

---

## Sampling design

Stage 2 reuses the temporal time groups from `LatentSampler`, then adds a second sampling step over neurons.

### Step 1 — Temporal sampling (LatentSampler, unchanged)

For each training step, draw:
```
t_anchor:       1 anchor time index
t_pos:          1 positive time index  (= t_anchor + pos_offset)
t_neg[K]:       K negative time indices (in-batch, temporally distant)
```

This is identical to Stage 1. K is the batch size — negatives are the rest of the batch.

### Step 2 — Unit sampling (UnitSampler, new)

Within the time groups above, sample M unit indices from `{0,...,N-1}`:

```
unit_ids_A:   [M]    — units sampled for anchor pool
unit_ids_B:   [M]    — units sampled for positive/negative pool
              (can be same set or independently drawn; see note below)
```

Then extract firing rates for each sampled unit at each time group:
```
r_A:    [M]       — spike counts for unit_ids_A at t_anchor
r_B:    [M]       — spike counts for unit_ids_B at t_pos (for L_coact)
r_N:    [M, K]    — spike counts for unit_ids_B at each t_neg (for L_ctx)
```

And look up their embeddings:
```
e_A:    [M, d_u]  — embeddings for unit_ids_A
e_B:    [M, d_u]  — embeddings for unit_ids_B
```

**Sampling strategy:** sample `unit_ids_A` and `unit_ids_B` independently each step. This gives M×M cross-pairs per step, which is efficient. M should be small (e.g. 32–64) to keep the pairwise weight matrix manageable at [M, M].

### UnitSampler interface

```python
class UnitSampler:
    def __init__(self, n_units: int, m_units: int, device):
        ...

    def sample(self) -> tuple[Tensor, Tensor]:
        # returns unit_ids_A [M], unit_ids_B [M]
        # both drawn uniformly without replacement from {0,...,N-1}
```

---

## Loss formulation

```
L_unit = lambda_coact * L_coact  +  lambda_ctx * L_ctx
```

All weights are computed with `.detach()` — no gradients flow through `z_t` or `r_i,t` into the latent encoder or the spike data.

### L_coact — co-activity loss

Uses `t_anchor` and `t_pos` (nearby windows, should share latent state).

For neuron `i` from `unit_ids_A` and neuron `j` from `unit_ids_B`:

```
w_act(i, j) = sigmoid( -(|r_A[i] - r_B[j]|) / tau_act ).detach()
```

This is high when both neurons fire at similar levels. To weight co-firing more than co-silence, multiply by a bias term based on mean activity:

```
activity_bias(i, j) = sigmoid( (r_A[i] + r_B[j]) / tau_bias ).detach()
w_act(i, j)         = w_act(i, j) * activity_bias(i, j)
```

Loss structure (soft InfoNCE over the M×M weight matrix):
- Anchor: `e_A[i]`  for each i in {0,...,M-1}
- Soft positives: `e_B[j]` for each j, weighted by `w_act[i, j]`
- Negatives: `e_B[j]` from units sampled at `t_neg` times (in-batch temporal negatives)

### L_ctx — context-conditioned similarity loss

Uses `t_anchor` and `t_neg` (temporally distant windows). Two neurons can still be pulled together across distant times if their firing is comparable and the latent context is similar.

```
w_fire(i, j) = sigmoid( -(|r_A[i] - r_N[j, k]|) / tau_fire ).detach()
               for each negative time k

w_ctx(t_anchor, t_neg[k]) = cosine( z_t_anchor, z_t_neg[k] ).clamp(min=0).detach()
                             (only positive similarity counts)

pair_weight(i, j, k) = w_fire(i, j, k) * w_ctx(t_anchor, t_neg[k])
```

Loss: anchor `e_A[i]`, soft positives `e_B[j]` at negative times weighted by `pair_weight`. This is a weighted InfoNCE over the [M, M, K] weight tensor summed over k.

---

## Files

### Modified (from original CoMETs)

| File | Changes |
|---|---|
| `configs.py` | Rename `TrainingConfig` → `LatentTrainingConfig`; add `UnitTrainingConfig` |
| `core.py` | Rename `train()` → `train_latent()`, `CoMETs` → `CoMETsModel`, embed methods; no logic changes |
| `sampler.py` | Rename `InBatchSampler` → `LatentSampler`; add `UnitSampler` class here |

### New files

| File | Contents |
|---|---|
| `unit_encoder.py` | `UnitEncoderConfig`, `UnitEncoder(nn.Module)` — embedding table + optional projector |
| `unit_loss.py` | `w_act`, `w_fire`, `w_ctx` weight functions; `CoactLoss`, `CtxLoss`, `UnitLoss` |
| `unit_train.py` | `train_unit(latent_model, spikes, sampling_cfg, unit_cfg) -> UnitEncoder` |
| `unit_eval.py` | `ratemap_correlation`, `place_cell_enrichment`, `split_half_stability`, `nn_purity` |

### Unchanged

`encoders.py`, `heads.py`, `infonce.py`, `viz.py` — no changes needed.

---

## Configs

```python
@dataclass
class LatentTrainingConfig:     # renamed from TrainingConfig
    epochs: int = 200
    lr: float = 3e-4
    temperature: float = 1.0
    weight_decay: float = 1e-5
    proj_dim: int = 16
    beta: float = 0.5
    alpha_cross: float = 0.5
    head_cls: nn.Module = None

@dataclass
class UnitTrainingConfig:
    epochs: int = 100
    lr: float = 1e-3
    d_u: int = 8                # unit embedding dimension
    m_units: int = 32           # units sampled per step per pool (M)
    tau_act: float = 1.0        # temperature for w_act gate
    tau_bias: float = 1.0       # temperature for co-firing bias
    tau_fire: float = 1.0       # temperature for w_fire gate
    temperature: float = 0.1    # InfoNCE temperature
    lambda_coact: float = 1.0
    lambda_ctx: float = 1.0
```

---

## Notebook workflow

```python
import comets

# Stage 1: train latent encoder
latent_model = comets.train_latent(
    spikes, mod_spikes,
    lfp,    mod_lfp,
    sampling_cfg, LatentTrainingConfig(...)
)

# Precompute frozen latent trajectory (also used for ridge decoding)
Z = latent_model.embed_latent(spikes)   # [T, D]

# Stage 2: train unit encoder
unit_model = comets.train_unit(
    latent_model, spikes, sampling_cfg, UnitTrainingConfig(...)
)

# Extract unit embeddings for evaluation
E = unit_model.embeddings()   # [N, d_u]
```

---

## Evaluation plan

### Primary
1. **Rate-map correlation**: Spearman between pairwise embedding distance in `E` and pairwise rate-map similarity. Continuous metric — avoids binary place-cell label circularity.
2. **Place-cell enrichment**: permutation test — are place cells more clustered in `E` than chance?

### Secondary
3. **Split-half stability**: train unit encoder on two non-overlapping temporal halves; measure Procrustes alignment of resulting `E` matrices.
4. **NN purity**: if cell-type labels available (pyramidal vs interneuron), check k-NN label purity in `E`.

### Baselines
- Random `nn.Embedding` (untrained)
- PCA of mean firing rate profiles per neuron

---

## Scope — explicitly out

- Time-varying / drifting unit embeddings — future work
- Identity loss — removed (stationary embeddings make it trivially satisfied)
- Transformer architecture
- Multi-session shared embedding space
