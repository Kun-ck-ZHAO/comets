# CoMETs — Contrastive Multi-view Embedding for Neural Time Series

A two-stage framework for learning from simultaneously recorded spike and LFP data. Stage 1 aligns the two modalities contrastively to produce a shared population latent. Stage 2 uses that latent as context to learn per-neuron embeddings — a different decoding target, not behavioral variables but unit identity.

---

## Motivation

Spikes and LFP are two views of the same underlying population state. Spike data encodes which cells fired and when; LFP captures mesoscopic oscillatory structure (theta rhythm, phase precession in hippocampus) that the spiking representation alone cannot recover. Contrastive alignment forces the spike encoder to organize in a space that is also coherent with the LFP — producing a richer latent than either modality trained independently.

The framework is CEBRA-style temporal contrastive learning extended to the cross-modal case. The key addition is a cross-modal InfoNCE term that pulls spike and LFP embeddings at the same time into alignment. A non-linear projection head (MLP) absorbs the alignment cost so the encoder itself stays free to organize by behavioral structure.

---

## Framework

### Stage 1 — Multimodal Latent Alignment

**Inputs**: spikes `[T, N]`, LFP `[T, C]` (synchronized, same time axis)

**Output**: population latent `Z [T, D]`

**Loss** (per step, in-batch temporal negatives):

```
L_within = β · L_AA  +  (1−β) · L_BB        # same-modality InfoNCE
L_cross  = β · L_AB  +  (1−β) · L_BA        # cross-modality InfoNCE
L_total  = (1−α) · L_within  +  α · L_cross
```

`α_cross` controls how much cross-modal alignment is enforced. `β` controls the spike/LFP balance within each term. Positives are nearby time windows; negatives are in-batch and temporally masked.

**Projection head choice matters**: with a linear head, cross-modal alignment leaks into the encoder and degrades spatial structure in the spike embedding. With an MLP head, alignment is absorbed in the projection space and the encoder improves on both modalities simultaneously.

**Evaluation**: spatial decoding R² via 10-fold ridge regression on the concatenated `Z_spike ⊕ Z_lfp`.

---

### Stage 2 — Unit Identity Embedding

**Input**: frozen `Z [T, D]` from Stage 1, spike matrix `[T, N]`

**Output**: per-neuron embedding `E [N, d_u]` — one stationary vector per cell

The unit embedding is shaped by two loss terms, both using `.detach()` so no gradient flows into Stage 1:

**L_coact** — co-activity loss: neurons that fire together at nearby times are pulled together. Weight is a log-rate product (`log(r_i+1) · log(r_j+1)`) — naturally zero when either neuron is silent, which suppresses the co-silence problem in sparse spike data. A batch-centered relu gate further focuses the loss on pairs that fire above the current batch mean.

**L_ctx** — context-conditioned loss: neurons that fire at similar levels under similar latent population states (as measured by cosine similarity in `Z`) are pulled together across arbitrary time windows. Uses dual-modality context (spike latent + LFP latent) weighted by `λ_spike_ctx` and `λ_lfp_ctx`.

```
L_unit = λ_coact · L_coact  +  λ_ctx · L_ctx
```

**Evaluation**: Spearman ρ between pairwise cosine distances in `E` and pairwise cosine distances in place cell ratemap space.

---

## Results

Validated on hippocampal place cell recordings from four sessions (CRCNS ec013: sessions 156, 440, 441, 442), linear track, 50 Hz binning.

### Stage 1 — Spatial decoding R²

| Config | Projection | r²\_spike | r²\_lfp | r²\_concat |
|---|---|---|---|---|
| within-only (α=0, β=0.5) | Linear | 0.568 | −0.093 | 0.584 |
| spike-dom (α=0.1, β=0.95) | Linear | 0.527 | 0.121 | 0.545 |
| **spike-dom (α=0.1, β=0.95)** | **MLP p32** | **0.614** | **0.497** | **0.613** |

Within-only training wins narrowly with a linear head — no cross-modal interference, pure temporal proximity optimisation. But the LFP encoder learns nothing (r²\_lfp ≈ −0.09). The MLP head fixes this: cross-modal alignment boosts both modalities simultaneously.

### Stage 2 — Unit embedding quality (mean ρ, 4 sessions)

Best latent: `heavy_lfp` encoder with spike-dominant cross-modal loss.

| Unit loss (λ\_coact, λ\_ctx) | Mean ρ |
|---|---|
| (0.75, 0.25) co-activity dominant | 0.58 |
| (0.50, 0.50) balanced | 0.60 |
| **(0.25, 0.75) context dominant** | **0.619** |

Context-dominant loss wins universally across all latent configs and sessions. Where neurons fire together (shared latent context) matters more than whether they fire together (co-activity alone).

### Central finding

**The latent that best decodes position is not the latent that best characterises unit identity.**

Within-only training (no LFP alignment) ranks first for spatial R² but produces weaker unit embeddings. Cross-modal training with an MLP head produces a latent that is better for unit identity even though it is not the best spatial decoder. Stage 1 and Stage 2 quality should be evaluated independently — selecting latent hyperparameters on spatial R² alone will leave unit embedding quality on the table.

---

## Limitations

- Validated on a single behavioral paradigm (hippocampal spatial navigation). Generalization to other brain regions, non-spatial tasks, or non-hippocampal spike/LFP pairs is untested.
- Temporal contrastive learning from neural data is no longer a novel contribution in isolation — CEBRA and related work have established the paradigm. The cross-modal extension and unit embedding stage are the differentiating elements.
- Unit embedding evaluation uses ratemap correlation, which is tied to place cell identity. Extending to cell types (pyramidal vs interneuron), non-spatial selectivity, or multi-session shared embedding spaces is open.

---

## Repo Layout

```
comets/                         # Stage 1 reference implementation (original package)
comets_test1/
  comets/                       # Full package — Stage 1 + Stage 2
    configs.py                  #   ModalityConfig, SamplingConfig
                                #   LatentTrainingConfig, UnitTrainingConfig
    core.py                     #   CoMETsModel, train_latent()
    encoders.py                 #   Conv1D, SkippedConv, GRU, MLP encoder families
    heads.py                    #   IdentityHead, LinearHead, OneLayerMLP, TwoLayerMLP
    infonce.py                  #   InfoNCELoss (temporal masking, in-batch negatives)
    sampler.py                  #   LatentSampler, UnitSampler
    unit_encoder.py             #   UnitEncoder (embedding table + optional projector)
    unit_loss.py                #   coact_loss, ctx_loss, UnitLoss
    unit_train.py               #   train_unit()
    unit_eval.py                #   ratemap_correlation, si_gradient_correlation,
                                #   split_half_stability
  conf/                         # Hydra configs (latent_train.yaml, unit_train.yaml)
  latent_train.py               # CLI: sweep Stage 1
  unit_train.py                 # CLI: sweep Stage 2 on frozen latents
  eval.py                       # Evaluation script
  comets_tutorial.ipynb         # End-to-end walkthrough (Stage 1 + Stage 2)
  crack_latent.ipynb            # Latent hyperparameter analysis
  results.ipynb                 # Full grid results and figures
real_data/
  lfp_spike_recovery.ipynb      # Real data pipeline (per-session)
  lfp_spike_recovery_concat.ipynb  # Multi-session concatenated analysis
```

---

## Setup

```bash
pip install torch numpy scipy scikit-learn matplotlib hydra-core tqdm
```

The active package lives in `comets_test1/comets/`. Point your path there:

```python
import sys
sys.path.insert(0, "comets_test1")
import comets
```

---

## API

### Stage 1

```python
from comets import (
    ModalityConfig, SamplingConfig, LatentTrainingConfig,
    Conv1DEncoder, Conv1DEncoderConfig, TwoLayerMLP,
    train_latent,
)

latent_model = train_latent(
    spikes, ModalityConfig(Conv1DEncoder, spike_enc_cfg, win_len=10),
    lfp,    ModalityConfig(Conv1DEncoder, lfp_enc_cfg,   win_len=10),
    SamplingConfig(batch_size=512, pos_offset=1, negative_mask_dist=10),
    LatentTrainingConfig(alpha_cross=0.1, beta=0.95, head_cls=TwoLayerMLP, proj_dim=32),
)

Z = latent_model.embed_latent(spikes)   # [T, D]
```

### Stage 2

```python
from comets import UnitTrainingConfig, train_unit

unit_model = train_unit(
    latent_model, spikes, lfp,
    sampling_cfg,
    UnitTrainingConfig(lambda_coact=0.25, lambda_ctx=0.75, d_u=8),
)

E = unit_model.embeddings()   # [N, d_u] — per-neuron embedding
```

### Evaluation

```python
from comets import ratemap_correlation, si_gradient_correlation, split_half_stability

rho, p        = ratemap_correlation(E, ratemaps)        # Spearman ρ vs place field overlap
rho_si        = si_gradient_correlation(E, si_values)   # ρ vs spatial information gradient
stability     = split_half_stability(E1, E2)            # Procrustes-aligned cosine similarity
```
