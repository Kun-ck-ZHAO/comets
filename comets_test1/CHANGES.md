# CHANGES — CoMETs Extension (comets_test1)

Tracks every modification made while implementing PLAN.md.
Sections: **Done as planned**, **Deviation / extra**, **Confusion / open question**.

---

## Round 2 changes (post-discussion revisions)

### `comets/configs.py`
- Removed `tau_act`, `tau_bias`, `tau_fire` — sigmoid-based firing gates replaced by log-product.
- Added `lambda_spike_ctx`, `lambda_lfp_ctx` — relative weights for spike vs LFP context similarity in L_ctx.

### `comets/unit_loss.py` — full rewrite

**Firing rate gate redesign (co-silence fix):**
- Old: `sigmoid(-|r_i - r_j| / tau)` — rate *similarity*, symmetric around 0, dominated by co-silence.
- New: `log(r_i+1) * log(r_j+1)` — co-firing *product*, zero when either neuron silent, scales with joint intensity. No tau, no sigmoid.

**L_coact redesign:**
- Old: M×M pool (pool A at t_anchor vs pool B at t_pos).
- New: 2M pool — same M units at both t_anchor and t_pos concatenated. Captures same-window co-firing (strongest signal), which the old formulation missed.
- **Batch-centered relu gate** added on top of log-rates: `delta = relu(log(r+1) - batch_mean)`. Only pairs where both units fire above the batch mean contribute weight. Handles exploding units automatically — a unit firing at 10× the mean raises the threshold, zeroing out weak co-firing pairs.
- Mask: all same-unit pairs excluded from both numerator and denominator (main diagonal + cross-time same-unit pairs = block structure). Computed via `uid_pool[:, None] == uid_pool[None, :]`.

**L_ctx redesign:**
- Old: z_anchor (single time point) compared against K negative windows. Still anchored to t_anchor.
- New: two independent K-banks (`ctx_A`, `ctx_B`) drawn from the full time range — no reference to t_anchor/t_pos. Fully decoupled from L_coact.
- Context similarity: element-wise cosine `cosine(Z[ctx_A[k]], Z[ctx_B[k]])` per k, not all-pairs K². Keeps cost O(K) not O(K²).
- **Dual-modality context:** supports both spike latent and LFP latent context similarities, weighted by `lambda_spike_ctx` and `lambda_lfp_ctx`. LFP is optional — if `lfp=None` is passed to `train_unit`, LFP term is skipped.
- Removed `CoactLoss`, `CtxLoss` as `nn.Module` wrappers; replaced with plain functions `coact_loss`, `ctx_loss`. `UnitLoss` module kept.

**`_soft_infonce` update:**
- Added `exclude_mask` parameter — pairs excluded from both numerator and denominator. Used for same-unit masking in L_coact. Previously only weights zeroed the numerator; now denominator is also cleaned.

### `comets/unit_train.py`
- Added optional `lfp` argument to `train_unit`. If provided, `Z_lfp` is precomputed and passed to `ctx_loss`.
- L_coact: single `torch.randperm` draw for M units; concatenated into 2M pool.
- L_ctx: two independent `torch.randint` draws for `ctx_A` and `ctx_B` banks; independent unit pools via `UnitSampler.sample()`.
- Removed all `tau_act`, `tau_bias`, `tau_fire` references.

### `comets/__init__.py`
- Removed: `compute_w_act`, `compute_w_fire`, `compute_w_ctx`, `CoactLoss`, `CtxLoss`.
- Added: `coact_loss`, `ctx_loss`.

---

## Design decisions logged (Round 2)

**Co-silence dominance:** `sigmoid(-|r_i - r_j| / tau)` gives high weight to (0,0) pairs which are the most common in sparse spike data. Replaced by log-product which is naturally zero when either neuron is silent.

**2M pool for L_coact:** Same-window co-firing (neuron i vs neuron j both at t_anchor) is the strongest evidence of functional similarity and was completely absent in the original M×M cross-window formulation.

**Batch-centered relu:** Addresses exploding units without requiring global statistics. The batch mean of log-rates is a local reference that adapts to the current population activity level.

**L_ctx two-bank structure:** Using z_anchor as the context reference ties L_ctx to the same time point as L_coact, undermining the "independent signal" goal. Two freely-drawn K-banks make L_ctx genuinely complementary.

**Dual-modality context (spike + LFP):** Both modalities encode population state from different angles. LFP captures mesoscopic dynamics that may not be fully represented in spiking. Weighted average of cosine similarities allows continuous interpolation.

**Soft InfoNCE literature grounding:** Base InfoNCE from van den Oord et al. 2018. The soft-weighted extension is a natural generalization; closest published forms are SupCon (Khosla et al. 2020) for hard multiple positives and soft nearest-neighbor loss (Frosst et al. 2019). The exact formulation here was derived from first principles — not a direct citation. Flag before writing into a methods section.

---

## Modified files

### `comets/configs.py`
- Renamed `TrainingConfig` → `LatentTrainingConfig` (same fields, same defaults).
- Added `UnitTrainingConfig` with all fields from PLAN.md.
- **Extra:** kept `TrainingConfig = LatentTrainingConfig` as a one-liner alias so existing notebooks that import `TrainingConfig` don't break.
- **Extra:** added `weight_decay: float = 1e-5` to `UnitTrainingConfig`. PLAN.md omitted it, but AdamW needs it and Stage 1 config has it. Flagged as intentional addition.

### `comets/core.py`
- Renamed class `CoMETs` → `CoMETsModel`.
- Renamed `train()` → `train_latent()`.
- Renamed `embedA` → `embed_latent_spikes`, `embedB` → `embed_latent_lfp`.
- Renamed `embed_all` → kept as `embed_all` (compatibility) but returns `(ZA, ZB)` as before.
- **Added `embed_latent(spikes)`** — see confusion note below.
- Updated progress bar label: `"Training"` → `"Stage 1: Latent Encoder"`.
- **Extra:** kept `CoMETs = CoMETsModel` and `train = train_latent` as backward-compat aliases at module bottom.

### `comets/sampler.py`
- Renamed `InBatchSampler` → `LatentSampler`.
- Added `UnitSampler` class per PLAN.md spec (draws two independent pools of M units via `torch.randperm`).
- **Extra:** kept `InBatchSampler = LatentSampler` alias for backward compat.

### `comets/__init__.py`
- Exported all new names: `LatentTrainingConfig`, `UnitTrainingConfig`, `LatentSampler`, `UnitSampler`, `CoMETsModel`, `train_latent`, `UnitEncoderConfig`, `UnitEncoder`, `CoactLoss`, `CtxLoss`, `UnitLoss`, `compute_w_act`, `compute_w_fire`, `compute_w_ctx`, `train_unit`, and all four eval functions.
- Kept all old names as backward-compat exports.
- **Fixed pre-existing bug:** `CEBRASKipEncoder` was in `__all__` but missing a comma (was silently concatenated with `"SkippedConvEncoder"`). Now correctly listed as a separate string.

---

## New files

### `comets/unit_encoder.py`
- `UnitEncoderConfig(n_units, d_u=8, projector=False)` — dataclass.
- `UnitEncoder(nn.Module)` — `nn.Embedding` table + optional linear projector.
- `forward(unit_ids)` returns L2-normalized embeddings `[M, d_u]`.
- `embeddings()` returns all-units tensor `[N, d_u]` detached on CPU.
- **Extra:** `nn.init.normal_(embed.weight, std=0.01)` — small-std init avoids collapsed geometry at step 0.

### `comets/unit_loss.py`
- `compute_w_act(r_A, r_B, tau_act, tau_bias)` → `[M, M]` weight matrix, detached.
- `compute_w_fire(r_A, r_N, tau_fire)` → `[M, M, K]` weight tensor, detached.
- `compute_w_ctx(z_anchor, z_neg)` → `[K]` cosine similarity, clamped ≥ 0, detached.
- `_soft_infonce(e_A, e_B, W, temperature)` — internal helper implementing the soft InfoNCE with guard for all-zero-weight rows.
- `CoactLoss`, `CtxLoss`, `UnitLoss` — `nn.Module` wrappers.
- **Extra:** `_soft_infonce` skips rows where all weights are zero (masked out of the mean) to avoid `-inf` numerators corrupting gradients. Logged with a comment.

### `comets/unit_train.py`
- `train_unit(latent_model, spikes, sampling_cfg, cfg, device)` → `UnitEncoder`.
- Precomputes frozen `Z = latent_model.embed_latent(spikes)` once before the loop.
- Temporal sampling: 1 anchor + 1 positive per step, `K = sampling_cfg.batch_size` negative times.
- Unit sampling: `UnitSampler.sample()` each step.
- Optimizer: `AdamW` + `CosineAnnealingLR` + grad clip 1.0 (matching Stage 1 pattern).
- **Extra:** `steps_per_epoch = max(1, T // K)` to mirror Stage 1 structure; PLAN.md didn't specify steps per epoch for Stage 2.

### `comets/unit_eval.py`
- `ratemap_correlation(E, ratemaps)` — Spearman rho between pairwise embedding distance and ratemap cosine distance. Returns `(rho, p_value)`.
- `place_cell_enrichment(E, is_place_cell, n_perms, seed)` — permutation test on mean intra-class distance. Returns `(observed_stat, p_value)`.
- `split_half_stability(E1, E2)` — `scipy.linalg.orthogonal_procrustes` alignment then mean cosine similarity. Returns scalar.
- `nn_purity(E, labels, k)` — k-NN label purity. Returns scalar.

---

## Confusions / open questions

### 1. `embed_latent` signature ambiguity
PLAN.md rename table says `embed_all → embed_latent`, but `embed_all(A, B)` takes two arguments and returns `(ZA, ZB)`, while the notebook shows `embed_latent(spikes)` taking one argument and returning `[T, D]`.

**Resolution:** kept `embed_all(A, B)` unchanged (returns tuple). Added new `embed_latent(spikes)` that returns the spike-modality embedding only (modality A). This is what Stage 2 needs: a single `[T, D]` latent trajectory derived from the same spike data used to compute firing rates. The plan notebook example is satisfied exactly.

### 2. Stationary embeddings as temporal negatives
The plan says "Negatives: e_B[j] from units sampled at t_neg times". Since `UnitEncoder` embeddings are stationary (don't depend on time), these embeddings are identical to those at t_pos — the temporal indexing only affects firing-rate gates (`r_N`) and context weights (`w_ctx`), not the embedding lookup itself.

**Resolution:** implemented exactly this — `e_B = unit_enc(uid_B)` is called once; the K negative time windows only enter through `r_N [M, K]` and `z_neg [K, D]`. The `pair_weight` sums over K before the InfoNCE, giving an effective [M, M] weight.

### 3. Number of negatives K per step in Stage 2
PLAN.md says "K is the batch size — negatives are the rest of the batch." In Stage 1, the full batch is the negative pool. In Stage 2, there is no explicit "batch" of anchors (we sample 1 anchor time per step). 

**Resolution:** used `K = sampling_cfg.batch_size` as the number of negative time samples drawn per step. This keeps K consistent with Stage 1 and avoids introducing a new hyperparameter.

### 4. `weight_decay` in `UnitTrainingConfig`
PLAN.md's config block does not list `weight_decay`. However, `train_unit` uses `AdamW` which requires it.

**Resolution:** added `weight_decay: float = 1e-5` to `UnitTrainingConfig` to match the `LatentTrainingConfig` default.

### 5. `split_half_stability` signature
PLAN.md says: `split_half_stability(spikes1, spikes2, latent_model, sampling_cfg, unit_cfg)` — implying it trains two unit encoders internally. This would make the function very expensive and require all the training infrastructure inside eval.

**Resolution:** implemented as `split_half_stability(E1, E2)` — the caller trains two `UnitEncoder`s externally on the two halves and passes the resulting embedding matrices. This is cleaner (separation of concerns) and makes the function testable in isolation. The Procrustes alignment + mean cosine similarity is the actual evaluation logic. **Flagged as deviation from plan signature.**

---

## Unchanged files (per plan)
`encoders.py`, `heads.py`, `infonce.py`, `viz.py` — no changes.
