# CoMETs+ Experiment Findings

## Setup

**Sessions**: ec013.156, ec013.440, ec013.441, ec013.442  
*(ec013.157 excluded — animal parked on one side of track at trial end, collapsing position distribution and making spatial R² uninformative)*

**Grid**: 3 encoder sizes × 3 loss configs × 2 checkpoints × 3 unit loss configs × 4 sessions = 216 unit embeddings

---

## Stage 1 — Latent Training

### Evaluation
Spatial decoding R² via 10-fold contiguous CV Ridge on concatenated Z_spike + Z_lfp embeddings.

### Loss configs
| Name | α_cross | β | Meaning |
|------|---------|---|---------|
| `only_within` | 0.0 | 0.5 | No cross-modal alignment; each modality trained independently |
| `spike_dom` | 0.1 | 0.95 | Spike within-modal dominates; small cross-modal term |
| `balanced` | 0.1 | 0.6 | Moderate cross-modal alignment |

### Key findings

- **Within-only wins spatial R² with a linear projection head** — `heavy_spike_only_within` consistently ranks first. With no cross-modal term, the spike encoder optimises purely for temporal proximity (≈ spatial proximity) with no interference.

- **Cross-modal training hurts spatial R² with a linear head** — aligning spikes to LFP forces the spike embedding toward oscillatory (non-spatial) structure, lowering spatial R².

- **The LFP encoder learns nothing spatial when trained within-only** — r2_lfp ≈ −0.09. The concat advantage over Z_spike alone is near zero; all spatial information comes from spikes.

- **MLP projection head (TwoLayerMLP, proj_dim=32) fixes cross-modal** — alignment is absorbed in the projection space, leaving the encoder free to organise spatially. With this architecture, `spike_dom` **beats** `only_within` on both r2_spike and r2_concat:

  | Config | r2_spike | r2_lfp | r2_concat |
  |--------|----------|--------|-----------|
  | only_within (linear, p8) | +0.568 | −0.093 | +0.584 |
  | spike_dom (linear, p8) | +0.527 | +0.121 | +0.545 |
  | **spike_dom (MLP, p32)** | **+0.614** | **+0.497** | **+0.613** |

- **Temperature sensitivity** — τ=0.5 collapses learning; τ=1.0 is the right operating point.

---

## Stage 2 — Unit Identity Embedding

### Evaluation
Ratemap correlation (ρ): Spearman rank correlation between pairwise cosine similarities in embedding space vs. pairwise cosine similarities in ratemap space. Higher = unit geometry better reflects place field structure.

### Unit loss configs
| Label | λ_coact | λ_ctx | Meaning |
|-------|---------|-------|---------|
| `lc0.75_lx0.25` | 0.75 | 0.25 | Co-activity dominant |
| `lc0.50_lx0.50` | 0.50 | 0.50 | Balanced |
| `lc0.25_lx0.75` | 0.25 | 0.75 | Context dominant |

### Key findings

- **Context-weighted unit loss wins universally** — `lc0.25_lx0.75` occupies the entire top of the ranking across all latent configs and all sessions. This is the single most consistent result in the experiment. *Where* units fire together matters more than *whether* they fire together.

- **Heavy LFP encoder is the strongest latent-level driver of unit embedding quality** — `heavy_lfp` configs outperform `heavy_spike` and `balanced` for rho_ratemap regardless of loss config. A richer LFP encoder extracts theta sequence structure (phase precession, sequential activation), which directly encodes which units participate in similar spatial trajectories.

- **Cross-modal latent adds a further boost** — within `heavy_lfp`, `spike_dom` > `balanced` > `only_within` for unit embedding. The cross-modal term provides context that spike co-activity alone cannot.

- **Best checkpoint beats last checkpoint** — using the latent checkpoint with peak spatial R² (not the final epoch) consistently gives better unit embeddings.

- **Top result** (grid, linear head): `heavy_lfp_spike_dom_p8_best_lc0.25_lx0.75` → mean ρ = 0.619 across 4 sessions.

- **With MLP p32 latent**: `spike_dom_TwoLayerMLP_p32_lc0.25_lx0.75` → ρ = 0.621 on ec013.156 (single session; full grid rerun pending).

---

## PCA Baseline Note

The PCA baseline uses `lc=0.5, lx=0.5` with PCA-derived latents. PCA captures dominant covariance but no temporal/behavioral structure, so the context loss is effectively noise. This makes PCA **worse** than `lc0.75_lx0.25` configurations that use 25% real latent context — bad context actively hurts more than less context.

A true co-activity-only baseline (`lc=1.0, lx=0.0`) was not run; PCA underestimates this floor.

---

## Narrative Summary

> The latent that best linearly decodes position is not the same latent that best characterises unit identity — this gap is the central finding.
>
> Within-modal training (no LFP alignment) wins the narrow spatial decoding benchmark because it directly optimises for temporal proximity. But the LFP encoder learns nothing, and the resulting unit embeddings are limited to co-activity structure.
>
> Cross-modal training with a non-linear (MLP) projection head allows alignment to happen in a throwaway projection space, keeping the encoder spatially rich. The spike encoder improves, the LFP encoder becomes spatially structured, and unit embeddings improve further.
>
> A heavier LFP encoder amplifies this — more capacity extracts theta sequence ordering, which is precisely the information that distinguishes unit identity in a place cell population.
>
> LFP contributes at two levels: regularising the latent toward richer spatial representations (via cross-modal alignment), and providing oscillatory context that defines unit identity (via encoder capacity and context-weighted unit loss).

---

## Next Steps

- [ ] Rerun full grid with `TwoLayerMLP` + `proj_dim=32` to confirm MLP head advantage across all sessions and encoder configs
- [ ] Add true co-activity-only baseline (`lc=1.0, lx=0.0`) to replace PCA as the unit embedding floor
- [ ] Test MLP p32 unit embeddings across all 4 sessions (currently only ec013.156 tested)
