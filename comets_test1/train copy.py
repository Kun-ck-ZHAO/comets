"""
train.py — CoMETs two-stage training with Hydra config.

Stage 1: latent population encoder  (spike ↔ LFP contrastive)
Stage 2: unit identity encoder      (co-activity + context-conditioned)

Outputs saved to  runs/{run_name}/
  config.json    — hparams + data shape + timing
  Z_spikes.npy   — [T, proj_dim]  spike latent trajectory
  Z_lfp.npy      — [T, proj_dim]  LFP latent trajectory
  E.npy          — [N, d_u]       unit embeddings

Usage:
  python train.py                                              # defaults
  python train.py session=ec013.252                            # different session
  python train.py encoder=heavy_spike                          # swap encoder preset
  python train.py stage1.alpha_cross=0.0 stage1.beta=0.8      # CEBRA-like
  python train.py --multirun \\
      session=440,441,442,ec013.156,ec013.157 \\
      encoder=balanced,heavy_spike,heavy_lfp \\
      stage1.alpha_cross=0.0,0.2 stage1.beta=0.5,0.8
"""

import json
import pickle
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import comets
from comets import (
    ModalityConfig, SamplingConfig,
    LatentTrainingConfig, UnitTrainingConfig,
    CEBRASKipEncoder, SkippedConvEncoderConfig,
    LinearHead,
    train_latent, train_unit,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session(data_dir: str, session: str):
    """Load full time series from pickle. Returns (spike_count, spike_norm, lfp, position, meta)."""
    path = Path(data_dir) / session / f"{session}_lfp_spike.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)

    lfp         = np.asarray(data["lfp_theta_50hz"],   dtype=np.float32)
    spike_count = np.asarray(data["spike_bin"],         dtype=np.float32)
    spike_rate  = np.asarray(data["spike_rate"],        dtype=np.float32)
    position    = np.asarray(data["position_50hz"],     dtype=np.float64)
    fs_common   = float(data["fs_common"])

    if position.ndim == 2:
        position = position[:, 0]

    rmin = spike_rate.min(axis=0, keepdims=True)
    rmax = spike_rate.max(axis=0, keepdims=True)
    spike_norm = (spike_rate - rmin) / (rmax - rmin + 1e-9)

    meta = dict(fs_common=fs_common)
    return spike_count, spike_norm, lfp, position, meta


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.to_absolute_path("."))

    # --- data ---
    data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    spike_count, spike_norm, lfp, position, meta = load_session(data_dir, cfg.session)
    T, N = spike_norm.shape
    C    = lfp.shape[1]
    fs   = meta["fs_common"]

    print(f"\nSession: {cfg.session}   T={T}  N={N}  C={C}  fs={fs} Hz")

    # --- run name ---
    s1, s2 = cfg.stage1, cfg.stage2
    run_name = cfg.run_name or (
        f"{cfg.session}"
        f"_{cfg.encoder_name}"
        f"_a{s1.alpha_cross}"
        f"_b{s1.beta}"
        f"_p{s1.proj_dim}"
        f"_lc{s2.lambda_coact}"
        f"_lx{s2.lambda_ctx}"
        f"_du{s2.d_u}"
    )
    out_dir = root / cfg.runs_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_name}  →  {out_dir}")

    # --- modality configs ---
    mA, mB = cfg.modA, cfg.modB
    modA = ModalityConfig(
        encoder_cls=CEBRASKipEncoder,
        encoder_cfg=SkippedConvEncoderConfig(
            in_channels=N, emb_dim=mA.emb_dim, hidden=mA.hidden, num_blocks=mA.num_blocks,
        ),
        win_len=mA.win_len,
    )
    modB = ModalityConfig(
        encoder_cls=CEBRASKipEncoder,
        encoder_cfg=SkippedConvEncoderConfig(
            in_channels=C, emb_dim=mB.emb_dim, hidden=mB.hidden, num_blocks=mB.num_blocks,
        ),
        win_len=mB.win_len,
    )
    sampling = SamplingConfig(
        batch_size=cfg.training.batch_size,
        pos_offset=cfg.training.pos_offset,
        negative_mask_dist=cfg.training.negative_mask_dist,
    )

    # --- Stage 1 ---
    latent_cfg = LatentTrainingConfig(
        epochs=s1.epochs,
        lr=s1.lr,
        temperature=1.0,
        proj_dim=s1.proj_dim,
        alpha_cross=s1.alpha_cross,
        beta=s1.beta,
        head_cls=LinearHead,
    )
    print(f"\n[Stage 1] epochs={s1.epochs}  proj_dim={s1.proj_dim}  "
          f"alpha_cross={s1.alpha_cross}  beta={s1.beta}")
    t0 = time.time()
    latent_model = train_latent(spike_norm, modA, lfp, modB, sampling, latent_cfg)
    t_latent = time.time() - t0
    print(f"Stage 1 done in {t_latent:.1f}s")

    Z_spikes = latent_model.embed_latent_spikes(spike_norm)   # [T, proj_dim]
    Z_lfp    = latent_model.embed_latent_lfp(lfp)             # [T, proj_dim]

    # --- Stage 2 ---
    unit_cfg = UnitTrainingConfig(
        epochs=s2.epochs,
        lr=s2.lr,
        d_u=s2.d_u,
        lambda_coact=s2.lambda_coact,
        lambda_ctx=s2.lambda_ctx,
    )
    print(f"\n[Stage 2] epochs={s2.epochs}  d_u={s2.d_u}  "
          f"lambda_coact={s2.lambda_coact}  lambda_ctx={s2.lambda_ctx}")
    t1 = time.time()
    unit_enc = train_unit(
        latent_model, spike_count, sampling, unit_cfg,
        z_spikes=Z_spikes, z_lfp=Z_lfp,
    )
    t_unit = time.time() - t1
    print(f"Stage 2 done in {t_unit:.1f}s")

    E = unit_enc.embeddings().numpy()   # [N, d_u]

    # --- save ---
    np.save(out_dir / "Z_spikes.npy", Z_spikes)
    np.save(out_dir / "Z_lfp.npy",    Z_lfp)
    np.save(out_dir / "E.npy",         E)

    config_out = dict(
        run_name=run_name,
        session=cfg.session,
        encoder_name=cfg.encoder_name,
        data_dir=data_dir,
        proj_dim=int(s1.proj_dim),
        alpha_cross=float(s1.alpha_cross),
        beta=float(s1.beta),
        latent_epochs=int(s1.epochs),
        latent_lr=float(s1.lr),
        d_u=int(s2.d_u),
        lambda_coact=float(s2.lambda_coact),
        lambda_ctx=float(s2.lambda_ctx),
        unit_epochs=int(s2.epochs),
        unit_lr=float(s2.lr),
        batch_size=int(cfg.training.batch_size),
        pos_offset=int(cfg.training.pos_offset),
        negative_mask_dist=int(cfg.training.negative_mask_dist),
        modA_emb_dim=int(mA.emb_dim), modA_hidden=int(mA.hidden),
        modA_num_blocks=int(mA.num_blocks), modA_win_len=int(mA.win_len),
        modB_emb_dim=int(mB.emb_dim), modB_hidden=int(mB.hidden),
        modB_num_blocks=int(mB.num_blocks), modB_win_len=int(mB.win_len),
        N=N, T=T, C=C,
        fs_common=fs,
        t_latent_s=round(t_latent, 2),
        t_unit_s=round(t_unit, 2),
    )
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print(f"  Z_spikes: {Z_spikes.shape}  Z_lfp: {Z_lfp.shape}  E: {E.shape}")
    print(f"  Total time: {t_latent + t_unit:.1f}s")


if __name__ == "__main__":
    main()
