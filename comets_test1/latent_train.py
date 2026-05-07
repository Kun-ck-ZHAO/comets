"""
latent_train.py — Stage 1: train latent population encoder on all sessions.

Evaluates spatial decoding (10-fold CV R² on concat embeddings) every
eval_every epochs. Saves Z_spikes_best/last and Z_lfp_best/last per session.

Outputs per run:
  runs/latent/{encoder}_{loss}_p{D}/
    config.json                   shared hparams
    {session}/
      Z_spikes_best.npy   Z_spikes_last.npy
      Z_lfp_best.npy      Z_lfp_last.npy
      latent_log.csv      epoch, r2_concat
      config.json         N, T, C, fs, best_epoch, best_r2, elapsed_s
    train_log.txt         timestamped console log

Usage (see conf/latent_train.yaml for full options):
  python latent_train.py encoder=heavy_spike loss_name=spike_dom \\
      stage1.alpha_cross=0.1 stage1.beta=0.95
"""

import csv
import json
import pickle
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from comets import (
    ModalityConfig, SamplingConfig, LatentTrainingConfig,
    CEBRASKipEncoder, SkippedConvEncoderConfig, LinearHead,
    train_latent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_session(data_dir: str, session: str):
    path = Path(data_dir) / session / f"{session}_lfp_spike.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    lfp         = np.asarray(data["lfp_theta_50hz"], dtype=np.float32)
    spike_count = np.asarray(data["spike_bin"],      dtype=np.float32)
    spike_rate  = np.asarray(data["spike_rate"],     dtype=np.float32)
    position    = np.asarray(data["position_50hz"],  dtype=np.float64)
    fs          = float(data["fs_common"])
    if position.ndim == 2:
        position = position[:, 0]
    rmin = spike_rate.min(axis=0, keepdims=True)
    rmax = spike_rate.max(axis=0, keepdims=True)
    spike_norm = (spike_rate - rmin) / (rmax - rmin + 1e-9)
    return spike_norm, lfp, position, fs


def window_kfold_r2(Z, position, win_size=10, n_folds=10, alpha=1.0):
    """10-fold contiguous CV Ridge → mean R²."""
    Z   = np.asarray(Z,        dtype=np.float64)
    pos = np.asarray(position, dtype=np.float64)
    T_  = min(len(Z), len(pos))
    nw  = T_ // win_size
    Z_w = Z[:nw * win_size].reshape(nw, win_size, -1).mean(axis=1)
    y_w = pos[:nw * win_size].reshape(nw, win_size).mean(axis=1)
    fs  = nw // n_folds
    n   = fs * n_folds
    Z_w, y_w = Z_w[:n], y_w[:n]
    r2s = []
    for k in range(n_folds):
        mask = np.zeros(n, dtype=bool)
        mask[k * fs:(k + 1) * fs] = True
        reg = Ridge(alpha=alpha).fit(Z_w[~mask], y_w[~mask])
        r2s.append(float(r2_score(y_w[mask], reg.predict(Z_w[mask]))))
    return float(np.mean(r2s))


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="latent_train")
def main(cfg: DictConfig) -> None:
    root     = Path(hydra.utils.to_absolute_path("."))
    data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    s1       = cfg.stage1

    run_name = f"{cfg.encoder_name}_{cfg.loss_name}_p{s1.proj_dim}"
    run_dir  = root / cfg.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "train_log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"{'='*60}")
    log(f"Run: {run_name}")
    log(f"Encoder: {cfg.encoder_name}   Loss: {cfg.loss_name}")
    log(f"  alpha_cross={s1.alpha_cross}  beta={s1.beta}  proj_dim={s1.proj_dim}")
    log(f"  epochs={s1.epochs}  eval_every={cfg.eval_every}")
    log(f"Sessions ({len(cfg.sessions)}): {list(cfg.sessions)}")
    log(f"{'='*60}")

    # shared run config
    with open(run_dir / "config.json", "w") as f:
        json.dump(dict(
            run_name=run_name,
            encoder_name=cfg.encoder_name,
            loss_name=cfg.loss_name,
            proj_dim=int(s1.proj_dim),
            alpha_cross=float(s1.alpha_cross),
            beta=float(s1.beta),
            epochs=int(s1.epochs),
            lr=float(s1.lr),
            eval_every=int(cfg.eval_every),
            batch_size=int(cfg.training.batch_size),
            sessions=list(cfg.sessions),
            data_dir=data_dir,
            modA=dict(cfg.modA),
            modB=dict(cfg.modB),
        ), f, indent=2)

    mA, mB = cfg.modA, cfg.modB
    n_sess = len(cfg.sessions)

    for si, session in enumerate(cfg.sessions):
        log(f"\n[Session {si+1}/{n_sess}] {session}")

        spike_norm, lfp, position, fs = load_session(data_dir, session)
        T, N = spike_norm.shape
        C    = lfp.shape[1]
        log(f"  T={T}  N={N}  C={C}  fs={fs} Hz")

        sess_dir = run_dir / session
        sess_dir.mkdir(exist_ok=True)

        # init log CSV
        with open(sess_dir / "latent_log.csv", "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "r2_concat"])

        modA = ModalityConfig(
            encoder_cls=CEBRASKipEncoder,
            encoder_cfg=SkippedConvEncoderConfig(
                in_channels=N, emb_dim=mA.emb_dim,
                hidden=mA.hidden, num_blocks=mA.num_blocks,
            ),
            win_len=mA.win_len,
        )
        modB = ModalityConfig(
            encoder_cls=CEBRASKipEncoder,
            encoder_cfg=SkippedConvEncoderConfig(
                in_channels=C, emb_dim=mB.emb_dim,
                hidden=mB.hidden, num_blocks=mB.num_blocks,
            ),
            win_len=mB.win_len,
        )
        sampling = SamplingConfig(
            batch_size=cfg.training.batch_size,
            pos_offset=cfg.training.pos_offset,
            negative_mask_dist=cfg.training.negative_mask_dist,
        )
        latent_cfg = LatentTrainingConfig(
            epochs=s1.epochs, lr=s1.lr, temperature=1.0,
            proj_dim=s1.proj_dim, alpha_cross=s1.alpha_cross,
            beta=s1.beta, head_cls=LinearHead,
        )

        best = {"r2": -np.inf, "epoch": -1}

        def make_callback(spk=spike_norm, lf=lfp, pos=position,
                          sdir=sess_dir, b=best, lg=log):
            def cb(epoch, model):
                Z_spk = model.embed_latent_spikes(spk)
                Z_lf  = model.embed_latent_lfp(lf)
                r2    = window_kfold_r2(np.concatenate([Z_spk, Z_lf], axis=1), pos)
                with open(sdir / "latent_log.csv", "a", newline="") as f:
                    csv.writer(f).writerow([epoch, f"{r2:.6f}"])
                is_best = r2 > b["r2"]
                if is_best:
                    b["r2"] = r2;  b["epoch"] = epoch
                    np.save(sdir / "Z_spikes_best.npy", Z_spk)
                    np.save(sdir / "Z_lfp_best.npy",    Z_lf)
                lg(f"  epoch {epoch:3d}  r2_concat={r2:+.4f}" + ("  ← best" if is_best else ""))
            return cb

        t0    = time.time()
        model = train_latent(
            spike_norm, modA, lfp, modB, sampling, latent_cfg,
            epoch_callback=make_callback(),
            eval_every=cfg.eval_every,
        )
        elapsed = time.time() - t0

        Z_spk_last = model.embed_latent_spikes(spike_norm)
        Z_lf_last  = model.embed_latent_lfp(lfp)
        np.save(sess_dir / "Z_spikes_last.npy", Z_spk_last)
        np.save(sess_dir / "Z_lfp_last.npy",    Z_lf_last)

        with open(sess_dir / "config.json", "w") as f:
            json.dump(dict(
                session=session, N=N, T=T, C=C, fs_common=fs,
                best_epoch=best["epoch"], best_r2=float(best["r2"]),
                elapsed_s=round(elapsed, 1),
            ), f, indent=2)

        log(f"  Finished in {elapsed:.0f}s  "
            f"best_epoch={best['epoch']}  best_r2={best['r2']:+.4f}")

    log(f"\nAll {n_sess} sessions done  →  {run_dir}/")


if __name__ == "__main__":
    main()
