"""
unit_train.py — Stage 2: train unit identity encoder on all sessions.

Loads frozen latent embeddings from a completed latent run, trains unit
encoder, evaluates ratemap_correlation every eval_every epochs.
Saves E_best and E_last per session.

Outputs per run:
  runs/unit/{latent_run}_{ckp}_lc{lc}_lx{lx}_du{D}/
    config.json
    {session}/
      E_best.npy   E_last.npy
      unit_log.csv    epoch, rho_ratemap
      config.json     N, best_epoch, best_rho, elapsed_s
    train_log.txt

Usage:
  python unit_train.py latent_run_dir=runs/latent/balanced_only_within_p8
  python unit_train.py latent_run_dir=... latent_ckp=last \\
      stage2.lambda_coact=0.75 stage2.lambda_ctx=0.25
"""

import csv
import json
import pickle
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter1d

from comets import SamplingConfig, UnitTrainingConfig, train_unit
from comets.unit_eval import ratemap_correlation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_session_raw(data_dir: str, session: str):
    path = Path(data_dir) / session / f"{session}_lfp_spike.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    spike_count = np.asarray(data["spike_bin"],     dtype=np.float32)
    position    = np.asarray(data["position_50hz"], dtype=np.float64)
    fs          = float(data["fs_common"])
    if position.ndim == 2:
        position = position[:, 0]
    return spike_count, position, fs


def compute_ratemaps(spike_counts, position, n_bins=40, sigma=2.0, fs=50.0):
    edges    = np.linspace(position.min(), position.max(), n_bins + 1)
    pos_bins = np.clip(np.digitize(position, edges) - 1, 0, n_bins - 1)
    occ      = np.bincount(pos_bins, minlength=n_bins) / fs
    rm       = np.zeros((spike_counts.shape[1], n_bins), dtype=np.float32)
    for i in range(spike_counts.shape[1]):
        spk   = np.bincount(pos_bins, weights=spike_counts[:, i], minlength=n_bins)
        rm[i] = spk / (occ + 1e-9)
    return gaussian_filter1d(rm, sigma=sigma, axis=1)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="unit_train")
def main(cfg: DictConfig) -> None:
    root           = Path(hydra.utils.to_absolute_path("."))
    latent_run_dir = Path(hydra.utils.to_absolute_path(cfg.latent_run_dir))

    with open(latent_run_dir / "config.json") as f:
        lrc = json.load(f)

    latent_run_name = lrc["run_name"]
    data_dir        = lrc["data_dir"]
    sessions        = lrc["sessions"]
    s2              = cfg.stage2

    run_name = (f"{latent_run_name}_{cfg.latent_ckp}"
                f"_lc{s2.lambda_coact}_lx{s2.lambda_ctx}_du{s2.d_u}")
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
    log(f"Latent: {latent_run_name}  ckp={cfg.latent_ckp}")
    log(f"  lc={s2.lambda_coact}  lx={s2.lambda_ctx}  d_u={s2.d_u}")
    log(f"  epochs={s2.epochs}  eval_every={cfg.eval_every}")
    log(f"Sessions ({len(sessions)}): {sessions}")
    log(f"{'='*60}")

    with open(run_dir / "config.json", "w") as f:
        json.dump(dict(
            run_name=run_name,
            latent_run_name=latent_run_name,
            latent_run_dir=str(latent_run_dir),
            latent_ckp=cfg.latent_ckp,
            d_u=int(s2.d_u),
            lambda_coact=float(s2.lambda_coact),
            lambda_ctx=float(s2.lambda_ctx),
            epochs=int(s2.epochs),
            lr=float(s2.lr),
            eval_every=int(cfg.eval_every),
            sessions=sessions,
            data_dir=data_dir,
        ), f, indent=2)

    sampling = SamplingConfig(batch_size=cfg.training.batch_size)
    n_sess   = len(sessions)

    for si, session in enumerate(sessions):
        log(f"\n[Session {si+1}/{n_sess}] {session}  (ckp={cfg.latent_ckp})")

        Z_spikes = np.load(latent_run_dir / session / f"Z_spikes_{cfg.latent_ckp}.npy")
        Z_lfp    = np.load(latent_run_dir / session / f"Z_lfp_{cfg.latent_ckp}.npy")

        spike_count, position, fs = load_session_raw(data_dir, session)
        T, N = spike_count.shape
        log(f"  T={T}  N={N}")

        ratemaps = compute_ratemaps(spike_count, position, fs=fs)

        sess_dir = run_dir / session
        sess_dir.mkdir(exist_ok=True)

        with open(sess_dir / "unit_log.csv", "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "rho_ratemap"])

        unit_cfg = UnitTrainingConfig(
            epochs=s2.epochs, lr=s2.lr,
            d_u=s2.d_u, lambda_coact=s2.lambda_coact, lambda_ctx=s2.lambda_ctx,
        )

        best = {"rho": -np.inf, "epoch": -1}

        def make_callback(rm=ratemaps, sdir=sess_dir, b=best, lg=log):
            def cb(epoch, unit_enc):
                E   = unit_enc.embeddings().numpy()
                rho, _ = ratemap_correlation(E, rm)
                with open(sdir / "unit_log.csv", "a", newline="") as f:
                    csv.writer(f).writerow([epoch, f"{rho:.6f}"])
                is_best = rho > b["rho"]
                if is_best:
                    b["rho"] = rho;  b["epoch"] = epoch
                    np.save(sdir / "E_best.npy", E)
                lg(f"  epoch {epoch:3d}  rho_ratemap={rho:+.4f}" + ("  ← best" if is_best else ""))
            return cb

        t0       = time.time()
        unit_enc = train_unit(
            None, spike_count, sampling, unit_cfg,
            z_spikes=Z_spikes, z_lfp=Z_lfp,
            epoch_callback=make_callback(),
            eval_every=cfg.eval_every,
        )
        elapsed = time.time() - t0

        E_last = unit_enc.embeddings().numpy()
        np.save(sess_dir / "E_last.npy", E_last)

        with open(sess_dir / "config.json", "w") as f:
            json.dump(dict(
                session=session, N=N, T=T,
                best_epoch=best["epoch"], best_rho=float(best["rho"]),
                elapsed_s=round(elapsed, 1),
            ), f, indent=2)

        log(f"  Finished in {elapsed:.0f}s  "
            f"best_epoch={best['epoch']}  best_rho={best['rho']:+.4f}")

    log(f"\nAll {n_sess} sessions done  →  {run_dir}/")


if __name__ == "__main__":
    main()
