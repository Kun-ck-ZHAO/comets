"""
eval.py — CoMETs evaluation with Hydra config.

Loads a saved run (Z_spikes, Z_lfp, E, config.json), reloads raw data,
runs 10-fold contiguous CV for spatial decoding and unit embedding metrics,
then appends one row to results/results.csv.

Usage:
  python eval.py run_dir=runs/ec013.157_a0.2_b0.8_p8_lc1.0_lx1.0_du8
  for d in runs/*/; do python eval.py run_dir=$d; done
"""

import csv
import json
import os
import pickle
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA as skPCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from comets.unit_eval import ratemap_correlation, si_gradient_correlation


# ---------------------------------------------------------------------------
# Spatial helpers (self-contained; fs passed explicitly)
# ---------------------------------------------------------------------------

def compute_ratemaps(spike_counts, position, n_bins=40, sigma=2.0, fs=50.0):
    edges    = np.linspace(position.min(), position.max(), n_bins + 1)
    centers  = 0.5 * (edges[:-1] + edges[1:])
    pos_bins = np.clip(np.digitize(position, edges) - 1, 0, n_bins - 1)
    occ      = np.bincount(pos_bins, minlength=n_bins) / fs
    rm       = np.zeros((spike_counts.shape[1], n_bins))
    for i in range(spike_counts.shape[1]):
        spk   = np.bincount(pos_bins, weights=spike_counts[:, i], minlength=n_bins)
        rm[i] = spk / (occ + 1e-9)
    rm = gaussian_filter1d(rm, sigma=sigma, axis=1)
    return rm, centers


def spatial_information(ratemaps, position, n_bins=40):
    edges    = np.linspace(position.min(), position.max(), n_bins + 1)
    pos_bins = np.clip(np.digitize(position, edges) - 1, 0, n_bins - 1)
    p_x      = np.bincount(pos_bins, minlength=n_bins).astype(float)
    p_x     /= p_x.sum()
    mean_rt  = (ratemaps * p_x[None, :]).sum(axis=1, keepdims=True)
    ratio    = ratemaps / (mean_rt + 1e-9)
    return (p_x[None, :] * ratio * np.log2(np.clip(ratio, 1e-9, None))).sum(axis=1)


# ---------------------------------------------------------------------------
# 10-fold contiguous CV
# ---------------------------------------------------------------------------

def window_kfold(Z, position, Z_lfp=None, win_size=10, n_folds=10, alpha=1.0):
    """
    Non-overlapping window averaging + contiguous k-fold CV with Ridge regression.

    Returns (mae_mean, mae_std, r2_mean, r2_std).
    """
    Z   = np.asarray(Z,        dtype=np.float64)
    pos = np.asarray(position, dtype=np.float64)
    T_  = min(len(Z), len(pos))
    nw  = T_ // win_size
    Z_w = Z[:nw * win_size].reshape(nw, win_size, -1).mean(axis=1)
    y_w = pos[:nw * win_size].reshape(nw, win_size).mean(axis=1)

    if Z_lfp is not None:
        Zl   = np.asarray(Z_lfp, dtype=np.float64)
        Zl_w = Zl[:nw * win_size].reshape(nw, win_size, -1).mean(axis=1)
        Z_w  = np.concatenate([Z_w, Zl_w], axis=1)

    fold_size = nw // n_folds
    n_use     = fold_size * n_folds
    Z_w = Z_w[:n_use]
    y_w = y_w[:n_use]

    maes, r2s = [], []
    for k in range(n_folds):
        test_mask  = np.zeros(n_use, dtype=bool)
        test_mask[k * fold_size:(k + 1) * fold_size] = True
        X_tr, y_tr = Z_w[~test_mask], y_w[~test_mask]
        X_te, y_te = Z_w[test_mask],  y_w[test_mask]
        reg   = Ridge(alpha=alpha).fit(X_tr, y_tr)
        y_hat = reg.predict(X_te)
        maes.append(float(np.mean(np.abs(y_te - y_hat))))
        r2s.append(float(r2_score(y_te, y_hat)))

    return float(np.mean(maes)), float(np.std(maes)), float(np.mean(r2s)), float(np.std(r2s))


# ---------------------------------------------------------------------------
# Data loading (mirrors train.py)
# ---------------------------------------------------------------------------

def load_session(data_dir: str, session: str):
    path = Path(data_dir) / session / f"{session}_lfp_spike.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)

    lfp         = np.asarray(data["lfp_theta_50hz"], dtype=np.float32)
    spike_count = np.asarray(data["spike_bin"],       dtype=np.float32)
    spike_rate  = np.asarray(data["spike_rate"],      dtype=np.float32)
    position    = np.asarray(data["position_50hz"],   dtype=np.float64)
    fs_common   = float(data["fs_common"])

    if position.ndim == 2:
        position = position[:, 0]

    rmin = spike_rate.min(axis=0, keepdims=True)
    rmax = spike_rate.max(axis=0, keepdims=True)
    spike_norm = (spike_rate - rmin) / (rmax - rmin + 1e-9)

    return spike_count, spike_norm, lfp, position, fs_common


# ---------------------------------------------------------------------------
# CSV append helper
# ---------------------------------------------------------------------------

def append_csv(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg: DictConfig) -> None:
    run_dir = Path(hydra.utils.to_absolute_path(cfg.run_dir))

    # --- load run config ---
    with open(run_dir / "config.json") as f:
        rc = json.load(f)

    print(f"\nEvaluating: {rc['run_name']}")
    print(f"  session={rc['session']}  T={rc['T']}  N={rc['N']}  C={rc['C']}")

    # --- load embeddings ---
    Z_spikes = np.load(run_dir / "Z_spikes.npy")   # [T, proj_dim]
    Z_lfp    = np.load(run_dir / "Z_lfp.npy")       # [T, proj_dim]
    E        = np.load(run_dir / "E.npy")            # [N, d_u]

    # --- reload raw data ---
    spike_count, spike_norm, lfp, position, fs = load_session(rc["data_dir"], rc["session"])

    # --- PCA baseline latents ---
    proj_dim = rc["proj_dim"]
    Z_pca_spk = skPCA(n_components=proj_dim, random_state=0).fit_transform(spike_norm).astype(np.float32)
    Z_pca_lfp = skPCA(n_components=proj_dim, random_state=0).fit_transform(lfp).astype(np.float32)

    ws      = cfg.win_size
    nf      = cfg.n_folds

    # --- 10-fold CV: CoMETs latents ---
    print("\n[Latent decoding — 10-fold CV]")
    mae_spk, mae_spk_std, r2_spk, r2_spk_std = window_kfold(Z_spikes, position, win_size=ws, n_folds=nf)
    mae_lfp, mae_lfp_std, r2_lfp, r2_lfp_std = window_kfold(Z_lfp,    position, win_size=ws, n_folds=nf)
    mae_cat, mae_cat_std, r2_cat, r2_cat_std = window_kfold(Z_spikes, position, Z_lfp=Z_lfp, win_size=ws, n_folds=nf)
    print(f"  CoMETs spk  MAE={mae_spk:.3f}±{mae_spk_std:.3f}  R²={r2_spk:.3f}±{r2_spk_std:.3f}")
    print(f"  CoMETs lfp  MAE={mae_lfp:.3f}±{mae_lfp_std:.3f}  R²={r2_lfp:.3f}±{r2_lfp_std:.3f}")
    print(f"  CoMETs cat  MAE={mae_cat:.3f}±{mae_cat_std:.3f}  R²={r2_cat:.3f}±{r2_cat_std:.3f}")

    # --- 10-fold CV: PCA latents ---
    pca_mae_spk, pca_mae_spk_std, pca_r2_spk, pca_r2_spk_std = window_kfold(Z_pca_spk, position, win_size=ws, n_folds=nf)
    pca_mae_lfp, pca_mae_lfp_std, pca_r2_lfp, pca_r2_lfp_std = window_kfold(Z_pca_lfp, position, win_size=ws, n_folds=nf)
    pca_mae_cat, pca_mae_cat_std, pca_r2_cat, pca_r2_cat_std = window_kfold(Z_pca_spk, position, Z_lfp=Z_pca_lfp, win_size=ws, n_folds=nf)
    print(f"  PCA    spk  MAE={pca_mae_spk:.3f}±{pca_mae_spk_std:.3f}  R²={pca_r2_spk:.3f}±{pca_r2_spk_std:.3f}")
    print(f"  PCA    lfp  MAE={pca_mae_lfp:.3f}±{pca_mae_lfp_std:.3f}  R²={pca_r2_lfp:.3f}±{pca_r2_lfp_std:.3f}")
    print(f"  PCA    cat  MAE={pca_mae_cat:.3f}±{pca_mae_cat_std:.3f}  R²={pca_r2_cat:.3f}±{pca_r2_cat_std:.3f}")

    # --- ratemaps + SI ---
    print("\n[Unit metrics]")
    ratemaps, _ = compute_ratemaps(spike_count, position, n_bins=cfg.n_bins, sigma=cfg.sigma, fs=fs)
    si_obs      = spatial_information(ratemaps, position, n_bins=cfg.n_bins)

    rho_ratemap,  p_ratemap  = ratemap_correlation(E, ratemaps)
    rho_si_grad,  p_si_grad  = si_gradient_correlation(E, si_obs)

    d_u = rc["d_u"]
    E_pca = skPCA(n_components=d_u, random_state=0).fit_transform(spike_norm).astype(np.float32)
    pca_rho_ratemap, pca_p_ratemap = ratemap_correlation(E_pca, ratemaps)
    pca_rho_si_grad, pca_p_si_grad = si_gradient_correlation(E_pca, si_obs)

    print(f"  CoMETs  ρ_ratemap={rho_ratemap:+.3f}  ρ_SI_grad={rho_si_grad:+.3f}")
    print(f"  PCA     ρ_ratemap={pca_rho_ratemap:+.3f}  ρ_SI_grad={pca_rho_si_grad:+.3f}")

    # --- build CSV row ---
    row = dict(
        run_name=rc["run_name"],
        session=rc["session"],
        proj_dim=proj_dim,
        alpha_cross=rc["alpha_cross"],
        beta=rc["beta"],
        latent_epochs=rc["latent_epochs"],
        d_u=d_u,
        lambda_coact=rc["lambda_coact"],
        lambda_ctx=rc["lambda_ctx"],
        unit_epochs=rc["unit_epochs"],
        batch_size=rc["batch_size"],
        N=rc["N"], T=rc["T"], C=rc["C"],
        # CoMETs latent
        mae_spk=mae_spk, mae_spk_std=mae_spk_std, r2_spk=r2_spk, r2_spk_std=r2_spk_std,
        mae_lfp=mae_lfp, mae_lfp_std=mae_lfp_std, r2_lfp=r2_lfp, r2_lfp_std=r2_lfp_std,
        mae_cat=mae_cat, mae_cat_std=mae_cat_std, r2_cat=r2_cat, r2_cat_std=r2_cat_std,
        # PCA latent
        pca_mae_spk=pca_mae_spk, pca_mae_spk_std=pca_mae_spk_std,
        pca_r2_spk=pca_r2_spk,   pca_r2_spk_std=pca_r2_spk_std,
        pca_mae_lfp=pca_mae_lfp, pca_mae_lfp_std=pca_mae_lfp_std,
        pca_r2_lfp=pca_r2_lfp,   pca_r2_lfp_std=pca_r2_lfp_std,
        pca_mae_cat=pca_mae_cat, pca_mae_cat_std=pca_mae_cat_std,
        pca_r2_cat=pca_r2_cat,   pca_r2_cat_std=pca_r2_cat_std,
        # CoMETs unit
        rho_ratemap=rho_ratemap, p_ratemap=p_ratemap,
        rho_si_grad=rho_si_grad, p_si_grad=p_si_grad,
        # PCA unit
        pca_rho_ratemap=pca_rho_ratemap, pca_p_ratemap=pca_p_ratemap,
        pca_rho_si_grad=pca_rho_si_grad, pca_p_si_grad=pca_p_si_grad,
        # meta
        n_folds=nf, win_size=ws,
        t_latent_s=rc.get("t_latent_s", ""), t_unit_s=rc.get("t_unit_s", ""),
    )

    csv_path = Path(hydra.utils.to_absolute_path(cfg.results_csv))
    append_csv(csv_path, row)
    print(f"\nAppended to {csv_path}")


if __name__ == "__main__":
    main()
