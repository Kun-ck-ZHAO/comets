#!/bin/bash
# grid.sh — full CoMETs experiment grid
# Phase 1: 9 latent runs  (3 encoders × 3 loss configs, all 5 sessions each)
# Phase 2: 54 unit runs   (9 latent × 2 ckps × 3 unit configs, all 5 sessions each)

set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

PYTHON=/home/kunz/miniconda3/envs/torchDL/bin/python

# ── Loss configs ──────────────────────────────────────────────────────────────
LOSS_NAMES=(only_within spike_dom balanced)
declare -A LOSS_ARGS
LOSS_ARGS[only_within]="stage1.alpha_cross=0.0 stage1.beta=0.5"
LOSS_ARGS[spike_dom]="stage1.alpha_cross=0.1 stage1.beta=0.95"
LOSS_ARGS[balanced]="stage1.alpha_cross=0.1 stage1.beta=0.6"

# ── Encoder presets ───────────────────────────────────────────────────────────
ENCODERS=(balanced heavy_spike heavy_lfp)

# ── Unit configs ──────────────────────────────────────────────────────────────
declare -A UNIT_ARGS
UNIT_ARGS[lc0.75_lx0.25]="stage2.lambda_coact=0.75 stage2.lambda_ctx=0.25"
UNIT_ARGS[lc0.50_lx0.50]="stage2.lambda_coact=0.50 stage2.lambda_ctx=0.50"
UNIT_ARGS[lc0.25_lx0.75]="stage2.lambda_coact=0.25 stage2.lambda_ctx=0.75"
UNIT_LABELS=(lc0.75_lx0.25 lc0.50_lx0.50 lc0.25_lx0.75)

CKPS=(best last)
PROJ_DIM=8

total_latent=$((${#ENCODERS[@]} * ${#LOSS_NAMES[@]}))
total_unit=$((total_latent * ${#CKPS[@]} * ${#UNIT_LABELS[@]}))

echo "=================================================="
echo "  PHASE 1: Latent training  (${total_latent} runs × 5 sessions)"
echo "=================================================="

latent_done=0
latent_skip=0

for encoder in "${ENCODERS[@]}"; do
  for loss in "${LOSS_NAMES[@]}"; do
    run_name="${encoder}_${loss}_p${PROJ_DIM}"
    check="runs/latent/${run_name}/ec013.156/Z_spikes_last.npy"

    if [ -f "$check" ]; then
      echo "[SKIP] latent: ${run_name}"
      ((latent_skip++)) || true
      continue
    fi

    echo ""
    echo ">>> [latent $((latent_done+latent_skip+1))/${total_latent}] ${run_name}"
    $PYTHON latent_train.py \
      encoder="${encoder}" \
      loss_name="${loss}" \
      ${LOSS_ARGS[$loss]}
    ((latent_done++)) || true
  done
done

echo ""
echo "Phase 1 done: ${latent_done} trained, ${latent_skip} skipped."

echo ""
echo "=================================================="
echo "  PHASE 2: Unit training  (${total_unit} runs × 5 sessions)"
echo "=================================================="

unit_done=0
unit_skip=0

for encoder in "${ENCODERS[@]}"; do
  for loss in "${LOSS_NAMES[@]}"; do
    latent_run="runs/latent/${encoder}_${loss}_p${PROJ_DIM}"

    if [ ! -d "$latent_run" ]; then
      echo "[SKIP] no latent dir: ${latent_run}"
      continue
    fi

    for ckp in "${CKPS[@]}"; do
      for ulabel in "${UNIT_LABELS[@]}"; do
        unit_run="${encoder}_${loss}_p${PROJ_DIM}_${ckp}_${ulabel}_du${PROJ_DIM}"
        check="runs/unit/${unit_run}/ec013.156/E_last.npy"

        if [ -f "$check" ]; then
          echo "[SKIP] unit: ${unit_run}"
          ((unit_skip++)) || true
          continue
        fi

        echo ""
        echo ">>> [unit $((unit_done+unit_skip+1))/${total_unit}] ${unit_run}"
        $PYTHON unit_train.py \
          latent_run_dir="${latent_run}" \
          latent_ckp="${ckp}" \
          ${UNIT_ARGS[$ulabel]}
        ((unit_done++)) || true
      done
    done
  done
done

echo ""
echo "Phase 2 done: ${unit_done} trained, ${unit_skip} skipped."
echo ""
echo "=================================================="
echo "  ALL DONE"
echo "  Latent embeddings : runs/latent/"
echo "  Unit embeddings   : runs/unit/"
echo "=================================================="
