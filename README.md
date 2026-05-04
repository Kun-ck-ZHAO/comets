# NeuralManifold CoMET Workspace

CoMET/CoMETs is a contrastive multi-view embedding toolkit for time series, with this repository focused on:
- reusable modeling code in `comets/`
- experiment notebooks for synthetic and real neural data
- cached checkpoints and generated figures

## Repository Layout

```text
comet/
  comets/                  # Core package (models, loss, sampler, train loop, visualization)
  data_checkpoint/         # Cached datasets/checkpoints/results (.pkl/.csv)
  sfn_fig/                 # Generated figures and plotting notebook
  archive/                 # Historical notebooks and older package snapshots
  *.ipynb                  # Active experiment notebooks
  pyproject.toml           # Python package metadata (project name: comet)
  README.md                # This file
  ARCHITECTURE.md          # Technical architecture and flow
```

## Setup

### Requirements
- Python 3.9+
- PyTorch
- NumPy
- scikit-learn
- Matplotlib

Dependencies are declared in `pyproject.toml`.

### Install (editable)

```bash
pip install -e .
```

After install, import with:

```python
import comets
```

## Core API (Package: `comets`)

- Configs: `ModalityConfig`, `SamplingConfig`, `TrainingConfig`
- Encoders: `Conv1DEncoder`, `SkippedConvEncoder`, `CEBRASKipEncoder`, `Conv2DEncoder`, `MLPEncoder`, `GRUEncoder`
- Heads: `IdentityHead`, `LinearHead`, `OneLayerMLP`, `TwoLayerMLP`
- Training: `comets.train(A, modA, B, modB, sampling, train_cfg)`
- Outputs: trained `CoMETs` object with `embedA`, `embedB`, `embed_all`
- Viz: `comets.plotemb2(...)`

See `ARCHITECTURE.md` for internal training/loss flow.

## Typical Workflow

1. Prepare aligned modalities `A` and `B` with shape `[T, C]`.
2. Define per-modality encoder config and window length via `ModalityConfig`.
3. Define `SamplingConfig` (`batch_size`, positive offset, negative temporal mask distance).
4. Define `TrainingConfig` (epochs, LR, temperature, projection dim, loss balance).
5. Train with `comets.train(...)`.
6. Extract embeddings with `model.embedA(A)` and `model.embedB(B)`.
7. Evaluate/visualize in notebook (decoding, recovery metrics, PCA/UMAP/CEBRA baselines, etc.).

## Notebooks (Current, Top-Level)

### Synthetic demos
- `comet_torus_demo.ipynb`: Builds a synthetic torus latent manifold and two observed views, then trains CoMETs to recover aligned embeddings. It compares embedding geometry against baseline reductions/visualizations.
- `comet_coupling_demo.ipynb`: Simulates coupled oscillator (Kuramoto-style) dynamics as a two-view time-series problem. It trains CoMETs and evaluates whether cross-view structure is preserved in latent space.
- `comet_simu_demo.ipynb`: Generates synthetic spike/LFP-like multimodal data with known latent drivers. It runs CoMETs end-to-end and compares recovery quality to PCA/UMAP/related baselines.

### Real-data / evaluation workflows
- `comet_PTP_AGW_demo.ipynb`: Main PTP/AGW real-data demo notebook for preparing paired modalities, training CoMETs, and inspecting embedding quality. It is the lighter demonstration workflow before large sweeps.
- `comet_PTP_AGW_demoidk.ipynb`: Alternate/experimental variant of the PTP/AGW demo with modified settings and exploratory cells. Useful for quick prototyping but less canonical than the main demo.
- `PTP_AGW_results.ipynb`: Structured evaluation notebook that runs repeated CoMETs experiments and reports quantitative metrics, including shuffle/control analyses. Designed to produce summary result tables and plots.
- `PTP_AGW_results_alpha.ipynb`: Hyperparameter-focused extension of the results workflow, emphasizing alpha/loss-balance effects on performance. Used for sensitivity analysis and tuning behavior.
- `PTP_AGW_results_heavy.ipynb`: Large-scale/high-compute version of the results pipeline with broader sweeps or more repetitions. Intended for final/robust benchmarking rather than quick iteration.
- `lfp_spike_recovery.ipynb`: Real-data recovery pipeline for LFP-spike paired embeddings with downstream reconstruction/decoding checks. Includes CoMETs training and metric-driven validation.
- `lfp_spike_recovery_concat.ipynb`: Concatenated/aggregated variant of the LFP-spike recovery workflow for combined-session or merged-data analysis. Used to test stability under larger pooled datasets.
- `lfp_spike_visual.ipynb`: Visualization-first notebook for inspecting raw/processed LFP-spike signals and embedding trajectories. Primarily exploratory QA and figure generation.
- `torus_demo.ipynb`: Older torus workflow predating the newer `comet_torus_demo.ipynb` structure. Kept for reference/comparison with earlier experiment conventions.

## Data and Outputs

- `data_checkpoint/`: Stores reusable intermediate artifacts and result caches (`.pkl`, `.csv`) so expensive preprocessing/training does not need to be rerun (for example, `ec013.156_lfp_spike.pkl`, `comets_best_trajectories.pkl`, grid-search CSVs).
- `sfn_fig/`: Stores exported figures and plotting artifacts generated from result notebooks (PNG/SVG outputs and plotting helper notebooks).
- `archive/`: Historical notebooks and older package snapshots used for provenance and method evolution; not the primary active workflow.
## Notes

- This repo is notebook-driven; there is no unified CLI runner yet.
- `archive/` includes earlier package variants and exploratory notebooks; treat these as historical unless needed for comparison.

## Development Suggestions

- Add lightweight script entrypoints for repeatable runs (train/eval/export).
- Add minimal test coverage for sampler, windowing, and InfoNCE masking semantics.
- Standardize naming/docs between project name (`comet`) and import package (`comets`).

