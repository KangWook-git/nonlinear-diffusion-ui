# Changelog

All notable public-release changes are documented here.

## v2.3 — 2026-10-01

First archival-ready public baseline.

### Forward
- 1D / 2D / 3D forward simulation.
- Isotropic FDM / FVM / FEM backends retained.
- Axis-aligned diagonal-anisotropic FVM in 1D / 2D / 3D.
- Configurable constitutive, initial-condition, and boundary-condition layers.

### Inverse
- 1D / 2D FVM mean-response fitting.
- Fixed/estimated parameter selection, bounds, initial guesses, and optional log-space optimization.
- Synthetic round-trip verification workflow.
- Residual, bound-proximity, local sensitivity, singular-value, collinearity, and effective-rank diagnostics.

### Reproducibility / UI
- Input state frozen at run start.
- Separate input summary, result summary, and time-series/residual tables.
- Forward and inverse Save Run / Load Run.
- Optional auto-save.
- Reproducibility records in JSON/CSV/Markdown with optional compressed NPZ fields.

### Public-release packaging
- Expanded README and quick-start instructions.
- Added `CITATION.cff`, MIT `LICENSE`, `.gitignore`, conda environment file, smoke test, publishing guide, changelog, version file, and AI-assistance disclosure.
- Renamed the main notebook to `Nonlinear_Diffusion_UI_v2_3.ipynb` for a shorter public filename.
- Updated stale backend comments referring to UI v1; numerical algorithms were not altered by that cleanup.
