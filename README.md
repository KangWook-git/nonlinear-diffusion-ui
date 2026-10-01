# Nonlinear Diffusion Interactive Model / Learning App — v2.3

A reproducible Jupyter-based research software prototype for exploring nonlinear diffusion in 1D, 2D, and 3D, with forward and inverse workflows, multiple discretization backends, configurable constitutive laws and boundary conditions, and run-record export/reload.

## Governing model

The core forward model is

$$
\frac{\partial u}{\partial t}
=\nabla\cdot\left(\mathbf D(u)\nabla u\right),
$$

with isotropic diffusion or axis-aligned diagonal anisotropy,

$$
\mathbf{D}(u)=\mathrm{diag}\left(D_x(u),D_y(u),D_z(u)\right).
$$

The software is organized around a learning-and-modeling workflow:

**Choose → Simulate → Observe → Question → Ask AI**

## What v2.3 includes

### Forward modeling

- 1D / 2D / 3D simulation.
- Isotropic nonlinear diffusion with FDM, FVM, or FEM backends.
- Axis-aligned diagonal anisotropic diffusion with FVM.
- Dirichlet, Neumann, and Robin boundary conditions.
- Configurable initial conditions and diffusivity laws.
- Picard / Newton / hybrid nonlinear solution paths where supported by the backend.
- Plotting and numerical summaries inside the notebook UI.

### Inverse modeling

- 1D / 2D FVM inverse workflow based on domain-mean observations.
- Explicit **Fixed** versus **Estimate** parameter selection.
- Initial guesses, bounds, and optional log-space optimization.
- Observed / predicted / residual tables.
- Local sensitivity and practical-identifiability diagnostics, including scaled sensitivity curves, singular values, effective rank, and sensitivity-column similarity.
- Built-in synthetic round-trip workflow for separating numerical inverse failure from experimental-model mismatch.

### Reproducibility record layer

Each completed run can be saved to a run folder. Depending on mode and options, the record contains:

- `input.json`
- `input.csv`
- `summary.csv`
- `timeseries.csv`
- `report.md`
- `field.npz` (optional compressed field data)
- `parameters.csv` (inverse)
- `diagnostics.json` (inverse)
- `sensitivity.csv` (inverse)

`Load Run` restores the saved inputs into the UI. The calculation should then be executed again in the current environment to reproduce the result.

## Repository layout

```text
.
├── Nonlinear_Diffusion_UI_v2_3.ipynb   # Main interactive notebook
├── fdm_backend.py                       # Isotropic finite-difference backend
├── fvm_backend.py                       # Isotropic finite-volume backend
├── fem_backend.py                       # Isotropic finite-element backend
├── anisotropic_fvm_backend.py           # Diagonal-anisotropic FVM backend
├── requirements.txt                     # Minimal pip dependencies
├── environment.yml                      # Optional conda environment
├── tests/
│   └── smoke_test.py                    # Lightweight backend consistency check
├── CITATION.cff                         # Machine-readable citation metadata
├── CHANGELOG.md                         # Release history
├── AI_DISCLOSURE.md                     # Development transparency note
├── PUBLISHING_GUIDE_KO.md               # GitHub → Zenodo first-publication guide
├── LICENSE                              # MIT license
├── VERSION                              # Current release version
└── .gitignore                           # Local/runtime files excluded from Git
```

## Quick start

Run the notebook **from this repository folder** so that the backend modules are importable.

### Option A — pip / venv

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open `Nonlinear_Diffusion_UI_v2_3.ipynb` and run the cells from top to bottom.

### Option B — conda

```bash
conda env create -f environment.yml
conda activate nonlinear-diffusion-ui
jupyter lab
```

## Lightweight backend test

From the repository root:

```bash
python tests/smoke_test.py
```

The smoke test checks that all four numerical backends can execute a small 1D zero-flux problem and preserve a constant field to numerical precision. It is not a full verification suite.

## Interpretation and limitations

- Current anisotropy is **axis-aligned diagonal anisotropy**. Rotated material axes require a later tensor form such as $\mathbf D=\mathbf Q\mathbf D_{\mathrm{mat}}\mathbf Q^T$.
- The inverse workflow currently uses 1D / 2D FVM and domain-mean observations. A successful least-squares termination is not proof of model validity, uniqueness, or global identifiability.
- Local singular-value/rank and sensitivity diagnostics are practical diagnostics, not structural-identifiability proofs.
- Current Robin coefficients are constant. State-, time-, or correlation-dependent boundary laws are future extensions.
- 3D inverse fitting is deliberately excluded because repeated nonlinear 3D solves are computationally expensive and require stronger identifiability evidence.
- Saved run records are reproducibility snapshots. Re-running a saved input under a different Python/package environment can produce differences.

## Versioning

This repository uses release tags such as `v2.3`, `v2.4`, and `v3.0`. A GitHub release can be archived automatically by Zenodo, which then assigns a persistent DOI to that specific software release.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). GitHub can render this file directly through **Cite this repository**. After the first Zenodo archive is created, use the Zenodo DOI when citing the released software version.

Suggested human-readable form before the DOI is assigned:

> Kang, W. (2026). *Nonlinear Diffusion Interactive Model / Learning App* (Version 2.3) [Computer software].

## License

This release is distributed under the [MIT License](LICENSE).

## Development transparency

The project was developed with substantial AI-assisted coding, refactoring, documentation, and debugging under the author's direction and review. See [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md) for the disclosure used in this public release.

## Author

**Wook Kang**

---

For a first GitHub + Zenodo publication, follow [`PUBLISHING_GUIDE_KO.md`](PUBLISHING_GUIDE_KO.md).
