"""Lightweight backend smoke test for Nonlinear Diffusion UI v2.3.

This is intentionally not a full verification suite. It checks that each backend
can execute a small 1D zero-flux problem and preserve a constant field.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fdm_backend as FDM
import fvm_backend as FVM
import fem_backend as FEM
import anisotropic_fvm_backend as AFVM


TOL = 1.0e-12
U0 = 0.3


def _assert_constant(name, arr):
    arr = np.asarray(arr, dtype=float)
    err = float(np.max(np.abs(arr - U0)))
    if not np.isfinite(err) or err > TOL:
        raise AssertionError(f"{name}: constant-field error = {err:.3e}")
    print(f"PASS {name:17s} max|u-u0| = {err:.3e}")


def main():
    shape = (5,)
    lengths = (0.01,)

    fdm_bcs = {(0, 0): FDM.BC_N(0.0), (0, 1): FDM.BC_N(0.0)}
    out = FDM.solve_fdm_box_nd(shape, lengths, fdm_bcs, u0=U0, dt=1.0, nsteps=2)
    _assert_constant("FDM", out["final"])

    fvm_bcs = {(0, 0): FVM.BC_N(0.0), (0, 1): FVM.BC_N(0.0)}
    out = FVM.solve_fvm_box_nd(shape, lengths, fvm_bcs, u0=U0, dt=1.0, nsteps=2)
    _assert_constant("FVM", out["final"])

    fem_bcs = {(0, 0): FEM.FEM_BC_N(0.0), (0, 1): FEM.FEM_BC_N(0.0)}
    out = FEM.solve_fem_box_nd(1, lengths, (5,), fem_bcs, u0=U0, dt=1.0, nsteps=2)
    _assert_constant("FEM", out["final"])

    afvm_bcs = {
        "x-": {"type": "neumann", "value": 0.0},
        "x+": {"type": "neumann", "value": 0.0},
    }
    D_axis = [lambda u: np.full_like(np.asarray(u, dtype=float), 1.0e-10)]
    out = AFVM.run_time(
        np.full(shape, U0), shape, lengths, 1.0, 2, afvm_bcs, D_axis
    )
    _assert_constant("anisotropic FVM", out["final"])

    print("All backend smoke tests passed.")


if __name__ == "__main__":
    main()
