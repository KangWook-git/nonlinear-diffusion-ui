"""Diagonal-anisotropic nonlinear diffusion on Cartesian finite volumes.

PDE
---
    du/dt = div(D(u) grad u)

with an axis-aligned diagonal tensor
    D(u) = diag(D_x(u), D_y(u), D_z(u)).

Scope
-----
- 1D / 2D / 3D forward simulation
- backward Euler in time
- Picard linearization
- cell-centered uniform Cartesian finite volumes
- Dirichlet / Neumann / Robin boundaries
- exact stepping to requested observation times for inverse problems

Boundary sign convention
------------------------
q = -D grad u, and Neumann ``flux`` means outward q.n.
Robin: q.n = h (u_s - u_env), including the half-cell diffusion
resistance between the cell center and the boundary surface.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

_FACE_LABELS = (("x-", "x+"), ("y-", "y+"), ("z-", "z+"))


def geometry(shape, lengths):
    shape = tuple(int(n) for n in shape)
    lengths = tuple(float(L) for L in lengths)
    dim = len(shape)
    if dim not in (1, 2, 3):
        raise ValueError("Only 1D, 2D, and 3D are supported.")
    if len(lengths) != dim:
        raise ValueError("shape and lengths must have the same dimension.")
    if any(n < 2 for n in shape):
        raise ValueError("Each active mesh count must be at least 2.")
    if any(L <= 0 for L in lengths):
        raise ValueError("All active lengths must be positive.")
    spacing = tuple(L / n for L, n in zip(lengths, shape))
    volume = float(np.prod(spacing))
    face_areas = tuple(volume / h for h in spacing)
    axes = tuple((np.arange(n) + 0.5) * h for n, h in zip(shape, spacing))
    return dict(dim=dim, shape=shape, lengths=lengths, spacing=spacing,
                volume=volume, face_areas=face_areas, axes=axes)


def flat_id(idx, shape):
    return np.ravel_multi_index(idx, shape, order="C")


def _bc_key(axis, side):
    return _FACE_LABELS[axis][side]


def normalize_bcs(dim, bcs):
    """Accept either face-label keys ('x-') or tuple keys (axis, side)."""
    out = {}
    for axis in range(dim):
        for side in (0, 1):
            label = _bc_key(axis, side)
            if label in bcs:
                bc = bcs[label]
            elif (axis, side) in bcs:
                bc = bcs[(axis, side)]
            else:
                raise ValueError(f"Missing boundary condition for {label}")
            out[label] = dict(bc)
    return out


def robin_H(Dp, h, half_distance):
    """Effective transfer coefficient after surface + half-cell resistance."""
    h = float(h)
    Dp = float(Dp)
    if h < 0:
        raise ValueError("Robin h must be non-negative.")
    if h == 0.0:
        return 0.0
    if np.isinf(h):
        return Dp / half_distance
    if Dp <= 1e-300:
        return 0.0
    return 1.0 / (1.0 / h + half_distance / Dp)


def _eval_axis_D(U, D_axis_funcs):
    U = np.asarray(U, dtype=float)
    vals = []
    for axis, f in enumerate(D_axis_funcs):
        d = np.asarray(f(U), dtype=float)
        if d.shape != U.shape:
            d = np.broadcast_to(d, U.shape).copy()
        if not np.all(np.isfinite(d)):
            raise ValueError(f"D_{'xyz'[axis]} contains NaN or infinity.")
        if np.any(d <= 0):
            raise ValueError(
                f"D_{'xyz'[axis]} must remain positive; min={np.min(d):.6g}"
            )
        vals.append(d)
    return vals


def assemble_picard(U_iter, U_old, shape, lengths, dt, bcs, D_axis_funcs):
    g = geometry(shape, lengths)
    dim = g["dim"]
    shape = g["shape"]
    spacing = g["spacing"]
    V = g["volume"]
    areas = g["face_areas"]
    bcs = normalize_bcs(dim, bcs)

    U_iter = np.asarray(U_iter, dtype=float).reshape(shape)
    U_old = np.asarray(U_old, dtype=float).reshape(shape)
    if len(D_axis_funcs) != dim:
        raise ValueError("Need one diffusivity function per active spatial axis.")
    Daxis = _eval_axis_D(U_iter, D_axis_funcs)

    Ntot = int(np.prod(shape))
    A = lil_matrix((Ntot, Ntot), dtype=float)
    b = np.zeros(Ntot, dtype=float)
    storage = V / float(dt)

    # Storage
    for idx in np.ndindex(shape):
        P = flat_id(idx, shape)
        A[P, P] += storage
        b[P] += storage * U_old[idx]

    # Internal faces: arithmetic mean of axis-specific diffusivity.
    for idx in np.ndindex(shape):
        P = flat_id(idx, shape)
        for axis in range(dim):
            if idx[axis] >= shape[axis] - 1:
                continue
            nbr = list(idx)
            nbr[axis] += 1
            nbr = tuple(nbr)
            N = flat_id(nbr, shape)
            Df = 0.5 * (Daxis[axis][idx] + Daxis[axis][nbr])
            G = areas[axis] * Df / spacing[axis]
            A[P, P] += G
            A[P, N] -= G
            A[N, N] += G
            A[N, P] -= G

    # Boundary faces
    for idx in np.ndindex(shape):
        P = flat_id(idx, shape)
        for axis in range(dim):
            dx = spacing[axis]
            Af = areas[axis]
            for side in (0, 1):
                if idx[axis] != (0 if side == 0 else shape[axis] - 1):
                    continue
                bc = bcs[_bc_key(axis, side)]
                typ = str(bc["type"]).lower()
                if typ == "dirichlet":
                    uD = float(bc["value"])
                    DD = float(np.asarray(D_axis_funcs[axis](np.array(uD))))
                    if DD <= 0 or not np.isfinite(DD):
                        raise ValueError("Boundary diffusivity must be positive and finite.")
                    Db = 0.5 * (Daxis[axis][idx] + DD)
                    G = 2.0 * Af * Db / dx
                    A[P, P] += G
                    b[P] += G * uD
                elif typ == "neumann":
                    # outward flux q.n is positive out of the domain
                    qn = float(bc.get("flux", bc.get("value", 0.0)))
                    b[P] -= Af * qn
                elif typ == "robin":
                    h = float(bc["h"])
                    env = float(bc["env"])
                    H = robin_H(Daxis[axis][idx], h, 0.5 * dx)
                    G = Af * H
                    A[P, P] += G
                    b[P] += G * env
                else:
                    raise ValueError(f"Unknown BC type: {typ}")

    return A.tocsr(), b


def picard_step(U_old, shape, lengths, dt, bcs, D_axis_funcs,
                tol=1e-9, max_iter=60, relaxation=1.0):
    U_old = np.asarray(U_old, dtype=float).reshape(shape)
    U = U_old.copy()
    relaxation = float(relaxation)
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must satisfy 0 < relaxation <= 1.")

    history = []
    for it in range(1, int(max_iter) + 1):
        A, b = assemble_picard(U, U_old, shape, lengths, dt, bcs, D_axis_funcs)
        U_lin = np.asarray(spsolve(A, b), dtype=float).reshape(shape)
        if not np.all(np.isfinite(U_lin)):
            raise RuntimeError("Linear solve produced NaN or infinity.")
        U_new = relaxation * U_lin + (1.0 - relaxation) * U
        err = float(np.max(np.abs(U_new - U)))
        history.append(err)
        U = U_new
        if err <= tol:
            return U, history
    raise RuntimeError(f"Picard iteration did not converge within {max_iter} iterations.")


def run_time(U0, shape, lengths, dt, nsteps, bcs, D_axis_funcs,
             picard_tol=1e-9, picard_max_iter=60, relaxation=1.0):
    g = geometry(shape, lengths)
    U = np.asarray(U0, dtype=float).reshape(g["shape"]).copy()
    if not np.all(np.isfinite(U)):
        raise ValueError("Initial field contains NaN or infinity.")
    snapshots = [U.copy()]
    iteration_counts = []
    for _ in range(int(nsteps)):
        U, hist = picard_step(
            U, g["shape"], g["lengths"], float(dt), bcs, D_axis_funcs,
            tol=picard_tol, max_iter=picard_max_iter, relaxation=relaxation,
        )
        snapshots.append(U.copy())
        iteration_counts.append(len(hist))
    return {
        "dim": g["dim"], "shape": g["shape"], "lengths": g["lengths"],
        "spacing": g["spacing"], "axes": g["axes"],
        "snapshots": snapshots, "final": snapshots[-1],
        "iteration_counts": np.asarray(iteration_counts, dtype=int),
        "times": np.arange(int(nsteps) + 1, dtype=float) * float(dt),
        "mean_series": np.asarray([np.mean(s) for s in snapshots], dtype=float),
    }


def solve_to_times(shape, lengths, observation_times, u0, bcs, D_axis_funcs,
                   dt_max, picard_tol=1e-9, picard_max_iter=60,
                   relaxation=1.0, store_fields=False):
    """Integrate from t=0 and report volume means exactly at requested times."""
    g = geometry(shape, lengths)
    times = np.asarray(observation_times, dtype=float).ravel()
    if times.size == 0:
        raise ValueError("At least one observation time is required.")
    if np.any(~np.isfinite(times)) or np.any(times < 0):
        raise ValueError("Observation times must be finite and non-negative.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Observation times must be strictly increasing.")
    if dt_max <= 0:
        raise ValueError("dt_max must be positive.")

    U = np.asarray(u0, dtype=float).reshape(g["shape"]).copy()
    t = 0.0
    means, counts = [], []
    fields = [] if store_fields else None

    for target in times:
        while t < target - max(1e-14, 1e-12 * max(1.0, target)):
            dt = min(float(dt_max), target - t)
            U, hist = picard_step(
                U, g["shape"], g["lengths"], dt, bcs, D_axis_funcs,
                tol=picard_tol, max_iter=picard_max_iter, relaxation=relaxation,
            )
            t += dt
            counts.append(len(hist))
        means.append(float(np.mean(U)))
        if store_fields:
            fields.append(U.copy())

    return {
        "times": times, "mean_series": np.asarray(means), "final": U.copy(),
        "fields": fields, "iteration_counts": np.asarray(counts, dtype=int),
        "axes": g["axes"], "shape": g["shape"], "lengths": g["lengths"],
        "spacing": g["spacing"], "dim": g["dim"],
    }


def flux_components(U, lengths, D_axis_funcs):
    """Cell-centered q_i = -D_i(u) du/dx_i for diagnostics."""
    U = np.asarray(U, dtype=float)
    shape = U.shape
    g = geometry(shape, lengths)
    Daxis = _eval_axis_D(U, D_axis_funcs)
    q = []
    grad = []
    for axis, dx in enumerate(g["spacing"]):
        edge_order = 2 if shape[axis] >= 3 else 1
        gi = np.gradient(U, dx, axis=axis, edge_order=edge_order)
        grad.append(gi)
        q.append(-Daxis[axis] * gi)
    return tuple(grad), tuple(q)
