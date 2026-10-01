# Numerical core retained from the verified Technical Note backend.
# Public UI release v2.3 keeps this backend algorithm unchanged.

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def D_exp(u, D0=1e-10, beta=5.0, u_ref=0.2):
    u = np.asarray(u, dtype=float)
    return D0 * np.exp(beta * (u - u_ref))

def dD_exp(u, D0=1e-10, beta=5.0, u_ref=0.2):
    return beta * D_exp(u, D0=D0, beta=beta, u_ref=u_ref)

def eval_D(u, D0, beta, u_ref, D_func=None, D_kwargs=None):
    if D_func is None:
        return D_exp(u, D0, beta, u_ref)
    if D_kwargs is None:
        D_kwargs = {}
    return D_func(u, **D_kwargs)

def eval_dD(u, D0, beta, u_ref, dD_func=None, D_kwargs=None):
    if dD_func is None:
        return dD_exp(u, D0, beta, u_ref)
    if D_kwargs is None:
        D_kwargs = {}
    return dD_func(u, **D_kwargs)

def fvm_geometry(shape, lengths):
    shape = tuple((int(n) for n in shape))
    lengths = tuple((float(L) for L in lengths))
    if len(shape) not in (1, 2, 3):
        raise ValueError('Only 1D, 2D, and 3D are supported.')
    if len(shape) != len(lengths):
        raise ValueError('shape and lengths must have same dimension.')
    spacing = tuple((L / n for L, n in zip(lengths, shape)))
    volume = float(np.prod(spacing))
    face_areas = tuple((volume / h for h in spacing))
    axes = tuple(((np.arange(n) + 0.5) * h for n, h in zip(shape, spacing)))
    return {'dim': len(shape), 'shape': shape, 'lengths': lengths, 'spacing': spacing, 'volume': volume, 'face_areas': face_areas, 'axes': axes}

def flat_id(idx, shape):
    return np.ravel_multi_index(idx, shape, order='C')

def check_bcs(shape, bcs):
    dim = len(shape)
    required = {(axis, side) for axis in range(dim) for side in (0, 1)}
    missing = required - set(bcs.keys())
    if missing:
        raise ValueError(f'Missing BCs: {sorted(missing)}')

def BC_D(value):
    return {'type': 'dirichlet', 'value': float(value)}

def BC_N(flux=0.0):
    return {'type': 'neumann', 'value': float(flux)}

def BC_R(h, env):
    h = float(h)
    env = float(env)
    if h == 0.0:
        return BC_N(0.0)
    if np.isinf(h):
        return BC_D(env)
    if h < 0.0:
        raise ValueError('Robin h must be non-negative.')
    return {'type': 'robin', 'h': h, 'env': env}

def all_faces_bc(dim, bc):
    return {(axis, side): dict(bc) for axis in range(dim) for side in (0, 1)}

def robin_H(Dp, h, half_distance):
    if h == 0.0:
        return 0.0
    if not np.isfinite(h):
        return Dp / half_distance
    if Dp <= 1e-300:
        return 0.0
    return 1.0 / (1.0 / h + half_distance / Dp)

def assemble_picard_fvm(U_iter, U_old, shape, lengths, dt, bcs, *, D0=1e-10, beta=5.0, u_ref=0.2, D_func=None, dD_func=None, D_kwargs=None):
    geom = fvm_geometry(shape, lengths)
    spacing = geom['spacing']
    V = geom['volume']
    areas = geom['face_areas']
    check_bcs(shape, bcs)
    U_iter = np.asarray(U_iter, dtype=float).reshape(shape)
    U_old = np.asarray(U_old, dtype=float).reshape(shape)
    Ntot = int(np.prod(shape))
    A = lil_matrix((Ntot, Ntot), dtype=float)
    b = np.zeros(Ntot)
    Dcell = eval_D(U_iter, D0, beta, u_ref, D_func=D_func, D_kwargs=D_kwargs)
    storage = V / dt
    for idx in np.ndindex(shape):
        P = flat_id(idx, shape)
        A[P, P] += storage
        b[P] += storage * U_old[idx]
    for idx in np.ndindex(shape):
        P = flat_id(idx, shape)
        for axis in range(len(shape)):
            if idx[axis] < shape[axis] - 1:
                nbr = list(idx)
                nbr[axis] += 1
                nbr = tuple(nbr)
                N = flat_id(nbr, shape)
                Df = 0.5 * (Dcell[idx] + Dcell[nbr])
                G = areas[axis] / spacing[axis] * Df
                A[P, P] += G
                A[P, N] -= G
                A[N, N] += G
                A[N, P] -= G
    for idx in np.ndindex(shape):
        P = flat_id(idx, shape)
        for axis in range(len(shape)):
            dx = spacing[axis]
            Af = areas[axis]
            for side in (0, 1):
                on_face = idx[axis] == 0 if side == 0 else idx[axis] == shape[axis] - 1
                if not on_face:
                    continue
                bc = bcs[axis, side]
                typ = bc['type'].lower()
                if typ == 'dirichlet':
                    uD = bc['value']
                    DD = float(eval_D(uD, D0, beta, u_ref, D_func=D_func, D_kwargs=D_kwargs))
                    Db = 0.5 * (Dcell[idx] + DD)
                    G = 2.0 * Af / dx * Db
                    A[P, P] += G
                    b[P] += G * uD
                elif typ == 'neumann':
                    qn = bc.get('value', 0.0)
                    b[P] -= Af * qn
                elif typ == 'robin':
                    h = bc['h']
                    env = bc['env']
                    H = robin_H(float(Dcell[idx]), h, 0.5 * dx)
                    G = Af * H
                    A[P, P] += G
                    b[P] += G * env
                else:
                    raise ValueError(f'Unknown BC type: {typ}')
    return (A.tocsr(), b)

def residual_and_jacobian_fvm(U, U_old, shape, lengths, dt, bcs, *, D0=1e-10, beta=5.0, u_ref=0.2, D_func=None, dD_func=None, D_kwargs=None, build_jacobian=True):
    geom = fvm_geometry(shape, lengths)
    spacing = geom['spacing']
    V = geom['volume']
    areas = geom['face_areas']
    check_bcs(shape, bcs)
    U = np.asarray(U, dtype=float).reshape(shape)
    U_old = np.asarray(U_old, dtype=float).reshape(shape)
    Ntot = int(np.prod(shape))
    R = V / dt * (U.ravel(order='C') - U_old.ravel(order='C'))
    J = lil_matrix((Ntot, Ntot), dtype=float) if build_jacobian else None
    if build_jacobian:
        J.setdiag(np.full(Ntot, V / dt))
    Dcell = eval_D(U, D0, beta, u_ref, D_func=D_func, D_kwargs=D_kwargs)
    dDcell = eval_dD(U, D0, beta, u_ref, dD_func=dD_func, D_kwargs=D_kwargs)
    for idx in np.ndindex(shape):
        P = flat_id(idx, shape)
        for axis in range(len(shape)):
            if idx[axis] >= shape[axis] - 1:
                continue
            nbr = list(idx)
            nbr[axis] += 1
            nbr = tuple(nbr)
            N = flat_id(nbr, shape)
            T = areas[axis] / spacing[axis]
            Df = 0.5 * (Dcell[idx] + Dcell[nbr])
            du = U[idx] - U[nbr]
            Q = T * Df * du
            R[P] += Q
            R[N] -= Q
            if build_jacobian:
                dQdP = T * (Df + 0.5 * dDcell[idx] * du)
                dQdN = T * (-Df + 0.5 * dDcell[nbr] * du)
                J[P, P] += dQdP
                J[P, N] += dQdN
                J[N, P] -= dQdP
                J[N, N] -= dQdN
    for idx in np.ndindex(shape):
        P = flat_id(idx, shape)
        for axis in range(len(shape)):
            dx = spacing[axis]
            Af = areas[axis]
            for side in (0, 1):
                on_face = idx[axis] == 0 if side == 0 else idx[axis] == shape[axis] - 1
                if not on_face:
                    continue
                bc = bcs[axis, side]
                typ = bc['type'].lower()
                if typ == 'dirichlet':
                    uD = bc['value']
                    DD = float(eval_D(uD, D0, beta, u_ref, D_func=D_func, D_kwargs=D_kwargs))
                    Db = 0.5 * (Dcell[idx] + DD)
                    du = U[idx] - uD
                    T = 2.0 * Af / dx
                    Q = T * Db * du
                    R[P] += Q
                    if build_jacobian:
                        J[P, P] += T * (Db + 0.5 * dDcell[idx] * du)
                elif typ == 'neumann':
                    qn = bc.get('value', 0.0)
                    R[P] += Af * qn
                elif typ == 'robin':
                    h = bc['h']
                    env = bc['env']
                    Dp = float(Dcell[idx])
                    dDp = float(dDcell[idx])
                    H = robin_H(Dp, h, 0.5 * dx)
                    du = U[idx] - env
                    Q = Af * H * du
                    R[P] += Q
                    if build_jacobian:
                        if h == 0.0 or Dp <= 1e-300:
                            Hp = 0.0
                        elif np.isinf(h):
                            Hp = dDp / (0.5 * dx)
                        else:
                            Hp = H ** 2 * (0.5 * dx) * dDp / Dp ** 2
                        J[P, P] += Af * (H + Hp * du)
                else:
                    raise ValueError(f'Unknown BC type: {typ}')
    return (R, J.tocsr() if build_jacobian else None)

def picard_step_fvm(U_old, shape, lengths, dt, bcs, *, tol=1e-10, maxiter=50, omega=1.0, **constitutive):
    U_old = np.asarray(U_old, dtype=float).reshape(shape)
    U = U_old.copy()
    for k in range(1, maxiter + 1):
        A, b = assemble_picard_fvm(U, U_old, shape, lengths, dt, bcs, **constitutive)
        Ulin = spsolve(A, b).reshape(shape)
        Unew = U + omega * (Ulin - U)
        err = np.max(np.abs(Unew - U)) / (1.0 + np.max(np.abs(Unew)))
        U = Unew
        if err < tol:
            return (U, k)
    raise RuntimeError('Picard iteration did not converge.')

def newton_step_fvm(U_old, shape, lengths, dt, bcs, *, U_init=None, tol=1e-09, maxiter=30, **constitutive):
    U_old = np.asarray(U_old, dtype=float).reshape(shape)
    U = U_old.copy() if U_init is None else np.asarray(U_init, dtype=float).reshape(shape).copy()
    R0, _ = residual_and_jacobian_fvm(U, U_old, shape, lengths, dt, bcs, build_jacobian=False, **constitutive)
    scale = max(np.linalg.norm(R0, np.inf), 1e-30)
    for k in range(1, maxiter + 1):
        R, J = residual_and_jacobian_fvm(U, U_old, shape, lengths, dt, bcs, build_jacobian=True, **constitutive)
        rnorm = np.linalg.norm(R, np.inf)
        if rnorm / scale < tol:
            return (U, k)
        dU = spsolve(J, -R).reshape(shape)
        lam = 1.0
        accepted = False
        while lam >= 1e-06:
            Utrial = U + lam * dU
            Rtrial, _ = residual_and_jacobian_fvm(Utrial, U_old, shape, lengths, dt, bcs, build_jacobian=False, **constitutive)
            if np.linalg.norm(Rtrial, np.inf) < rnorm:
                U = Utrial
                accepted = True
                break
            lam *= 0.5
        if not accepted:
            raise RuntimeError('Newton line search failed.')
    raise RuntimeError('Newton iteration did not converge.')

def hybrid_step_fvm(U_old, shape, lengths, dt, bcs, *, picard_sweeps=2, **constitutive):
    U_old = np.asarray(U_old, dtype=float).reshape(shape)
    U = U_old.copy()
    for _ in range(picard_sweeps):
        A, b = assemble_picard_fvm(U, U_old, shape, lengths, dt, bcs, **constitutive)
        U = spsolve(A, b).reshape(shape)
    Un, kN = newton_step_fvm(U_old, shape, lengths, dt, bcs, U_init=U, **constitutive)
    return (Un, picard_sweeps + kN)

def run_time_fvm(U0, shape, lengths, dt, nsteps, bcs, *, method='picard', **constitutive):
    U = np.asarray(U0, dtype=float).reshape(shape).copy()
    snapshots = [U.copy()]
    counts = []
    for _ in range(nsteps):
        if method == 'picard':
            U, k = picard_step_fvm(U, shape, lengths, dt, bcs, **constitutive)
        elif method == 'newton':
            U, k = newton_step_fvm(U, shape, lengths, dt, bcs, **constitutive)
        elif method == 'hybrid':
            U, k = hybrid_step_fvm(U, shape, lengths, dt, bcs, **constitutive)
        else:
            raise ValueError('method must be picard, newton, or hybrid')
        snapshots.append(U.copy())
        counts.append(k)
    return (np.array(snapshots), np.array(counts))
_FACE_TO_KEY = {'x-': (0, 0), 'x+': (0, 1), 'y-': (1, 0), 'y+': (1, 1), 'z-': (2, 0), 'z+': (2, 1)}

def make_box_bcs(dim, default=None, overrides=None):
    if dim not in (1, 2, 3):
        raise ValueError('dim must be 1, 2, or 3.')
    if default is None:
        default = BC_N(0.0)
    labels = ['x-', 'x+'] + (['y-', 'y+'] if dim >= 2 else []) + (['z-', 'z+'] if dim >= 3 else [])
    bcs = {_FACE_TO_KEY[label]: dict(default) for label in labels}
    if overrides:
        for label, bc in overrides.items():
            if label not in labels:
                raise ValueError(f'{label} is not active for dim={dim}')
            bcs[_FACE_TO_KEY[label]] = dict(bc)
    return bcs

def solve_fvm_box_nd(shape, lengths, bcs, *, u0=0.3, dt=1800.0, nsteps=6, method='picard', D0=1e-10, beta=5.0, u_ref=0.2, D_func=None, D_kwargs=None):
    shape = tuple((int(n) for n in shape))
    lengths = tuple((float(L) for L in lengths))
    geom = fvm_geometry(shape, lengths)
    check_bcs(shape, bcs)
    if np.isscalar(u0):
        U0 = np.full(shape, float(u0))
    else:
        U0 = np.asarray(u0, dtype=float).reshape(shape)
    snaps, counts = run_time_fvm(U0, shape, lengths, dt, nsteps, bcs, method=method, D0=D0, beta=beta, u_ref=u_ref, D_func=D_func, D_kwargs=D_kwargs)
    return {'dim': len(shape), 'geometry': geom, 'bcs': bcs, 'snapshots': snaps, 'final': snaps[-1], 'iteration_counts': counts}
