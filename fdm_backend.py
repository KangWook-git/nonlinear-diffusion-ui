# Numerical core retained from the verified Technical Note backend.
# Public UI release v2.3 keeps this backend algorithm unchanged.

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def D_nl(u, D0=1e-10, beta=5.0, u_ref=0.2):
    return D0 * np.exp(beta * (np.asarray(u) - u_ref))

def dD_nl(u, D0=1e-10, beta=5.0, u_ref=0.2):
    return beta * D_nl(u, D0, beta, u_ref)

def _eval_D_model(u, D0, beta, u_ref, D_func=None, D_kwargs=None):
    """Evaluate either the default exponential law or an injected constitutive law."""
    if D_func is None:
        return D_nl(u, D0, beta, u_ref)
    return np.asarray(D_func(u, **{} if D_kwargs is None else D_kwargs), dtype=float)

def _eval_dD_model(u, D0, beta, u_ref, dD_func=None, D_kwargs=None):
    """Derivative counterpart used by Newton/Jacobian assembly."""
    if dD_func is None:
        return dD_nl(u, D0, beta, u_ref)
    return np.asarray(dD_func(u, **{} if D_kwargs is None else D_kwargs), dtype=float)

def flat_index(idx, shape):
    return np.ravel_multi_index(idx, shape, order='C')

def robin_effective_H(u, grid_h, h_c, D0, beta, u_ref, D_func=None, dD_func=None, D_kwargs=None):
    """
    q = H(u_P) * (u_P - u_env)
    with half-cell diffusion resistance + surface resistance.
    """
    Dp = float(_eval_D_model(u, D0, beta, u_ref, D_func, D_kwargs))
    dDp = float(_eval_dD_model(u, D0, beta, u_ref, dD_func, D_kwargs))
    H = 1.0 / (1.0 / h_c + grid_h / (2.0 * Dp))
    Hp = H ** 2 * grid_h * dDp / (2.0 * Dp ** 2)
    return (H, Hp)

def check_bcs(shape, bcs):
    dim = len(shape)
    for axis in range(dim):
        for side in (0, 1):
            if (axis, side) not in bcs:
                raise ValueError(f'Missing BC at axis={axis}, side={side}')

def assemble_picard_system(u_coeff, u_old, shape, spacing, dt, bcs, D0=1e-10, beta=5.0, u_ref=0.2, D_func=None, dD_func=None, D_kwargs=None):
    """
    Freeze nonlinear coefficients at u_coeff.
    Solve A(u_coeff) u_new = b.
    """
    check_bcs(shape, bcs)
    Uc = np.asarray(u_coeff, dtype=float).reshape(shape)
    Uold = np.asarray(u_old, dtype=float).reshape(shape)
    N = int(np.prod(shape))
    A = lil_matrix((N, N))
    A.setdiag(np.ones(N))
    b = Uold.ravel(order='C').copy()
    Dnodes = _eval_D_model(Uc, D0, beta, u_ref, D_func, D_kwargs)
    for idx in np.ndindex(*shape):
        p = flat_index(idx, shape)
        for axis, hgrid in enumerate(spacing):
            if idx[axis] < shape[axis] - 1:
                nb_idx = list(idx)
                nb_idx[axis] += 1
                nb_idx = tuple(nb_idx)
                q = flat_index(nb_idx, shape)
                Df = 0.5 * (Dnodes[idx] + Dnodes[nb_idx])
                af = dt * Df / hgrid ** 2
                A[p, p] += af
                A[p, q] -= af
                A[q, q] += af
                A[q, p] -= af
            for side, on_boundary in ((0, idx[axis] == 0), (1, idx[axis] == shape[axis] - 1)):
                if not on_boundary:
                    continue
                bc = bcs[axis, side]
                bc_type = bc['type'].lower()
                if bc_type == 'dirichlet':
                    uD = float(bc['value'])
                    Db = 0.5 * (Dnodes[idx] + float(_eval_D_model(uD, D0, beta, u_ref, D_func, D_kwargs)))
                    aD = 2.0 * dt * Db / hgrid ** 2
                    A[p, p] += aD
                    b[p] += aD * uD
                elif bc_type == 'neumann':
                    qn = float(bc['value'])
                    b[p] -= dt * qn / hgrid
                elif bc_type == 'robin':
                    h_c = float(bc['h'])
                    u_env = float(bc['env'])
                    H, _ = robin_effective_H(Uc[idx], hgrid, h_c, D0, beta, u_ref, D_func, dD_func, D_kwargs)
                    aR = dt * H / hgrid
                    A[p, p] += aR
                    b[p] += aR * u_env
                else:
                    raise ValueError(f'Unknown BC type: {bc_type}')
    return (A.tocsr(), b)

def residual_and_jacobian(u, u_old, shape, spacing, dt, bcs, D0=1e-10, beta=5.0, u_ref=0.2, build_jacobian=True, D_func=None, dD_func=None, D_kwargs=None):
    """
    Fully implicit nonlinear residual R(u)=0
    and exact sparse Jacobian for arithmetic face diffusivity.
    """
    check_bcs(shape, bcs)
    U = np.asarray(u, dtype=float).reshape(shape)
    Uold = np.asarray(u_old, dtype=float).reshape(shape)
    N = int(np.prod(shape))
    R = (U - Uold).ravel(order='C').copy()
    J = None
    if build_jacobian:
        J = lil_matrix((N, N))
        J.setdiag(np.ones(N))
    Dnodes = _eval_D_model(U, D0, beta, u_ref, D_func, D_kwargs)
    dDnodes = _eval_dD_model(U, D0, beta, u_ref, dD_func, D_kwargs)
    for idx in np.ndindex(*shape):
        p = flat_index(idx, shape)
        for axis, hgrid in enumerate(spacing):
            if idx[axis] < shape[axis] - 1:
                nb_idx = list(idx)
                nb_idx[axis] += 1
                nb_idx = tuple(nb_idx)
                q = flat_index(nb_idx, shape)
                du = U[idx] - U[nb_idx]
                Df = 0.5 * (Dnodes[idx] + Dnodes[nb_idx])
                c = dt / hgrid ** 2
                R[p] += c * Df * du
                R[q] -= c * Df * du
                if build_jacobian:
                    dRp_dUp = c * (Df + 0.5 * dDnodes[idx] * du)
                    dRp_dUn = c * (-Df + 0.5 * dDnodes[nb_idx] * du)
                    J[p, p] += dRp_dUp
                    J[p, q] += dRp_dUn
                    J[q, p] -= dRp_dUp
                    J[q, q] -= dRp_dUn
            for side, on_boundary in ((0, idx[axis] == 0), (1, idx[axis] == shape[axis] - 1)):
                if not on_boundary:
                    continue
                bc = bcs[axis, side]
                bc_type = bc['type'].lower()
                if bc_type == 'dirichlet':
                    uD = float(bc['value'])
                    du = U[idx] - uD
                    Db = 0.5 * (Dnodes[idx] + float(_eval_D_model(uD, D0, beta, u_ref, D_func, D_kwargs)))
                    c = 2.0 * dt / hgrid ** 2
                    R[p] += c * Db * du
                    if build_jacobian:
                        J[p, p] += c * (Db + 0.5 * dDnodes[idx] * du)
                elif bc_type == 'neumann':
                    qn = float(bc['value'])
                    R[p] += dt * qn / hgrid
                elif bc_type == 'robin':
                    h_c = float(bc['h'])
                    u_env = float(bc['env'])
                    du = U[idx] - u_env
                    H, Hp = robin_effective_H(U[idx], hgrid, h_c, D0, beta, u_ref, D_func, dD_func, D_kwargs)
                    c = dt / hgrid
                    R[p] += c * H * du
                    if build_jacobian:
                        J[p, p] += c * (H + Hp * du)
                else:
                    raise ValueError(f'Unknown BC type: {bc_type}')
    if build_jacobian:
        J = J.tocsr()
    return (R, J)

def picard_step(u_old, shape, spacing, dt, bcs, D0=1e-10, beta=5.0, u_ref=0.2, tol=1e-09, max_iter=50, omega=1.0, init=None, D_func=None, dD_func=None, D_kwargs=None):
    uk = np.array(u_old if init is None else init, dtype=float).reshape(shape).copy()
    history = []
    for k in range(max_iter):
        A, b = assemble_picard_system(uk, u_old, shape, spacing, dt, bcs, D0, beta, u_ref, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
        u_linear = spsolve(A, b).reshape(shape)
        u_new = uk + omega * (u_linear - uk)
        err = np.max(np.abs(u_new - uk))
        history.append(err)
        uk = u_new
        if err < tol:
            break
    return (uk, history)

def newton_step(u_old, shape, spacing, dt, bcs, D0=1e-10, beta=5.0, u_ref=0.2, tol_res=1e-10, tol_step=1e-10, max_iter=20, init=None, line_search=True, D_func=None, dD_func=None, D_kwargs=None):
    u = np.array(u_old if init is None else init, dtype=float).reshape(shape).copy()
    history = []
    for k in range(max_iter):
        R, J = residual_and_jacobian(u, u_old, shape, spacing, dt, bcs, D0, beta, u_ref, build_jacobian=True, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
        res_inf = np.linalg.norm(R, ord=np.inf)
        if res_inf < tol_res:
            history.append({'iter': k, 'res_inf': res_inf, 'step_inf': 0.0, 'lambda': 1.0})
            break
        du = spsolve(J, -R).reshape(shape)
        lam = 1.0
        if line_search:
            phi0 = 0.5 * (R @ R)
            while lam > 0.0001:
                u_trial = u + lam * du
                R_trial, _ = residual_and_jacobian(u_trial, u_old, shape, spacing, dt, bcs, D0, beta, u_ref, build_jacobian=False, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
                phi_trial = 0.5 * (R_trial @ R_trial)
                if phi_trial < phi0:
                    break
                lam *= 0.5
        step_inf = np.max(np.abs(lam * du))
        history.append({'iter': k, 'res_inf': res_inf, 'step_inf': step_inf, 'lambda': lam})
        u = u + lam * du
        if step_inf < tol_step:
            break
    return (u, history)

def hybrid_step(u_old, shape, spacing, dt, bcs, D0=1e-10, beta=5.0, u_ref=0.2, n_picard_warm=2, D_func=None, dD_func=None, D_kwargs=None):
    u_warm, pic_hist = picard_step(u_old, shape, spacing, dt, bcs, D0, beta, u_ref, tol=-1.0, max_iter=n_picard_warm, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
    u_new, newt_hist = newton_step(u_old, shape, spacing, dt, bcs, D0, beta, u_ref, init=u_warm, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
    return (u_new, pic_hist, newt_hist)

def run_time_integration(u0, shape, spacing, dt, nsteps, bcs, method='picard', D0=1e-10, beta=5.0, u_ref=0.2, D_func=None, dD_func=None, D_kwargs=None):
    u = np.asarray(u0, dtype=float).reshape(shape).copy()
    snapshots = [u.copy()]
    iteration_counts = []
    for n in range(nsteps):
        u_old = u.copy()
        if method == 'picard':
            u, hist = picard_step(u_old, shape, spacing, dt, bcs, D0, beta, u_ref, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
            iteration_counts.append(len(hist))
        elif method == 'newton':
            u, hist = newton_step(u_old, shape, spacing, dt, bcs, D0, beta, u_ref, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
            iteration_counts.append(len(hist))
        elif method == 'hybrid':
            u, ph, nh = hybrid_step(u_old, shape, spacing, dt, bcs, D0, beta, u_ref, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
            iteration_counts.append(len(ph) + len(nh))
        else:
            raise ValueError('method must be picard/newton/hybrid')
        snapshots.append(u.copy())
    return (snapshots, np.asarray(iteration_counts))
_FACE_TO_KEY = {'x-': (0, 0), 'x+': (0, 1), 'y-': (1, 0), 'y+': (1, 1), 'z-': (2, 0), 'z+': (2, 1)}

def BC_D(value):
    """Dirichlet BC: u = value."""
    return {'type': 'dirichlet', 'value': float(value)}

def BC_N(flux=0.0):
    """Neumann BC: prescribed outward flux."""
    return {'type': 'neumann', 'value': float(flux)}

def BC_R(h, env):
    """
    Robin family.

    h = 0       -> homogeneous Neumann
    finite h    -> Robin
    h = inf     -> Dirichlet u = env
    """
    h = float(h)
    env = float(env)
    if h == 0.0:
        return BC_N(0.0)
    if np.isinf(h):
        return BC_D(env)
    if h < 0.0:
        raise ValueError('Robin h must be non-negative.')
    return {'type': 'robin', 'h': h, 'env': env}

def make_box_bcs(dim, default=None, overrides=None):
    """
    Build complete BC dictionary for dim=1,2,3.

    Parameters
    ----------
    dim : int
        1, 2, or 3.
    default : BC dictionary
        Used on every active face unless overridden.
        Default: homogeneous Neumann.
    overrides : dict
        Example:
        {
            "x-": BC_D(0.35),
            "x+": BC_R(5e-8, 0.10),
            "y+": BC_N(0.0),
        }
    """
    if dim not in (1, 2, 3):
        raise ValueError('dim must be 1, 2, or 3.')
    if default is None:
        default = BC_N(0.0)
    active_labels = ['x-', 'x+'] + (['y-', 'y+'] if dim >= 2 else []) + (['z-', 'z+'] if dim >= 3 else [])
    bcs = {_FACE_TO_KEY[label]: dict(default) for label in active_labels}
    if overrides is not None:
        for label, bc in overrides.items():
            if label not in active_labels:
                raise ValueError(f'{label!r} is not an active face for dim={dim}.')
            bcs[_FACE_TO_KEY[label]] = dict(bc)
    return bcs

def cell_center_axes(shape, lengths):
    """Return cell-center coordinate arrays for each active axis."""
    shape = tuple((int(n) for n in shape))
    lengths = tuple((float(L) for L in lengths))
    if len(shape) != len(lengths):
        raise ValueError('shape and lengths must have the same dimension.')
    return tuple(((np.arange(n) + 0.5) * (L / n) for n, L in zip(shape, lengths)))

def solve_fdm_box_nd(shape, lengths, bcs, *, u0=0.3, dt=1800.0, nsteps=8, method='picard', D0=1e-10, beta=5.0, u_ref=0.2, D_func=None, dD_func=None, D_kwargs=None):
    """
    One entry point for 1D / 2D / 3D nonlinear diffusion.

    dim is inferred from len(shape).
    """
    shape = tuple((int(n) for n in shape))
    lengths = tuple((float(L) for L in lengths))
    dim = len(shape)
    if dim not in (1, 2, 3):
        raise ValueError('Only 1D, 2D, and 3D are supported.')
    if len(lengths) != dim:
        raise ValueError('lengths must match shape dimension.')
    spacing = tuple((L / n for L, n in zip(lengths, shape)))
    check_bcs(shape, bcs)
    if np.isscalar(u0):
        U0 = np.full(shape, float(u0))
    else:
        U0 = np.asarray(u0, dtype=float).reshape(shape)
    snapshots, counts = run_time_integration(U0, shape, spacing, dt, nsteps, bcs, method=method, D0=D0, beta=beta, u_ref=u_ref, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
    return {'dim': dim, 'shape': shape, 'lengths': lengths, 'spacing': spacing, 'axes': cell_center_axes(shape, lengths), 'bcs': bcs, 'snapshots': snapshots, 'final': snapshots[-1], 'iteration_counts': counts}
