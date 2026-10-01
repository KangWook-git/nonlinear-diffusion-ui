# Numerical core retained from the verified Technical Note backend.
# Public UI release v2.3 keeps this backend algorithm unchanged.

import numpy as np
import math
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def D_nl(u, D0=1e-10, beta=5.0, u_ref=0.2):
    return D0 * np.exp(beta * (np.asarray(u) - u_ref))

def dD_nl(u, D0=1e-10, beta=5.0, u_ref=0.2):
    return beta * D_nl(u, D0, beta, u_ref)

def _eval_D_model(u, D0, beta, u_ref, D_func=None, D_kwargs=None):
    """Evaluate default exponential law or an injected constitutive law."""
    if D_func is None:
        return D_nl(u, D0, beta, u_ref)
    return np.asarray(D_func(u, **{} if D_kwargs is None else D_kwargs), dtype=float)

def _eval_dD_model(u, D0, beta, u_ref, dD_func=None, D_kwargs=None):
    """Derivative counterpart used by Newton/Jacobian assembly."""
    if dD_func is None:
        return dD_nl(u, D0, beta, u_ref)
    return np.asarray(dD_func(u, **{} if D_kwargs is None else D_kwargs), dtype=float)

def simplex_geometry(xe):
    """
    xe: (d+1, d)
    Returns measure and gradients grad(N_i), shape (d+1, d).
    """
    xe = np.asarray(xe, dtype=float)
    nloc, dim = xe.shape
    assert nloc == dim + 1
    A = np.column_stack([np.ones(nloc), xe])
    invA = np.linalg.inv(A)
    grads = invA[1:, :].T
    B = (xe[1:] - xe[0]).T
    measure = abs(np.linalg.det(B)) / math.factorial(dim)
    return (measure, grads)

def simplex_mass(measure, dim):
    nloc = dim + 1
    return measure / ((dim + 1) * (dim + 2)) * (np.ones((nloc, nloc)) + np.eye(nloc))

def simplex_K0(measure, grads):
    return measure * (grads @ grads.T)

def boundary_measure(xb):
    xb = np.asarray(xb, dtype=float)
    m, dim = xb.shape
    if m == 1:
        return 1.0
    if m == 2:
        return np.linalg.norm(xb[1] - xb[0])
    if m == 3:
        a = xb[1] - xb[0]
        b = xb[2] - xb[0]
        return 0.5 * np.linalg.norm(np.cross(a, b))
    raise ValueError('Boundary entity must have 1, 2, or 3 nodes.')

def boundary_mass_and_load(xb):
    xb = np.asarray(xb, dtype=float)
    m = xb.shape[0]
    meas = boundary_measure(xb)
    if m == 1:
        Mb = np.array([[1.0]])
        lb = np.array([1.0])
    elif m == 2:
        Mb = meas / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        lb = meas / 2.0 * np.ones(2)
    elif m == 3:
        Mb = meas / 12.0 * (np.ones((3, 3)) + np.eye(3))
        lb = meas / 3.0 * np.ones(3)
    else:
        raise ValueError
    return (Mb, lb)

def mesh_1d(L=0.05, n=40):
    x = np.linspace(0.0, L, n + 1)[:, None]
    elems = np.column_stack([np.arange(n), np.arange(1, n + 1)]).astype(int)
    boundary = {(0, 0): [np.array([0], dtype=int)], (0, 1): [np.array([n], dtype=int)]}
    return (x, elems, boundary)

def mesh_2d_rect(Lx=0.05, Ly=0.03, nx=16, ny=10):
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    points = np.array([[x, y] for x in xs for y in ys], dtype=float)

    def node(i, j):
        return i * (ny + 1) + j
    elems = []
    for i in range(nx):
        for j in range(ny):
            n00 = node(i, j)
            n10 = node(i + 1, j)
            n11 = node(i + 1, j + 1)
            n01 = node(i, j + 1)
            elems.append([n00, n10, n11])
            elems.append([n00, n11, n01])
    boundary = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    for j in range(ny):
        boundary[0, 0].append(np.array([node(0, j), node(0, j + 1)], dtype=int))
        boundary[0, 1].append(np.array([node(nx, j), node(nx, j + 1)], dtype=int))
    for i in range(nx):
        boundary[1, 0].append(np.array([node(i, 0), node(i + 1, 0)], dtype=int))
        boundary[1, 1].append(np.array([node(i, ny), node(i + 1, ny)], dtype=int))
    return (points, np.asarray(elems, dtype=int), boundary)

def mesh_3d_box(Lx=0.05, Ly=0.04, Lz=0.03, nx=5, ny=4, nz=3):
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    zs = np.linspace(0.0, Lz, nz + 1)
    points = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)

    def node(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k
    elems = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v0 = node(i, j, k)
                v1 = node(i + 1, j, k)
                v2 = node(i + 1, j + 1, k)
                v3 = node(i, j + 1, k)
                v4 = node(i, j, k + 1)
                v5 = node(i + 1, j, k + 1)
                v6 = node(i + 1, j + 1, k + 1)
                v7 = node(i, j + 1, k + 1)
                elems.extend([[v0, v1, v2, v6], [v0, v2, v3, v6], [v0, v3, v7, v6], [v0, v7, v4, v6], [v0, v4, v5, v6], [v0, v5, v1, v6]])
    boundary = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): [], (2, 0): [], (2, 1): []}
    for j in range(ny):
        for k in range(nz):
            for side, i in [(0, 0), (1, nx)]:
                a = node(i, j, k)
                b = node(i, j + 1, k)
                c = node(i, j + 1, k + 1)
                d = node(i, j, k + 1)
                boundary[0, side].extend([np.array([a, b, c], dtype=int), np.array([a, c, d], dtype=int)])
    for i in range(nx):
        for k in range(nz):
            for side, j in [(0, 0), (1, ny)]:
                a = node(i, j, k)
                b = node(i + 1, j, k)
                c = node(i + 1, j, k + 1)
                d = node(i, j, k + 1)
                boundary[1, side].extend([np.array([a, b, c], dtype=int), np.array([a, c, d], dtype=int)])
    for i in range(nx):
        for j in range(ny):
            for side, k in [(0, 0), (1, nz)]:
                a = node(i, j, k)
                b = node(i + 1, j, k)
                c = node(i + 1, j + 1, k)
                d = node(i, j + 1, k)
                boundary[2, side].extend([np.array([a, b, c], dtype=int), np.array([a, c, d], dtype=int)])
    return (points, np.asarray(elems, dtype=int), boundary)

def collect_dirichlet_nodes(boundary, bcs):
    vals = {}
    for key, entities in boundary.items():
        bc = bcs[key]
        if bc['type'].lower() != 'dirichlet':
            continue
        g = float(bc['value'])
        for ent in entities:
            for node in ent:
                if node in vals and abs(vals[node] - g) > 1e-12:
                    raise ValueError('Conflicting Dirichlet values at a corner/edge node.')
                vals[int(node)] = g
    return vals

def apply_dirichlet_linear(A, b, dirichlet):
    """
    Symmetric elimination for linear Picard system.
    """
    A = A.tolil(copy=True)
    b = b.copy()
    for node, g in dirichlet.items():
        col = A[:, node].toarray().ravel()
        b -= col * g
    for node, g in dirichlet.items():
        A[:, node] = 0.0
        A[node, :] = 0.0
        A[node, node] = 1.0
        b[node] = g
    return (A.tocsr(), b)

def assemble_picard(Ucoeff, Uold, points, elems, boundary, bcs, dt, D0=1e-10, beta=5.0, u_ref=0.2, D_func=None, dD_func=None, D_kwargs=None):
    N = len(points)
    dim = points.shape[1]
    A = lil_matrix((N, N))
    b = np.zeros(N)
    for conn in elems:
        xe = points[conn]
        ue_coeff = Ucoeff[conn]
        ue_old = Uold[conn]
        meas, grads = simplex_geometry(xe)
        Me = simplex_mass(meas, dim)
        K0e = simplex_K0(meas, grads)
        De = np.mean(_eval_D_model(ue_coeff, D0, beta, u_ref, D_func, D_kwargs))
        Ae = Me / dt + De * K0e
        be = Me / dt @ ue_old
        for a, Aglob in enumerate(conn):
            b[Aglob] += be[a]
            for c, Cglob in enumerate(conn):
                A[Aglob, Cglob] += Ae[a, c]
    for key, entities in boundary.items():
        bc = bcs[key]
        typ = bc['type'].lower()
        if typ == 'dirichlet':
            continue
        for ent in entities:
            xb = points[ent]
            Mb, lb = boundary_mass_and_load(xb)
            if typ == 'neumann':
                qn = float(bc['value'])
                fb = -qn * lb
                for a, Aglob in enumerate(ent):
                    b[Aglob] += fb[a]
            elif typ == 'robin':
                h = float(bc['h'])
                env = float(bc['env'])
                Kb = h * Mb
                fb = h * env * lb
                for a, Aglob in enumerate(ent):
                    b[Aglob] += fb[a]
                    for c, Cglob in enumerate(ent):
                        A[Aglob, Cglob] += Kb[a, c]
            else:
                raise ValueError(f'Unknown BC type {typ}')
    dirichlet = collect_dirichlet_nodes(boundary, bcs)
    A, b = apply_dirichlet_linear(A.tocsr(), b, dirichlet)
    return (A, b)

def residual_jacobian(U, Uold, points, elems, boundary, bcs, dt, D0=1e-10, beta=5.0, u_ref=0.2, build_jacobian=True, D_func=None, dD_func=None, D_kwargs=None):
    N = len(points)
    dim = points.shape[1]
    R = np.zeros(N)
    J = lil_matrix((N, N)) if build_jacobian else None
    for conn in elems:
        xe = points[conn]
        ue = U[conn]
        ue_old = Uold[conn]
        meas, grads = simplex_geometry(xe)
        Me = simplex_mass(meas, dim)
        K0e = simplex_K0(meas, grads)
        Dvals = _eval_D_model(ue, D0, beta, u_ref, D_func, D_kwargs)
        dDvals = _eval_dD_model(ue, D0, beta, u_ref, dD_func, D_kwargs)
        De = np.mean(Dvals)
        dDe = dDvals / len(conn)
        w = K0e @ ue
        Re = Me / dt @ (ue - ue_old) + De * w
        if build_jacobian:
            Je = Me / dt + De * K0e + np.outer(w, dDe)
        for a, Aglob in enumerate(conn):
            R[Aglob] += Re[a]
            if build_jacobian:
                for c, Cglob in enumerate(conn):
                    J[Aglob, Cglob] += Je[a, c]
    for key, entities in boundary.items():
        bc = bcs[key]
        typ = bc['type'].lower()
        if typ == 'dirichlet':
            continue
        for ent in entities:
            xb = points[ent]
            Mb, lb = boundary_mass_and_load(xb)
            if typ == 'neumann':
                qn = float(bc['value'])
                Rb = qn * lb
                for a, Aglob in enumerate(ent):
                    R[Aglob] += Rb[a]
            elif typ == 'robin':
                h = float(bc['h'])
                env = float(bc['env'])
                ue = U[ent]
                Rb = h * (Mb @ ue - env * lb)
                for a, Aglob in enumerate(ent):
                    R[Aglob] += Rb[a]
                    if build_jacobian:
                        for c, Cglob in enumerate(ent):
                            J[Aglob, Cglob] += h * Mb[a, c]
            else:
                raise ValueError(f'Unknown BC type {typ}')
    dirichlet = collect_dirichlet_nodes(boundary, bcs)
    if build_jacobian:
        J = J.tolil()
    for node, g in dirichlet.items():
        R[node] = U[node] - g
        if build_jacobian:
            J[node, :] = 0.0
            J[node, node] = 1.0
    if build_jacobian:
        J = J.tocsr()
    return (R, J)

def picard_step(Uold, points, elems, boundary, bcs, dt, D0=1e-10, beta=5.0, u_ref=0.2, tol=1e-09, max_iter=50, omega=1.0, init=None, D_func=None, dD_func=None, D_kwargs=None):
    U = np.array(Uold if init is None else init, dtype=float).copy()
    for node, g in collect_dirichlet_nodes(boundary, bcs).items():
        U[node] = g
    hist = []
    for k in range(max_iter):
        A, b = assemble_picard(U, Uold, points, elems, boundary, bcs, dt, D0, beta, u_ref, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
        Ulin = spsolve(A, b)
        Unew = U + omega * (Ulin - U)
        err = np.max(np.abs(Unew - U))
        hist.append(err)
        U = Unew
        if err < tol:
            break
    return (U, hist)

def newton_step(Uold, points, elems, boundary, bcs, dt, D0=1e-10, beta=5.0, u_ref=0.2, tol_res=1e-10, tol_step=1e-10, max_iter=20, init=None, D_func=None, dD_func=None, D_kwargs=None):
    U = np.array(Uold if init is None else init, dtype=float).copy()
    for node, g in collect_dirichlet_nodes(boundary, bcs).items():
        U[node] = g
    hist = []
    res0 = None
    for k in range(max_iter):
        R, J = residual_jacobian(U, Uold, points, elems, boundary, bcs, dt, D0, beta, u_ref, build_jacobian=True, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
        res_inf = np.linalg.norm(R, ord=np.inf)
        if res0 is None:
            res0 = max(res_inf, 1e-30)
        res_rel = res_inf / res0
        if res_rel < tol_res:
            hist.append({'iter': k, 'res_inf': res_inf, 'res_rel': res_rel, 'step_inf': 0.0, 'lambda': 1.0})
            break
        dU = spsolve(J, -R)
        lam = 1.0
        phi0 = 0.5 * (R @ R)
        while lam > 0.0001:
            Utrial = U + lam * dU
            for node, g in collect_dirichlet_nodes(boundary, bcs).items():
                Utrial[node] = g
            Rtrial, _ = residual_jacobian(Utrial, Uold, points, elems, boundary, bcs, dt, D0, beta, u_ref, build_jacobian=False, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
            if 0.5 * (Rtrial @ Rtrial) < phi0:
                break
            lam *= 0.5
        step_inf = np.max(np.abs(lam * dU))
        hist.append({'iter': k, 'res_inf': res_inf, 'res_rel': res_rel, 'step_inf': step_inf, 'lambda': lam})
        U = Utrial
        if step_inf < tol_step:
            break
    return (U, hist)

def hybrid_step(Uold, points, elems, boundary, bcs, dt, D0=1e-10, beta=5.0, u_ref=0.2, n_picard_warm=2, D_func=None, dD_func=None, D_kwargs=None):
    Uw, ph = picard_step(Uold, points, elems, boundary, bcs, dt, D0, beta, u_ref, tol=-1.0, max_iter=n_picard_warm, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
    Un, nh = newton_step(Uold, points, elems, boundary, bcs, dt, D0, beta, u_ref, init=Uw, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
    return (Un, ph, nh)

def run_time(U0, points, elems, boundary, bcs, dt, nsteps, method='picard', D0=1e-10, beta=5.0, u_ref=0.2, D_func=None, dD_func=None, D_kwargs=None):
    U = np.asarray(U0, dtype=float).copy()
    snaps = [U.copy()]
    counts = []
    for n in range(nsteps):
        Uold = U.copy()
        if method == 'picard':
            U, h = picard_step(Uold, points, elems, boundary, bcs, dt, D0, beta, u_ref, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
            counts.append(len(h))
        elif method == 'newton':
            U, h = newton_step(Uold, points, elems, boundary, bcs, dt, D0, beta, u_ref, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
            counts.append(len(h))
        elif method == 'hybrid':
            U, ph, nh = hybrid_step(Uold, points, elems, boundary, bcs, dt, D0, beta, u_ref, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
            counts.append(len(ph) + len(nh))
        else:
            raise ValueError
        snaps.append(U.copy())
    return (snaps, np.asarray(counts))
_FEM_FACE_TO_KEY = {'x-': (0, 0), 'x+': (0, 1), 'y-': (1, 0), 'y+': (1, 1), 'z-': (2, 0), 'z+': (2, 1)}

def FEM_BC_D(value):
    return {'type': 'dirichlet', 'value': float(value)}

def FEM_BC_N(flux=0.0):
    return {'type': 'neumann', 'value': float(flux)}

def FEM_BC_R(h, env):
    """
    Robin family with exact limit handling.

    h=0   -> homogeneous Neumann
    h=inf -> strong Dirichlet u=env
    """
    h = float(h)
    env = float(env)
    if h == 0.0:
        return FEM_BC_N(0.0)
    if np.isinf(h):
        return FEM_BC_D(env)
    if h < 0.0:
        raise ValueError('Robin h must be non-negative.')
    return {'type': 'robin', 'h': h, 'env': env}

def make_fem_box_bcs(dim, default=None, overrides=None):
    if dim not in (1, 2, 3):
        raise ValueError('dim must be 1, 2, or 3.')
    if default is None:
        default = FEM_BC_N(0.0)
    active_labels = ['x-', 'x+'] + (['y-', 'y+'] if dim >= 2 else []) + (['z-', 'z+'] if dim >= 3 else [])
    bcs = {_FEM_FACE_TO_KEY[label]: dict(default) for label in active_labels}
    if overrides is not None:
        for label, bc in overrides.items():
            if label not in active_labels:
                raise ValueError(f'{label!r} is not an active face for dim={dim}.')
            bcs[_FEM_FACE_TO_KEY[label]] = dict(bc)
    return bcs

def make_simplex_box_nd(dim, lengths, cells):
    """
    Dimension-aware mesh factory.

    dim=1 -> line mesh
    dim=2 -> triangular rectangle mesh
    dim=3 -> tetrahedral box mesh
    """
    lengths = tuple((float(L) for L in lengths))
    cells = tuple((int(n) for n in cells))
    if len(lengths) != dim or len(cells) != dim:
        raise ValueError('lengths and cells must match dim.')
    if dim == 1:
        return mesh_1d(L=lengths[0], n=cells[0])
    if dim == 2:
        return mesh_2d_rect(Lx=lengths[0], Ly=lengths[1], nx=cells[0], ny=cells[1])
    if dim == 3:
        return mesh_3d_box(Lx=lengths[0], Ly=lengths[1], Lz=lengths[2], nx=cells[0], ny=cells[1], nz=cells[2])
    raise ValueError('dim must be 1, 2, or 3.')

def solve_fem_box_nd(dim, lengths, cells, bcs, *, u0=0.3, dt=1800.0, nsteps=6, method='picard', D0=1e-10, beta=5.0, u_ref=0.2, D_func=None, dD_func=None, D_kwargs=None):
    """
    One entry point for 1D / 2D / 3D nonlinear simplex FEM.
    """
    points, elems, boundary = make_simplex_box_nd(dim, lengths, cells)
    expected_keys = {(axis, side) for axis in range(dim) for side in (0, 1)}
    if set(bcs.keys()) != expected_keys:
        raise ValueError('BC dictionary must define every active boundary face.')
    if np.isscalar(u0):
        U0 = np.full(len(points), float(u0))
    else:
        U0 = np.asarray(u0, dtype=float).copy()
    for node, value in collect_dirichlet_nodes(boundary, bcs).items():
        U0[node] = value
    snapshots, counts = run_time(U0, points, elems, boundary, bcs, dt, nsteps, method=method, D0=D0, beta=beta, u_ref=u_ref, D_func=D_func, dD_func=dD_func, D_kwargs=D_kwargs)
    return {'dim': dim, 'lengths': tuple(lengths), 'cells': tuple(cells), 'points': points, 'elems': elems, 'boundary': boundary, 'bcs': bcs, 'snapshots': snapshots, 'final': snapshots[-1], 'iteration_counts': counts}
