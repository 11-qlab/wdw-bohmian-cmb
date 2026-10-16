"""
Brute-force Wheeler-DeWitt solver.

Discretize (hbar=1):
    [ ∂²_a  -  (1/a²) ∂²_φ  -  a²  +  a⁴ V(φ) ] Ψ = 0

on an (a, φ) grid, build the sparse operator L, impose Dirichlet
on all four sides, then find the eigenvalue of L closest to zero.
That eigenvector is Ψ(a, φ).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ----------------------------------------------------------------------
# Keep the old API so existing tests still pass
# ----------------------------------------------------------------------
def phi_laplacian(n_phi: int, phi_max: float) -> np.ndarray:
    dphi = 2.0 * phi_max / (n_phi - 1)
    off = np.ones(n_phi - 1) / dphi**2
    main = -2.0 * np.ones(n_phi) / dphi**2
    L = sp.diags([off, main, off], [-1, 0, 1], shape=(n_phi, n_phi)).toarray()
    L[0, :] = 0.0
    L[-1, :] = 0.0
    return L


# ----------------------------------------------------------------------
# Brute-force 2D operator
# ----------------------------------------------------------------------
def build_operator(a, phi, m=1.0, hbar=1.0):
    n_a, n_p = a.size, phi.size
    da = a[1] - a[0]
    dp = phi[1] - phi[0]

    main_a = -2.0 / da**2 * np.ones(n_a)
    off_a = 1.0 / da**2 * np.ones(n_a - 1)
    D_a2 = sp.diags([off_a, main_a, off_a], [-1, 0, 1])

    main_p = -2.0 / dp**2 * np.ones(n_p)
    off_p = 1.0 / dp**2 * np.ones(n_p - 1)
    D_p2 = sp.diags([off_p, main_p, off_p], [-1, 0, 1])

    I_a = sp.identity(n_a)
    I_p = sp.identity(n_p)

    Lap_a = sp.kron(D_a2, I_p, format="csr")
    Lap_p = sp.kron(I_a, D_p2, format="csr")

    A, P = np.meshgrid(a, phi, indexing="ij")
    a_f = A.ravel()
    p_f = P.ravel()

    inv_a2 = 1.0 / a_f**2
    V = 0.5 * m**2 * p_f**2

    diag_inv_a2 = sp.diags(inv_a2)
    diag_a2 = sp.diags(a_f**2)
    diag_a4V = sp.diags(a_f**4 * V)

    L = Lap_a - diag_inv_a2 @ Lap_p - diag_a2 + diag_a4V
    return L.tocsr()


def apply_dirichlet(L, n_a, n_p):
    L = L.tolil()
    for j in range(n_p):
        for i in (j, (n_a - 1) * n_p + j):
            L[i, :] = 0.0
            L[i, i] = 1.0
    for i in range(n_a):
        for idx in (i * n_p, i * n_p + n_p - 1):
            L[idx, :] = 0.0
            L[idx, idx] = 1.0
    return L.tocsr()


def solve_wdw_bruteforce(
    a_min=0.3, a_max=4.0, n_a=80,
    phi_max=3.0, n_phi=64,
    m=1.0, hbar=1.0,
):
    a = np.linspace(a_min, a_max, n_a)
    phi = np.linspace(-phi_max, phi_max, n_phi)

    L = build_operator(a, phi, m, hbar)
    L = apply_dirichlet(L, n_a, n_phi)

    vals, vecs = spla.eigs(L, k=1, sigma=0.0, which="LM")
    mu = vals[0].real
    Psi = vecs[:, 0].real.reshape(n_a, n_phi)

    if Psi[n_a // 2, n_phi // 2] < 0:
        Psi = -Psi
    return a, phi, Psi, mu


def marginalized_quantum_potential(Psi, a, phi, hbar=1.0, m=1.0, floor=1e-12):
    dp = phi[1] - phi[0]
    P = np.sum(Psi**2, axis=1) * dp
    mask = P > floor * P.max()
    log_P = np.full_like(P, np.nan)
    log_P[mask] = np.log(P[mask])
    dlog = np.gradient(log_P, a, edge_order=2)
    d2log = np.gradient(dlog, a, edge_order=2)
    Q = -(hbar**2 / (2.0 * m)) * (d2log + dlog**2)
    return P, Q


if __name__ == "__main__":
    a, phi, Psi, mu = solve_wdw_bruteforce()
    P, Q = marginalized_quantum_potential(Psi, a, phi)

    print(f"grid:  n_a={a.size}  n_phi={phi.size}")
    print(f"a:     {a[0]:.3f} .. {a[-1]:.3f}")
    print(f"phi:   {phi[0]:.3f} .. {phi[-1]:.3f}")
    print(f"eigenvalue mu = {mu:.6e}")
    print(f"|Psi| max = {np.abs(Psi).max():.3e}")
    print(f"P(a): {P.min():.3e} .. {P.max():.3e}")
    print(f"P at a_min: {P[0]:.3e},  a_max: {P[-1]:.3e}")

    good = np.isfinite(Q)
    if good.any():
        print(f"Q(a): {Q[good].min():.4f} .. {Q[good].max():.4f}")