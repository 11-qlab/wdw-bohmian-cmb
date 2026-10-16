"""
HH-IC WDW mode solver + correct Bogoliubov extraction.

Strategy:
  1. Build the HH wavefunction at a_min as a Gaussian in φ.
  2. Project onto sine basis to get c_n(a_min).
  3. Take ∂_a c_n(a_min) = 0 (HH is a "cosine" type IC).
  4. Evolve the coupled system with the CORRECT sign WDW operator.
  5. At a_max, fit each c_n to WKB in/out and extract |β/α|².

The HH Gaussian width at small a is set by the Euclidean instanton:
    σ(a)² = a / (m · const)   (dimensional estimate)
At a_min = 0.05, m=1: σ ≈ 0.22.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags


def build_basis(n_modes, phi_max, n_phi=4001):
    phi = np.linspace(-phi_max, phi_max, n_phi)
    psi = np.zeros((n_modes, n_phi))
    for k in range(n_modes):
        n = k + 1
        psi[k] = np.sin(n * np.pi * (phi + phi_max) / (2.0 * phi_max))
    norms = np.trapezoid(psi**2, phi, axis=1)
    return phi, psi, norms


def project_gaussian(phi, psi, norms, sigma):
    G = np.exp(-0.5 * (phi / sigma) ** 2)
    c = np.array([np.trapezoid(psi[k] * G, phi) / norms[k]
                  for k in range(psi.shape[0])])
    return c


def v_matrix(phi, psi, norms, m):
    V_phi = 0.5 * m**2 * phi**2
    n_modes = psi.shape[0]
    V = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(n_modes):
            V[i, j] = np.trapezoid(psi[i] * V_phi * psi[j], phi) / norms[i]
    return V


def solve_hh_coupled(a_min=0.05, a_max=6.0, n_a=2000,
                     phi_max=3.0, n_modes=6, m=1.0,
                     sigma_a=0.3, rtol=1e-11, atol=1e-14):
    phi, psi, norms = build_basis(n_modes, phi_max)
    V = v_matrix(phi, psi, norms, m)
    n = np.arange(1, n_modes + 1)
    lam = (n * np.pi / (2.0 * phi_max)) ** 2

    sigma = sigma_a / np.sqrt(m * max(a_min, 1e-3))
    c0 = project_gaussian(phi, psi, norms, sigma)
    # HH regularity: dc/da at a_min comes from the Euclidean scaling
    # of the amplitude, ~ (1/a) c(a).  Small a_min makes this safe.
    dc0 = c0 / a_min

    y0 = np.concatenate([c0, dc0])

    def rhs(a, y):
        c = y[:n_modes]
        dc = y[n_modes:]
        d2c = a**2 * c - (lam / a**2) * c - (a**4) * (V @ c)
        return np.concatenate([dc, d2c])

    a_eval = np.geomspace(a_min, a_max, n_a)
    sol = solve_ivp(rhs, (a_min, a_max), y0, t_eval=a_eval,
                    method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y[:n_modes].T, lam, V


def extract_bogoliubov_single(a, c, dc, lam, V_nn, a_frac=0.8):
    """
    Extract |β/α|² by WKB matching at a window near a_max.
    Use several points and average to reduce noise.
    """
    i0 = int(a_frac * len(a))
    results = []
    for i in range(i0, len(a)):
        ai = a[i]
        w2 = ai**4 * V_nn - ai**2 + lam / ai**2
        if w2 <= 0:
            continue
        w = np.sqrt(w2)
        ci = c[i]
        dci = dc[i]
        num = abs(ci - 1j * dci / w) ** 2
        den = abs(ci + 1j * dci / w) ** 2
        if den > 1e-300:
            results.append(num / den)
    if not results:
        return np.nan
    return float(np.median(results))


if __name__ == "__main__":
    print(f"{'n':>3}  {'m':>5}  {'phi_max':>8}  {'V_nn':>8}  "
          f"{'lam':>8}  {'|β/α|²':>12}")
    print("-" * 58)
    for m in (0.5, 1.0, 2.0, 4.0):
        for phi_max in (2.0, 3.0, 5.0):
            try:
                a, c, lam, V = solve_hh_coupled(
                    phi_max=phi_max, n_modes=6, m=m)
                for k in range(4):  # show first four modes
                    bsq = extract_bogoliubov_single(
                        a, c[:, k], None, lam[k], V[k, k])
                    # need dc too — recompute
                    dc = np.gradient(c[:, k], a)
                    bsq = extract_bogoliubov_single(
                        a, c[:, k], dc, lam[k], V[k, k])
                    print(f"{k+1:>3}  {m:>5.2f}  {phi_max:>8.2f}  "
                          f"{V[k,k]:>8.3f}  {lam[k]:>8.4f}  "
                          f"{bsq:>12.4e}")
            except Exception as e:
                print(f"ERR m={m} phi_max={phi_max}: {e}")
