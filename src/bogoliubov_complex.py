"""Complex-mode WDW solver + Bogoliubov extraction (with phase param)."""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp


def v_nn(n, phi_max, m, n_phi=4001):
    phi = np.linspace(-phi_max, phi_max, n_phi)
    psi = np.sin(n * np.pi * (phi + phi_max) / (2.0 * phi_max))
    return np.trapezoid(psi**2 * 0.5 * m**2 * phi**2, phi) / np.trapezoid(psi**2, phi)


def solve_mode(n, phi_max, m, a_min=0.05, a_max=8.0, n_a=3000,
               phase=np.pi / 4):
    lam = (n * np.pi / (2.0 * phi_max)) ** 2
    Vnn = v_nn(n, phi_max, m)
    nu = np.sqrt(max(lam - 0.25, 0.0))
    p = 0.5 + 1j * nu

    # FIX: choose an IC with nonzero Wronskian, representing an
    # actual positive-frequency WKB mode at a_min.
    # c(a_min) = e^{-i phase} / sqrt(2 w_min)
    # c'(a_min) = (-i w_min) c(a_min)
    w2_min = a_min**4 * Vnn - a_min**2 + lam / a_min**2
    if w2_min > 0:
        w_min = np.sqrt(w2_min)
        c0 = np.exp(-1j * phase) / np.sqrt(2 * w_min)
        dc0 = -1j * w_min * c0
    else:
        # forbidden region: fall back to Euclidean power law
        c0 = a_min ** p * np.exp(-1j * phase)
        dc0 = (p / a_min) * c0

    def rhs(a, y):
        c, dc = y
        omega2 = a**4 * Vnn - a**2 + lam / a**2
        return [dc, -omega2 * c]

    a_eval = np.geomspace(a_min, a_max, n_a)
    sol = solve_ivp(rhs, (a_min, a_max), [c0, dc0], t_eval=a_eval,
                    method="DOP853", rtol=1e-12, atol=1e-16)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y[0], sol.y[1], lam, Vnn


def extract_bogoliubov(a, c, dc, lam, Vnn):
    """
    Returns (|beta/alpha|^2, |alpha|^2 - |beta|^2).

    |beta/alpha|^2 = |c - i c'/w|^2 / |c + i c'/w|^2
    The unitarity check value |alpha|^2 - |beta|^2 is derived
    from the conserved Wronskian.
    """
    # Unitarity from Wronskian:
    # W = i (|beta|^2 - |alpha|^2)  in the sign convention used
    # so |alpha|^2 - |beta|^2 = -W / i = i W
    W = c.conj() * dc - c * dc.conj()
    unitary = np.real(1j * W)  # should be ~ const, ideally = 1

    i0 = int(0.7 * len(a))
    vals = []
    for i in range(i0, len(a)):
        ai = a[i]
        w2 = ai**4 * Vnn - ai**2 + lam / ai**2
        if w2 <= 0:
            continue
        w = np.sqrt(w2)
        num = abs(c[i] - 1j * dc[i] / w) ** 2
        den = abs(c[i] + 1j * dc[i] / w) ** 2
        if den > 1e-300:
            vals.append(num / den)
    bsq = float(np.median(vals)) if vals else np.nan
    return bsq, float(np.median(unitary))


if __name__ == "__main__":
    print(f"{'n':>3}  {'m':>5}  {'phi_max':>8}  {'|β/α|²':>12}  {'unitarity':>12}")
    print("-" * 55)
    for n in (1, 2, 3, 4):
        for m in (0.5, 1.0, 2.0, 4.0):
            for phi_max in (3.0, 5.0):
                try:
                    a, c, dc, lam, Vnn = solve_mode(n, phi_max, m)
                    bsq, uni = extract_bogoliubov(a, c, dc, lam, Vnn)
                    print(f"{n:>3}  {m:>5.2f}  {phi_max:>8.2f}  "
                          f"{bsq:>12.4e}  {uni:>12.4e}")
                except Exception as e:
                    print(f"ERR n={n} m={m} phi_max={phi_max}: {e}")
