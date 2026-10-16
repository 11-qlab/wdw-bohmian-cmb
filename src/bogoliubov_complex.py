"""
Complex-mode WDW solver + real Bogoliubov extraction.

Mode equation (single mode, HH-regular IC):
    c'' + [a^4 V_nn - a^2 + lam/a^2] c = 0
       = c'' + omega^2(a) c = 0

We use the COMPLEX mode that corresponds to the Euclidean
no-boundary regularity:

    c(a_min)  = a_min^{1/2 + i*nu}
    c'(a_min) = (1/2 + i*nu)/a_min * c(a_min)
    nu = sqrt(max(lam - 1/4, 0))

For lam < 1/4, nu = 0 and the IC is real. We break this by
adding a small imaginary part representing the HH phase:

    c(a_min) = a_min^{p+} * exp(i * pi/4)   (a "Gaussian phase")
"""
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
    if abs(nu) < 1e-14:
        # lam <= 1/4: real exponent branch, add HH phase manually
        c0 = a_min ** 0.5 * np.exp(1j * phase)
    else:
        c0 = a_min ** p
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
    |β/α|² = |c - i c'/ω|² / |c + i c'/ω|²

    This is only meaningful when c is COMPLEX. Averaged over the
    last 30% of the evolution.
    """
    i0 = int(0.7 * len(a))
    vals = []
    for i in range(i0, len(a)):
        ai = a[i]
        w2 = ai**4 * Vnn - ai**2 + lam / ai**2
        if w2 <= 0:
            continue
        w = np.sqrt(w2)
        num = abs(c[i] + 1j * dc[i] / w) ** 2
        den = abs(c[i] - 1j * dc[i] / w) ** 2
        if den > 1e-300:
            vals.append(num / den)
    if not vals:
        return np.nan
    return float(np.median(vals))


if __name__ == "__main__":
    print(f"{'n':>3}  {'m':>5}  {'phi_max':>8}  {'V_nn':>8}  "
          f"{'lam':>8}  {'|β/α|²':>12}")
    print("-" * 58)
    for n in (1, 2, 3, 4):
        for m in (0.5, 1.0, 2.0, 4.0):
            for phi_max in (3.0, 5.0):
                try:
                    a, c, dc, lam, Vnn = solve_mode(n, phi_max, m)
                    # Check that c is genuinely complex
                    imag_frac = np.abs(c.imag).max() / np.abs(c).max()
                    bsq = extract_bogoliubov(a, c, dc, lam, Vnn)
                    print(f"{n:>3}  {m:>5.2f}  {phi_max:>8.2f}  "
                          f"{Vnn:>8.3f}  {lam:>8.4f}  {bsq:>12.4e}"
                          f"   [imag_frac={imag_frac:.3f}]")
                except Exception as e:
                    print(f"ERR n={n} m={m} phi_max={phi_max}: {e}")
