"""
Single-mode truncation of the WDW equation.

Ψ(a, φ) ≈ c(a) sin(π(φ+φ_max) / (2 φ_max))

c'' + [-a² + λ/a² + a⁴ m² <φ²>] c = 0

Solve as Cauchy from a_min << 1 with Hartle-Hawking
regularity c(a) ~ a^p+, p+ = (1 + sqrt(1+4λ))/2.
"""
import numpy as np
from scipy.integrate import solve_ivp


def solve_single_mode(a_min=0.01, a_max=5.0, n_a=600,
                      phi_max=3.0, m=1.0, hbar=1.0):
    lam = (np.pi / (2.0 * phi_max)) ** 2

    x = np.linspace(-phi_max, phi_max, 4001)
    psi = np.sin(np.pi * (x + phi_max) / (2.0 * phi_max))
    phi2_avg = np.trapezoid(x**2 * psi**2, x) / np.trapezoid(psi**2, x)

    p_plus = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * lam))

    def rhs(a, y):
        c, dc = y
        d2c = (a**2 - lam / a**2 - a**4 * m**2 * phi2_avg) * c
        return [dc, d2c]

    a_eval = np.linspace(a_min, a_max, n_a)
    y0 = [a_min**p_plus, p_plus * a_min**(p_plus - 1.0)]

    sol = solve_ivp(rhs, (a_min, a_max), y0, t_eval=a_eval,
                    method="DOP853", rtol=1e-11, atol=1e-14)
    if not sol.success:
        raise RuntimeError(sol.message)

    a = sol.t
    c = sol.y[0]

    log_c = np.log(np.abs(c))
    dlog = np.gradient(log_c, a, edge_order=2)
    d2log = np.gradient(dlog, a, edge_order=2)
    Q = -0.5 * hbar**2 * (d2log + dlog**2) / m

    return a, c, Q, lam, phi2_avg, p_plus


if __name__ == "__main__":
    a, c, Q, lam, phi2_avg, p_plus = solve_single_mode()

    print(f"lam_1       = {lam:.6f}")
    print(f"<phi^2>_1   = {phi2_avg:.6f}")
    print(f"p_+         = {p_plus:.6f}")
    print(f"c(a_min)    = {c[0]:.3e}")
    print(f"c(a_max)    = {c[-1]:.3e}")
    print(f"|c| max     = {np.abs(c).max():.3e}")

    # control parameter: |Q| / a²  — the physically meaningful ratio
    ratio = np.abs(Q) / a**2
    for i in (0, a.size // 4, a.size // 2, 3 * a.size // 4, a.size - 1):
        print(f"  a={a[i]:.3f}   Q={Q[i]:+.4e}   |Q|/a²={ratio[i]:.4e}")
