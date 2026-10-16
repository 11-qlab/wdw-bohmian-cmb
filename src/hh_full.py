"""
HH vs Vilenkin mode analysis — one file, both BCs.

Mode equation (even-parity basis, matches HH symmetry):
    c_k'' + omega_k^2(a) c_k = 0
    omega_k^2(a) = a^4 V_kk - a^2 + lambda_k / a^2
    lambda_k = ((2k-1) pi / (2 phi_max))^2
"""
import numpy as np
from scipy.integrate import solve_ivp


def Vkk(k, phi_max, m, n_phi=12001):
    phi = np.linspace(-phi_max, phi_max, n_phi)
    psi = np.cos((2*k-1)*np.pi*phi/(2.0*phi_max))
    nrm = np.trapezoid(psi**2, phi)
    return np.trapezoid(psi**2 * 0.5*m**2*phi**2, phi) / nrm


def make_rhs(lam, Vnn):
    def rhs(a, y):
        c, dc = y
        w2 = a**4 * Vnn - a**2 + lam / a**2
        return [dc, -w2 * c]
    return rhs


def two_solutions(lam, Vnn, a_min, a_max, n_a):
    """Real ICs: u=(1,0), v=(0,1)."""
    rhs = make_rhs(lam, Vnn)
    t = np.geomspace(a_min, a_max, n_a)
    su = solve_ivp(rhs, (a_min, a_max), [1.0, 0.0], t_eval=t,
                   method='DOP853', rtol=1e-12, atol=1e-15)
    sv = solve_ivp(rhs, (a_min, a_max), [0.0, 1.0], t_eval=t,
                   method='DOP853', rtol=1e-12, atol=1e-15)
    return t, su.y, sv.y


def hh_ic(lam, a_min):
    """HH regularity: real at a_min."""
    if lam < 0.25:
        p = 0.5 * (1.0 + np.sqrt(1.0 - 4.0*lam))
        return a_min**p, p * a_min**(p-1)
    nu = np.sqrt(lam - 0.25)
    # standing wave: c = sqrt(a) cos(nu ln(a/a0)); choose a0=1
    L = np.log(a_min)
    c0  = a_min**0.5 * np.cos(nu*L)
    dc0 = 0.5*a_min**(-0.5)*np.cos(nu*L) - nu*a_min**(-0.5)*np.sin(nu*L)
    return c0, dc0


def vilenkin_ic(lam, a_min):
    """Vilenkin tunneling: positive-frequency complex mode."""
    if lam < 0.25:
        p = 0.5 * (1.0 + np.sqrt(1.0 - 4.0*lam))
        return a_min**p + 0j, p * a_min**(p-1) + 0j
    nu = np.sqrt(lam - 0.25)
    c0  = a_min**0.5 * np.exp(1j*nu*np.log(a_min))
    dc0 = (0.5 + 1j*nu) * a_min**(-0.5) * np.exp(1j*nu*np.log(a_min))
    return c0, dc0


def wkb_decompose(a, c, dc, lam, Vnn):
    """c = A e^{-iθ}/√(2ω) + B e^{+iθ}/√(2ω).  Returns (|B/A|², arg(B/A))."""
    w2 = a**4 * Vnn - a**2 + lam / a**2
    if w2 <= 0:
        return np.nan, np.nan
    w = np.sqrt(w2)
    A_eff = c + 1j * dc / w     # ∝ A
    B_eff = c - 1j * dc / w     # ∝ B
    if abs(A_eff) < 1e-300:
        return np.nan, np.nan
    return abs(B_eff/A_eff)**2, np.angle(B_eff/A_eff)


def solve_bc(kind, k, phi_max, m, a_min=0.01, a_max=10.0, n_a=4000):
    lam = ((2*k-1)*np.pi/(2.0*phi_max))**2
    Vnn = Vkk(k, phi_max, m)
    rhs = make_rhs(lam, Vnn)
    if kind == "HH":
        c0, dc0 = hh_ic(lam, a_min)
    else:
        c0, dc0 = vilenkin_ic(lam, a_min)
    t = np.geomspace(a_min, a_max, n_a)
    sol = solve_ivp(rhs, (a_min, a_max), [c0, dc0], t_eval=t,
                    method='DOP853', rtol=1e-12, atol=1e-15)
    return t, sol.y[0], sol.y[1], lam, Vnn


def main():
    header = (f"{'k':>3} {'m':>5} {'phi_max':>8} "
              f"{'lam':>8} {'Vnn':>8} "
              f"{'|B/A|^2_HH':>12} {'arg(B/A)_HH':>12} "
              f"{'|B/A|^2_Vil':>12} {'arg(B/A)_Vil':>12}")
    print(header)
    print("-" * len(header))
    for phi_max in (3.0, 5.0):
        for m in (0.5, 1.0, 2.0, 4.0):
            for k in (1, 2, 3, 4):
                try:
                    tH, cH, dcH, lam, Vnn = solve_bc("HH", k, phi_max, m)
                    tV, cV, dcV, _,   _   = solve_bc("Vil", k, phi_max, m)
                    rH, aH = wkb_decompose(tH[-1], cH[-1], dcH[-1], lam, Vnn)
                    rV, aV = wkb_decompose(tV[-1], cV[-1], dcV[-1], lam, Vnn)
                    print(f"{k:>3} {m:>5.2f} {phi_max:>8.2f} "
                          f"{lam:>8.4f} {Vnn:>8.4f} "
                          f"{rH:>12.4e} {aH:>12.4e} "
                          f"{rV:>12.4e} {aV:>12.4e}")
                except Exception as e:
                    print(f"ERR k={k} m={m} phi_max={phi_max}: {e}")


if __name__ == "__main__":
    main()
