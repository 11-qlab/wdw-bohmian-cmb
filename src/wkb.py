"""
Proper Bogoliubov extraction via WKB in/out decomposition.

For each mode n, at large a the equation is
    c_n'' + ω_n²(a) c_n = 0
with  ω_n²(a) = a⁴ V_nn - a² + λ_n/a²
      (from the CORRECT WDW sign, rewritten as +ω² form)

Wait -- sign check:
    c_n'' = a² c_n - λ_n/a² c_n - a⁴ V_nn c_n
=>  c_n'' + (a⁴ V_nn - a² + λ_n/a²) c_n = 0
=>  ω_n²(a) = a⁴ V_nn - a² + λ_n/a²

For m²φ²/2 with V_nn > 0, at large a the a⁴V term dominates
and the "frequency" is real => oscillatory => WKB is valid.

Decompose:
    c_n(a) = (α_n / √(2ω_n)) e^{-i∫ω da} + (β_n / √(2ω_n)) e^{+i∫ω da}
"""
import numpy as np


def omega_squared(a, lam_n, V_nn):
    return a**4 * V_nn - a**2 + lam_n / a**2


def extract_bogoliubov(c_n, a, lam_n, V_nn):
    """Return |beta_n|^2 for mode n. Requires the integral of omega
    over the last few points to build the phase."""
    omega = np.sqrt(np.maximum(omega_squared(a, lam_n, V_nn), 1e-30))

    # WKB phase from a[i] to a[-1]
    phase = np.zeros_like(a)
    for i in range(len(a) - 2, -1, -1):
        da = a[i+1] - a[i]
        phase[i] = phase[i+1] + 0.5 * (omega[i] + omega[i+1]) * da

    # c and c' at last two points
    c_end = c_n[-1]
    dc_end = (c_n[-1] - c_n[-2]) / (a[-1] - a[-2])
    w_end = omega[-1]

    # Fit to  A e^{-i phase} + B e^{+i phase}  with WKB amplitude
    # c(a) ≈ A e^{-i phase}/√w + B e^{+i phase}/√w
    # c'(a) ≈ A (-i w) e^{-i phase}/√w + B (i w) e^{+i phase}/√w
    ph = phase[-1]
    pref = 1.0 / np.sqrt(w_end)
    M = np.array([
        [pref * np.exp(-1j * ph),      pref * np.exp(1j * ph)],
        [-1j * w_end * pref * np.exp(-1j * ph),
         +1j * w_end * pref * np.exp(1j * ph)],
    ])
    rhs = np.array([c_end, dc_end])
    AB = np.linalg.solve(M, rhs)
    A, B = AB
    # B/A = beta/alpha convention (outgoing/incoming)
    if abs(A) < 1e-30:
        return np.nan
    return abs(B / A) ** 2
