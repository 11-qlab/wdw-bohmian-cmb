"""
Wheeler-DeWitt + Bohmian mechanics scaffolding.

Goal: compute scale-dependent running of n_s and correlated
primordial non-Gaussianity from the quantum potential Q.
"""
import numpy as np

HBAR = 1.0


def quantum_potential(psi, a, m=1.0, threshold=1e-6):
    """
    Stable Q(a) = -(hbar^2 / 2m) * (psi'' / psi).

    Uses the identity psi''/psi = (ln psi)'' + [(ln psi)']^2
    and masks regions where |psi| < threshold * max|psi|.
    """
    psi = np.asarray(psi, dtype=float)
    a = np.asarray(a, dtype=float)

    mask = np.abs(psi) > threshold * np.abs(psi).max()
    log_psi = np.full_like(psi, np.nan)
    log_psi[mask] = np.log(np.abs(psi[mask]))

    dlog = np.gradient(log_psi, a)
    d2log = np.gradient(dlog, a)

    ratio = d2log + dlog**2          # = psi''/psi
    Q = -(HBAR**2 / (2.0 * m)) * ratio
    return Q


def running_ns(k, ns0=0.965, alpha=-1e-3, k0=0.05):
    """Scale-dependent spectral index n_s(k)."""
    return ns0 + alpha * np.log(k / k0)


if __name__ == "__main__":
    a = np.linspace(0.01, 0.99, 2000)
    psi = np.exp(-((a - 0.5) ** 2) / 0.02)
    Q = quantum_potential(psi, a)

    finite = Q[np.isfinite(Q)]
    print(f"Q finite range: {finite.min():.4f} .. {finite.max():.4f}")
    print(f"masked points:  {np.isnan(Q).sum()} / {Q.size}")

    # analytic check at the peak region
    Q_analytic = (HBAR**2 / (2.0 * 0.02)) * (1 - ((a - 0.5) ** 2) / 0.02)
    good = np.isfinite(Q) & (np.abs(a - 0.5) < 0.1)
    print(f"max |Q_num - Q_analytic|: {np.max(np.abs(Q[good] - Q_analytic[good])):.3e}")
