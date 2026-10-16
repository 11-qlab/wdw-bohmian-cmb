"""
Wheeler-DeWitt + Bohmian mechanics scaffolding.
"""
import numpy as np

HBAR = 1.0


def quantum_potential(psi, a, m=1.0, threshold=0.0):
    """
    Q(a) = -(hbar^2 / 2m) * (psi''/psi).

    Uses psi''/psi = (ln|psi|)'' + [(ln|psi|)']^2.
    Points with |psi| < threshold * max|psi| are set to NaN.

    edge_order=2 is required: with edge_order=1 the second
    derivative at the boundary is wrong by ~factor 2, which
    produces a spurious O(50) error in Q.
    """
    psi = np.asarray(psi, dtype=float)
    a = np.asarray(a, dtype=float)

    mask = (np.abs(psi) > threshold * np.abs(psi).max()) if threshold > 0 \
           else np.ones_like(psi, dtype=bool)

    log_psi = np.full_like(psi, np.nan)
    log_psi[mask] = np.log(np.abs(psi[mask]))

    dlog  = np.gradient(log_psi, a, edge_order=2)
    d2log = np.gradient(dlog,    a, edge_order=2)

    ratio = d2log + dlog**2
    return -(HBAR**2 / (2.0 * m)) * ratio


def running_ns(k, ns0=0.965, alpha=-1e-3, k0=0.05):
    return ns0 + alpha * np.log(k / k0)


if __name__ == "__main__":
    a = np.linspace(0.01, 0.99, 2000)
    sigma2 = 0.01
    psi = np.exp(-((a - 0.5) ** 2) / (2.0 * sigma2))
    Q = quantum_potential(psi, a)

    Q_analytic = (HBAR**2 / (2.0 * sigma2)) * (1.0 - ((a - 0.5) ** 2) / sigma2)

    finite = np.isfinite(Q)
    print(f"Q finite range: {Q[finite].min():.4f} .. {Q[finite].max():.4f}")
    print(f"max |Q_num - Q_analytic|: {np.max(np.abs(Q[finite] - Q_analytic[finite])):.3e}")
