import numpy as np
from src.wdw_bohmian import quantum_potential, running_ns


def test_gaussian_matches_analytic():
    a = np.linspace(0.01, 0.99, 2000)
    sigma2 = 0.01
    psi = np.exp(-((a - 0.5) ** 2) / (2.0 * sigma2))
    Q = quantum_potential(psi, a)
    Q_analytic = (1.0 / (2.0 * sigma2)) * (1.0 - ((a - 0.5) ** 2) / sigma2)
    good = np.isfinite(Q)
    assert np.max(np.abs(Q[good] - Q_analytic[good])) < 1e-8


def test_masking():
    a = np.linspace(0.0, 1.0, 500)
    psi = np.exp(-((a - 0.5) ** 2) / 1e-4)
    Q = quantum_potential(psi, a, threshold=1e-3)
    assert np.isnan(Q).any()
    assert np.isfinite(Q).any()


def test_running_ns_direction():
    k = np.logspace(-4, 0, 200)
    ns = running_ns(k, alpha=-1e-3)
    assert ns[0] > ns[-1]
