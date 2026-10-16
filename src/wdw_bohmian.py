"""
Wheeler-DeWitt + Bohmian mechanics scaffolding.

Goal: compute scale-dependent running of n_s and correlated
primordial non-Gaussianity from the quantum potential Q.
"""
import numpy as np

HBAR = 1.0

def quantum_potential(psi, a):
    """
    Q(a) = -(hbar^2 / 2m) * (psi'' / psi)
    psi : array, wavefunction of scale factor a
    a   : array, scale factor values
    """
    dpsi  = np.gradient(psi, a)
    d2psi = np.gradient(dpsi, a)
    return -(HBAR**2 / 2.0) * (d2psi / psi)

def running_ns(k, ns0=0.965, alpha=-1e-3, k0=0.05):
    """Scale-dependent spectral index n_s(k)."""
    return ns0 + alpha * np.log(k / k0)

if __name__ == "__main__":
    a   = np.linspace(1e-3, 1.0, 1000)
    psi = np.exp(-((a - 0.5)**2) / 0.02)
    Q   = quantum_potential(psi, a)
    print("Q range:", Q.min(), Q.max())
