import numpy as np
from src.wdw_solver import solve_wdw_bruteforce

for (na, np_) in [(40, 32), (60, 48), (80, 64), (120, 96), (160, 128)]:
    a, phi, Psi, mu = solve_wdw_bruteforce(n_a=na, n_phi=np_)
    frac = float((np.abs(Psi) > 1e-2 * np.abs(Psi).max()).mean())
    print(f"n_a={na:>4d}  n_phi={np_:>4d}  mu={mu:+.6e}  bulk-frac={frac:.3f}")
