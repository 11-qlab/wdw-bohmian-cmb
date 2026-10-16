import numpy as np
from src.wdw_solver import solve_wdw_bruteforce

a, phi, Psi, mu = solve_wdw_bruteforce()
print(f"mu = {mu:.4e}")

i_a, i_p = np.unravel_index(np.argmax(np.abs(Psi)), Psi.shape)
print(f"peak at a={a[i_a]:.3f}, phi={phi[i_p]:.3f}")

i_mid = a.size // 2
prof = np.abs(Psi[i_mid])
print(f"\n|Psi(a={a[i_mid]:.2f}, phi)| at 8 phi values:")
for k in np.linspace(0, phi.size - 1, 8, dtype=int):
    print(f"  phi={phi[k]:+.2f}   |Psi|={prof[k]:.3e}")

j0 = int(np.argmin(np.abs(phi)))
prof_a = np.abs(Psi[:, j0])
print(f"\n|Psi(a, phi=0)| at 8 a values:")
for k in np.linspace(0, a.size - 1, 8, dtype=int):
    print(f"  a={a[k]:.2f}   |Psi|={prof_a[k]:.3e}")

for thr in (1e-2, 1e-4, 1e-8, 1e-12):
    frac = float((np.abs(Psi) > thr * np.abs(Psi).max()).mean())
    print(f"fraction |Psi| > {thr:>6.0e}*max:  {frac:.3f}")
