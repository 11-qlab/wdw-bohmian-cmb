import numpy as np
import scipy.sparse.linalg as spla
from src.wdw_solver import build_operator, apply_dirichlet

a = np.linspace(0.3, 4.0, 80)
phi = np.linspace(-3.0, 3.0, 64)

L = build_operator(a, phi)
L = apply_dirichlet(L, a.size, phi.size)

vals = spla.eigsh(L, k=6, sigma=0.0, which="LM", return_eigenvectors=False)
print("6 eigenvalues closest to 0:")
for v in np.sort(vals):
    print(f"  {v.real:+.6e}")
