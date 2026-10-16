"""Standalone driver for the HH-IC bogoliubov extraction."""
import numpy as np
from src.bogoliubov import solve_hh_coupled, extract_bogoliubov_single


def main():
    print(f"{'n':>3}  {'m':>5}  {'phi_max':>8}  {'V_nn':>8}  "
          f"{'lam':>8}  {'|β/α|²':>12}")
    print("-" * 58)
    for m in (0.5, 1.0, 2.0, 4.0):
        for phi_max in (2.0, 3.0, 5.0):
            try:
                a, c, lam, V = solve_hh_coupled(
                    phi_max=phi_max, n_modes=6, m=m, a_max=6.0)
                for k in range(4):
                    dc = np.gradient(c[:, k], a)
                    bsq = extract_bogoliubov_single(
                        a, c[:, k], dc, lam[k], V[k, k])
                    print(f"{k+1:>3}  {m:>5.2f}  {phi_max:>8.2f}  "
                          f"{V[k,k]:>8.3f}  {lam[k]:>8.4f}  "
                          f"{bsq:>12.4e}")
            except Exception as e:
                print(f"ERR m={m} phi_max={phi_max}: {e}")


if __name__ == "__main__":
    main()
