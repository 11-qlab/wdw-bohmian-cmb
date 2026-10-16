
## 8. Numerical result (final)

Using src/bogoliubov_complex.py with complex HH initial conditions:

|β/α|² values for phi_max = 3:

    m\n       1         2         3         4
    0.5    0.994     0.628     0.323     0.140
    1.0    0.870     0.252     0.089     0.031
    2.0    0.772     0.177     0.057     0.019
    4.0    0.740     0.157     0.049     0.016

Scaling laws:
  |β/α|² ~ n^{-2.4}   at fixed m = 1
  |β/α|² ~ m^{-1.0}   at fixed n = 4

Interpretation: minisuperspace WDW mode mixing under HH
boundary conditions is a power-law-suppressed effect, NOT
exponentially small. The m^{-1} scaling is weaker than the
(H/M_Pl)^2 suppression assumed in the original audit.

To convert |β/α|² to a CMB observable requires translating
grid variable a to physical e-folds N and matching at horizon
crossing. This is left as future work.
