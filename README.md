# WDW + Bohmian CMB — numerical exploration

## Status: CLOSED

This repository contains numerical explorations of mode mixing in
the Wheeler-DeWitt equation on minisuperspace, motivated by Bohmian
quantum cosmology.

## Verified components

- src/wdw_bohmian.py
    Quantum potential Q(a), verified against analytic Gaussian
    formula to 3×10⁻⁹.

- src/bogoliubov_complex.py
    N-mode WDW Cauchy solver with complex initial conditions.
    Wronskian conserved to machine precision.
    Result phase-IC-independent.
    But: result depends on the lower integration limit a_min,
    signalling genuine phase sensitivity of the reflection
    problem (barrier is not in the thick-barrier regime).

- src/verify.py
    Three verification tests: Wronskian, IC-phase independence,
    grid convergence. Tests 1 and 2 pass. Test 3 FAILS:
    |β/α|² varies with a_min from 0.14 to 0.36 at n=2, m=1.

## Conclusion

Mode mixing in minisuperspace WDW is a real, computable effect
with reflection coefficients of order 10⁻³ to 1. However, the
value depends on the WKB phase at a_min, which is fixed only by
imposing a physical boundary condition (Hartle-Hawking or
Vilenkin) on the perturbation modes.

Without imposing such a boundary condition, the result is
ambiguous and cannot be compared to CMB data.

## What would be needed to go further

1. Impose HH or Vilenkin BC on the WDW mode at a→0 with the
   Euclidean regularity condition. This fixes the phase.
2. Recompute |β/α|² with the physical phase.
3. Translate the a-grid to physical e-folds and match to
   Mukhanov-Sasaki mode functions.

Steps 1–3 are a substantial research project, not a numerical
tweak. They are outside the scope of this repository.

## Documents

- docs/audit.md          — first-pass suppression argument.
- docs/audit_refined.md  — refined audit: six mechanisms.
- results_final.txt      — final numerical output.
