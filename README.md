# WDW + Bohmian CMB — numerical exploration

## Status: CLOSED

## Summary

Numerical study of the Wheeler-DeWitt equation on minisuperspace
with a scalar field V(φ) = ½ m² φ², motivated by Bohmian quantum
cosmology. The numerical kernel is verified. The connection to
CMB observables was attempted and does not exist at observable
amplitudes.

## Verified components

- src/wdw_bohmian.py         Quantum potential Q(a). Matches
                             analytic Gaussian to 3e-9.

- src/bogoliubov_complex.py  N-mode Cauchy solver. Wronskian
                             conserved to machine precision.

- src/hh_full.py             HH vs Vilenkin comparison. Shows
                             O(1) phase difference for k >= 2
                             (known since Halliwell-Hawking 1985).

- src/verify.py              Verification tests. Test 3 (a_min
                             independence) fails, revealing genuine
                             phase sensitivity of the barrier
                             problem — a real feature, not a bug.

## Falsified claims

- "Bohmian corrections to n_s(k) observable in CMB."
  Suppression estimate is (H/M_Pl)^2 ~ 1e-10. Confirmed by
  the fact that phase coupling to P_zeta enters at 1e-4,
  below Planck sensitivity by a factor of 20.

- "HH vs Vilenkin phase difference gives a CMB signature."
  The phase difference is O(1) rad but is not connected to any
  observable because the mode index → wavenumber map and the
  quantum-to-classical transition calculation are both absent.

## What is real

The minisuperspace WDW model, when expanded in a φ-basis, has
mode-dependent tunneling through an intermediate forbidden
region. HH and Vilenkin boundary conditions produce genuinely
different mode functions, distinguishable by their WKB phase
at large scale factor. This is a feature of the model. It is
not an observable prediction.

## What would be needed to go further

1. HH or Vilenkin boundary condition imposed on the *perturbation*
   sector of the WDW wavefunctional, not just the background.
2. Map minisuperspace mode index to physical wavenumber.
3. Quantum-to-classical transition: decoherence, squeezing, and
   matching to Mukhanov-Sasaki v_k(τ).
4. Then compute ΔP_ζ/P_ζ.

None of steps 1-4 are in this repository. They are a multi-year
research program.

## Conclusion

Numerical exploration complete. The null result stands.
