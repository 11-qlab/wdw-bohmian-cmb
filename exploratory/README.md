# Exploratory scripts — NOT part of the main result

These scripts were tried and found not to establish a physical
connection between the minisuperspace WDW calculation and CMB
observables. Kept for transparency.

- cmb_bridge.py      Toy ansatz relating phase to n_s(k).
                     Falsified by verify_bridge.py: the phase
                     enters at 1e-4 level, output is a pure power
                     law with R^2 = 1.000000.

- verify_bridge.py   Four tests establishing the bridge is a
                     numerical null. Included so no one re-derives it.

- proof_verify.py    Demonstration that HH IC produces B = A*
                     under the WKB split. This is a tautology
                     (any real solution of a real ODE has this
                     property), not a physics result.

None of these should be cited or built on.
