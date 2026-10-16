"""Verification tests for the Bogoliubov extraction."""
import numpy as np
from src.bogoliubov_complex import solve_mode, extract_bogoliubov


def wronskian(a, c, dc):
    return c.conj() * dc - c * dc.conj()


def run(n=2, m=1.0, phi_max=3.0, phase=np.pi/4,
        a_min=0.05, a_max=8.0, n_a=3000):
    lam = (n * np.pi / (2.0 * phi_max)) ** 2
    Vnn = None
    # reuse solve_mode but override phase via monkey
    import src.bogoliubov_complex as mod
    # hack: solve_mode accepts phase? check source
    a, c, dc, lam, Vnn = solve_mode(n, phi_max, m,
                                    a_min=a_min, a_max=a_max,
                                    n_a=n_a, phase=phase)
    W = wronskian(a, c, dc)
    bsq, _ = extract_bogoliubov(a, c, dc, lam, Vnn)
    return a, c, dc, W, bsq


def main():
    print("=== Test 1: Wronskian conservation ===")
    a, c, dc, W, bsq = run()
    Wreal = np.abs(W.imag) / max(np.abs(W.imag).max(), 1e-30)
    Wimag = np.abs(W.real) / max(np.abs(W.real).max(), 1e-30)
    print(f"|Re W| / max|W|: {Wimag.max():.3e}")
    print(f"|Im W| / max|W|: {Wreal.max():.3e}")
    print(f"|W(a_min)|: {abs(W[0]):.3e}")
    print(f"|W(a_max)|: {abs(W[-1]):.3e}")
    print("If |W(a_min)| ~ 0 -> IC is Wronskian-degenerate (unphysical)")
    print(f"|beta/alpha|^2 = {bsq:.4e}\n")

    print("=== Test 2: phase dependence ===")
    for ph in (np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2):
        _, _, _, _, b = run(phase=ph)
        print(f"phase={ph:.3f}  |beta/alpha|^2 = {b:.4e}")
    print("If these differ -> IC artifact, not physics\n")

    print("=== Test 3: grid independence ===")
    for (amin, amax, na) in [(0.05, 8, 3000), (0.02, 8, 3000),
                              (0.05, 12, 4000), (0.01, 8, 4000),
                              (0.05, 8, 6000)]:
        try:
            _, _, _, W, b = run(a_min=amin, a_max=amax, n_a=na)
            print(f"a_min={amin}  a_max={amax}  n_a={na}  "
                  f"|W(0)|={abs(W[0]):.2e}  |beta/alpha|^2={b:.4e}")
        except Exception as e:
            print(f"a_min={amin} a_max={amax} n_a={na}  ERR {e}")
    print("If these differ -> not converged\n")


if __name__ == "__main__":
    main()
