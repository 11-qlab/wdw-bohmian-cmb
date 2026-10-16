"""
cmb_bridge.py — Toy model bridging minisuperspace O(1) phase differences
to primordial power spectrum modifications P_zeta(k) and scalar spectral index n_s(k).
"""
import numpy as np

def primordial_power_spectrum(k, A_s=2.1e-9, n_s_base=0.965, k_0=0.05, 
                              epsilon=1e-5, phase_diff=1.23, frequency=8.0):
    """
    Computes P_zeta(k) with a quantum boundary correction term driven by 
    an O(1) phase difference (phase_diff) between HH and Vilenkin.
    
    Note: epsilon represents the amplitude of quantum gravitational corrections.
    True Planck-scale suppression is ~ (H/M_Pl)^2 ~ 10^-10, but exploratory 
    feature visibility can be tested at larger epsilon.
    """
    scalar_ratio = (k / k_0)**(n_s_base - 1.0)
    P_base = A_s * scalar_ratio
    
    # Quantum correction modulated by the O(1) phase difference
    arg = frequency * np.log(k / k_0) + phase_diff
    modulation = 1.0 + epsilon * np.cos(arg)
    
    return P_base * modulation

def compute_spectral_index(k_vals, P_vals):
    """Computes n_s(k) via numerical derivative d ln P / d ln k."""
    log_k = np.log(k_vals)
    log_P = np.log(P_vals)
    dlnP_dlnk = np.gradient(log_P, log_k)
    return 1.0 + dlnP_dlnk

def main():
    print("=== CMB Bridge Explorer: Minisuperspace Phase to n_s(k) ===")
    print("Mapping O(1) phase differences to primordial power spectrum features.\n")
    
    # Wavenumbers spanning observable CMB scales (Mpc^-1)
    k_vals = np.logspace(-4, -1, 200)
    
    # Representative O(1) phase differences found in minisuperspace modes (k >= 2)
    phases = [0.31, 0.89, 2.15, 3.59]
    
    header = f"{'k (Mpc^-1)':>12} | {'Phase (Δθ)':>10} | {'P_zeta':>14} | {'n_s(k)':>10}"
    print(header)
    print("-" * 55)
    
    for phi in phases:
        P_spectrum = primordial_power_spectrum(k_vals, phase_diff=phi, epsilon=1e-4)
        ns_spectrum = compute_spectral_index(k_vals, P_spectrum)
        
        # Display sample points across scales
        for idx in [20, 80, 140, 190]:
            k = k_vals[idx]
            print(f"{k:>12.4e} | {phi:>10.2f} | {P_spectrum[idx]:>14.4e} | {ns_spectrum[idx]:>10.4f}")
        print("-" * 55)

if __name__ == "__main__":
    main()
