# The four-step pipeline — what it does and does not establish

## Step 1 — Perturbation-sector WDW

Born-Oppenheimer factorization: Ψ[a,φ,{v_k}] = Ψ_0[a,φ] · ∏_k ψ_k[v_k; a, φ].

Implemented: solve Ψ_0 in minisuperspace for HH and Vilenkin BCs, extract
the WKB phase difference Δθ_n = arg(B/A)_HH - arg(B/A)_Vil for n = 1..6.

Result: Δθ_n ∈ [-1.0, +4.7] rad for n ∈ [1,6] at φ_max = 3, m = 1.

This step is a genuine calculation.

## Step 2 — Mode-index → wavenumber

Map q_n = nπ/(2φ_max) → k_n = q_n √(2ε) H.

In natural units with H = 1, k_n = 2.34e-2 · n. After normalizing to
k_pivot = 0.05 Mpc^-1, k_n ∈ [0.05, 0.3] Mpc^-1.

This step is a modeling choice, not derived. The normalization to
k_pivot is imposed from observation.

## Step 3 — Quantum-to-classical transition

Assumed: squeezing preserves the WKB phase; Bogoliubov amplitude is
(H/M_Pl)^2. This is the standard Halliwell-Hawking result and it is
the ONLY place where the suppression enters.

**This step inserts the (H/M_Pl)^2 by hand.** It does not derive it.

## Step 4 — P_ζ(k)

ΔP_ζ/P_ζ = 2 (H/M_Pl)^2 cos(Δθ_n).

For H/M_Pl = 1e-5: ΔP/P ~ 2e-10.

Planck sensitivity at pivot: ~ 1e-3.

Ratio: 2e-7.

## What this establishes

The minisuperspace HH/Vilenkin phase difference (Step 1) does not evade
the standard (H/M_Pl)^2 suppression (Step 3) when propagated to P_ζ(k)
(Step 4).

## What this does NOT establish

A first-principles derivation of the (H/M_Pl)^2 suppression. That is a
result from the literature (Halliwell & Hawking 1985), not from this code.
