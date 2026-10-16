"""Analyze batch results: fit R_n vs m."""
import json
import numpy as np


def main(path="results_batch.json"):
    with open(path) as f:
        data = json.load(f)
    res = [r for r in data["results"] if r["ok"]]
    print(f"loaded {len(res)} ok of {len(data['results'])}\n")

    print("Power-law fits  log(R_2) vs log(m):\n")
    print(f"{'phi_max':>8}  {'n_modes':>7}  {'slope':>10}  {'intercept':>10}  {'R^2':>8}")
    print("-" * 52)
    for phi_max in sorted({r["phi_max"] for r in res}):
        for n_modes in sorted({r["n_modes"] for r in res}):
            sub = [r for r in res
                   if r["phi_max"] == phi_max and r["n_modes"] == n_modes]
            if len(sub) < 2:
                continue
            m = np.array([r["m"] for r in sub])
            y = np.array([r["wkb_growth"][1] if len(r["wkb_growth"]) > 1
                          else np.nan for r in sub])
            good = np.isfinite(y) & (y > 0)
            if good.sum() < 2:
                continue
            x = np.log(m[good])
            yy = np.log(y[good])
            A = np.vstack([x, np.ones_like(x)]).T
            sol, *_ = np.linalg.lstsq(A, yy, rcond=None)
            slope, intercept = sol
            pred = A @ sol
            ss_res = np.sum((yy - pred)**2)
            ss_tot = np.sum((yy - yy.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            print(f"{phi_max:>8.2f}  {n_modes:>7d}  {slope:>+10.4f}  "
                  f"{intercept:>+10.4f}  {r2:>8.4f}")

    print("\nRaw R_n for n_modes=4:")
    print(f"\n{'m':>6}  {'phi_max':>8}  {'R_1':>12}  {'R_2':>12}  {'R_3':>12}  {'R_4':>12}")
    print("-" * 72)
    for r in res:
        if r["n_modes"] == 4:
            rr = r["wkb_growth"] + [np.nan] * (4 - len(r["wkb_growth"]))
            print(f"{r['m']:>6.2f}  {r['phi_max']:>8.2f}  "
                  + "  ".join(f"{x:>12.3e}" for x in rr[:4]))


if __name__ == "__main__":
    main()
