import json, itertools, time
import numpy as np
from src.bogoliubov import solve_nmode, wkb_growth

GRID = {
    "m":       [0.5, 1.0, 2.0, 4.0],
    "phi_max": [1.0, 2.0, 3.0, 5.0],
    "n_modes": [2, 4, 6],
}


def run_one(m, phi_max, n_modes):
    try:
        a, c, lam, V, p, c0 = solve_nmode(
            a_min=0.05, a_max=4.0, n_a=300,
            phi_max=phi_max, n_modes=n_modes, m=m)
        R = wkb_growth(c, a, lam, m, phi_max, V)
        return {"m": m, "phi_max": phi_max, "n_modes": n_modes, "ok": True,
                "c0": np.abs(c0).tolist(),
                "c_end": np.abs(c[-1]).tolist(),
                "wkb_growth": R.tolist(),
                "lam": lam.tolist(), "p_plus": p.tolist()}
    except Exception as e:
        return {"m": m, "phi_max": phi_max, "n_modes": n_modes,
                "ok": False, "err": str(e)}


def main():
    keys = list(GRID.keys())
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    results, t0 = [], time.time()
    for i, combo in enumerate(combos):
        kw = dict(zip(keys, combo))
        r = run_one(**kw)
        results.append(r)
        if r["ok"]:
            R = r["wkb_growth"]
            print(f"[{i+1:>2}/{len(combos)}] ok  m={kw['m']:<4} "
                  f"phi_max={kw['phi_max']:<4} n={kw['n_modes']:<2}  "
                  + " ".join(f"{x:>9.2e}" for x in R), flush=True)
        else:
            print(f"[{i+1:>2}/{len(combos)}] ERR {kw}  {r['err']}", flush=True)
    with open("results_batch.json", "w") as f:
        json.dump({"grid": GRID, "results": results,
                   "dt": time.time() - t0}, f, indent=2)
    print(f"\ndone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
