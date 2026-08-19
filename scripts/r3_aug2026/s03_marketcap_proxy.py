#!/usr/bin/env python3
"""STEP 3 -- the market-cap-proxy robustness check, re-run with USD volume.

Mirrors src/alignment/reviewer3_additions.py::partial_correlation_analysis
(line 304: `market_cap_proxy = stats_aligned[:, 4]  # avg_volume`) but lets the
proxy be either the published base-asset log volume or the USD-notional one.

NOTE the estimator used there (reviewer3_additions.py:20-45 `tucker_phi`) is a
THIRD variant: it zero-pads and then SKIPS zero columns, so it equals the mean
signed congruence over the target's real columns.  Reported alongside the two
headline estimators so the numbers are comparable.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats as sps
from scipy.linalg import orthogonal_procrustes

import common as C

OUTF = Path(__file__).parent / "results_03_marketcap.json"


def r3_tucker_phi(A, B):
    """Verbatim reviewer3_additions.py::tucker_phi (pads, then SKIPS zero cols)."""
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    A, B = C.pad_to(A, B)
    if A.shape[0] < 2:
        return np.nan
    R, _ = orthogonal_procrustes(A, B)
    A_rot = A @ R
    phis = []
    for j in range(A_rot.shape[1]):
        num = np.dot(A_rot[:, j], B[:, j])
        den = np.sqrt(np.dot(A_rot[:, j], A_rot[:, j]) * np.dot(B[:, j], B[:, j]))
        if den > 0:
            phis.append(num / den)
    return float(np.mean(phis)) if phis else 0.0


def residualize(X, control):
    Xr = np.zeros_like(X)
    for j in range(X.shape[1]):
        s, i, *_ = sps.linregress(control, X[:, j])
        Xr[:, j] = X[:, j] - (s * control + i)
    return Xr


def main():
    claims, csym = C.load_claims(clean=True)
    fac, fsym = C.load_factors()
    res = {}

    for vname, kw in [('base', dict(volume_basis='base')),
                      ('usd_close', dict(volume_basis='usd_close'))]:
        z, syms, raw = C.build_stats_matrix(**kw)
        Cl, St, Fa, common = C.align_three(claims, csym, z, syms, fac, fsym)

        proxy = St[:, 4]                       # avg_volume column
        proxy = (proxy - proxy.mean()) / (proxy.std() + 1e-10)

        Clr, Str, Far = (residualize(M, proxy) for M in (Cl, St, Fa))

        entry = {'n': len(common), 'comparisons': {}}
        spec = {
            'claims_vs_factors': (Cl, Fa, Clr, Far, 2),
            'claims_vs_stats':   (Cl, St, Clr, Str, 7),
            'stats_vs_factors':  (St, Fa, Str, Far, 2),
        }
        for name, (Xr, Yr, Xp, Yp, tgt) in spec.items():
            entry['comparisons'][name] = {
                'r3_phi_raw': r3_tucker_phi(Xr, Yr),
                'r3_phi_partial': r3_tucker_phi(Xp, Yp),
                'matched_phi_raw': float(C.matched_phi(Xr, Yr, tgt)),
                'matched_phi_partial': float(C.matched_phi(Xp, Yp, tgt)),
                'matched_p_partial': float(C.matched_perm_p(Xp, Yp, tgt)),
                'padded_phi_raw': float(C.padded_phi(Xr, Yr)),
                'padded_phi_partial': float(C.padded_phi(Xp, Yp)),
                'padded_p_partial': float(C.padded_perm_p(Xp, Yp)),
            }
            c = entry['comparisons'][name]
            print(f"[{vname:9s}] {name:20s} r3 {c['r3_phi_raw']:.3f}->{c['r3_phi_partial']:.3f}  "
                  f"matched {c['matched_phi_raw']:.3f}->{c['matched_phi_partial']:.3f} "
                  f"(p={c['matched_p_partial']:.3f})  "
                  f"padded {c['padded_phi_raw']:.3f}->{c['padded_phi_partial']:.3f} "
                  f"(p={c['padded_p_partial']:.3f})")
        res[vname] = entry
        print()

    json.dump(res, open(OUTF, 'w'), indent=2)
    print(f"wrote {OUTF}")


if __name__ == '__main__':
    main()
