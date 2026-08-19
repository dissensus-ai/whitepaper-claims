#!/usr/bin/env python3
"""STEP 4b -- permutation test using the FULL numerical congruence-maximising
rotation on both observed and permuted data (the Brokken-start test in s04 is
consistent but its optimiser is weaker; this checks the conclusion does not
depend on that)."""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

import common as C

OUTF = Path(__file__).parent / "results_04b_congmax_perm.json"
N_PERM = 200
N_RESTART = 6


def cmax(A, B, use_abs, seed):
    A, B = C.center(A), C.center(B)
    A, B = C.pad_to(A, B)
    _q, c = C.congruence_max_rotation(A, B, use_abs=use_abs,
                                      n_restart=N_RESTART, seed=seed)
    return c


def main():
    claims, csym = C.load_claims(clean=True)
    fac, fsym = C.load_factors()
    z, syms, _ = C.build_stats_matrix('base')
    Cl, St, Fa, common = C.align_three(claims, csym, z, syms, fac, fsym)
    from alignment.matched_dimension_analysis import reduce_dimensions
    Cl7 = reduce_dimensions(Cl, 7)

    res = {}
    for name, (A, B, ua) in {
        'matched_claims_vs_stats': (Cl7, St, False),
        'informative_padded_claims_vs_stats': (Cl, St, True),
    }.items():
        obs = cmax(A, B, ua, seed=0)
        rng = np.random.default_rng(42)
        n = A.shape[0]
        null = np.array([cmax(A, B[rng.permutation(n)], ua, seed=i)
                         for i in range(N_PERM)])
        p = (float(np.mean(null >= obs)) if ua
             else float(np.mean(np.abs(null) >= abs(obs))))
        res[name] = {'phi_congmax': float(obs), 'perm_p': p,
                     'null_mean': float(null.mean()),
                     'null_q95': float(np.quantile(null, 0.95)),
                     'n_perm': N_PERM}
        print(f"{name:36s} phi_max={obs:.4f}  p={p:.3f}  "
              f"null mean={null.mean():.4f} q95={np.quantile(null,0.95):.4f}")

    json.dump(res, open(OUTF, 'w'), indent=2)
    print(f"wrote {OUTF}")


if __name__ == '__main__':
    main()
