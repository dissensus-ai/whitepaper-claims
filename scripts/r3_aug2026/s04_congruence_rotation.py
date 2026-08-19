#!/usr/bin/env python3
"""STEP 4 (Task 2a) -- Procrustes (Frobenius) vs a congruence-MAXIMISING rotation.

The paper (main-blinded.tex:1180) asserts that 0.303 "is the congruence attained
at the rotation that best aligns the two spaces and cannot be raised by any
other rotation".  Orthogonal Procrustes minimises ||AQ - B||_F; it does not
maximise mean column congruence.  We quantify the gap, and -- crucially -- we
re-run the permutation test USING the congruence-maximising estimator, because a
higher point estimate is meaningless if the permutation null rises with it.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

import common as C

OUTF = Path(__file__).parent / "results_04_congruence_rotation.json"
N_PERM = 500


def frob_phi(A, B, use_abs):
    A, B = C.center(A), C.center(B)
    A, B = C.pad_to(A, B)
    Q, _ = orthogonal_procrustes(A, B)
    return C._colwise_congruence(A @ Q, B, use_abs=use_abs)


def cong_phi(A, B, use_abs, n_restart, seed=0):
    A, B = C.center(A), C.center(B)
    A, B = C.pad_to(A, B)
    _Q, c = C.congruence_max_rotation(A, B, use_abs=use_abs,
                                      n_restart=n_restart, seed=seed)
    return c


def brok_phi(A, B, use_abs):
    A, B = C.center(A), C.center(B)
    A, B = C.pad_to(A, B)
    Q = C.brokken_rotation(A, B)
    return C._colwise_congruence(A @ Q, B, use_abs=use_abs)


def main():
    claims, csym = C.load_claims(clean=True)
    fac, fsym = C.load_factors()
    z, syms, _ = C.build_stats_matrix('base')
    Cl, St, Fa, common = C.align_three(claims, csym, z, syms, fac, fsym)

    from alignment.matched_dimension_analysis import reduce_dimensions
    Cl7 = reduce_dimensions(Cl, 7)      # the PRIMARY matched-dimension geometry
    Cl2 = reduce_dimensions(Cl, 2)
    St2 = reduce_dimensions(St, 2)

    cases = {
        # matched-dimension geometry, mean SIGNED phi (paper's primary statistic)
        'matched_claims_vs_stats':   dict(A=Cl7, B=St, use_abs=False),
        'matched_claims_vs_factors': dict(A=Cl2, B=Fa, use_abs=False),
        'matched_stats_vs_factors':  dict(A=St2, B=Fa, use_abs=False),
        # zero-padded geometry, mean ABSOLUTE phi (paper's display-floor statistic)
        'padded_claims_vs_stats':    dict(A=Cl,  B=St, use_abs=True),
        'padded_claims_vs_factors':  dict(A=Cl,  B=Fa, use_abs=True),
        'padded_stats_vs_factors':   dict(A=St,  B=Fa, use_abs=True),
    }

    res = {'n': len(common), 'n_perm': N_PERM, 'cases': {}}
    for name, cfg in cases.items():
        A, B, ua = cfg['A'], cfg['B'], cfg['use_abs']
        t0 = time.time()
        p_frob = frob_phi(A, B, ua)
        p_brok = brok_phi(A, B, ua)
        p_max = cong_phi(A, B, ua, n_restart=40)
        # permutation null under BOTH estimators (Brokken start only for speed;
        # verified to match the full multi-restart optimum on the observed data)
        rng = np.random.default_rng(42)
        n = A.shape[0]
        null_f, null_c = [], []
        for _ in range(N_PERM):
            idx = rng.permutation(n)
            Bp = B[idx]
            null_f.append(frob_phi(A, Bp, ua))
            null_c.append(brok_phi(A, Bp, ua))
        null_f, null_c = np.array(null_f), np.array(null_c)
        # one-sided for the |.| statistic (as congruence.py), two-sided for the
        # signed statistic (as matched_dimension_analysis.py)
        if ua:
            p_f = float(np.mean(null_f >= p_frob))
            p_c = float(np.mean(null_c >= p_brok))
        else:
            p_f = float(np.mean(np.abs(null_f) >= abs(p_frob)))
            p_c = float(np.mean(np.abs(null_c) >= abs(p_brok)))

        res['cases'][name] = {
            'phi_frobenius': float(p_frob),
            'phi_brokken': float(p_brok),
            'phi_congruence_max_40restarts': float(p_max),
            'lift_brokken': float(p_brok - p_frob),
            'lift_max': float(p_max - p_frob),
            'perm_p_frobenius': p_f,
            'perm_p_brokken': p_c,
            'null_mean_frobenius': float(null_f.mean()),
            'null_mean_brokken': float(null_c.mean()),
            'null_q95_frobenius': float(np.quantile(np.abs(null_f), 0.95)),
            'null_q95_brokken': float(np.quantile(np.abs(null_c), 0.95)),
            'seconds': round(time.time() - t0, 1),
        }
        r = res['cases'][name]
        print(f"{name:28s} frob={p_frob:.4f}  brokken={p_brok:.4f}  "
              f"max={p_max:.4f}  (lift {r['lift_max']:+.4f})   "
              f"p: {p_f:.3f} -> {p_c:.3f}   "
              f"null mean {null_f.mean():.4f} -> {null_c.mean():.4f}  "
              f"[{r['seconds']}s]")

    json.dump(res, open(OUTF, 'w'), indent=2)
    print(f"\nwrote {OUTF}")


if __name__ == '__main__':
    main()
