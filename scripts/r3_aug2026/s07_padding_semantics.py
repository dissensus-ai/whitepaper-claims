#!/usr/bin/env python3
"""STEP 7 (Task 3) -- what the 'zero-padded' estimator actually computes.

Three different implementations of "Tucker phi with zero padding" exist in the
repo and they do NOT agree:

  (i)   src/alignment/congruence.py:46-52 + :74-79
        tuckers_phi returns 0.0 when the denominator is 0, and matrix_congruence
        averages over ALL max(dA,dB) columns  -> padded columns ARE counted as
        zeros -> DEFLATED.  This is the headline 0.223 estimator.
  (ii)  src/alignment/stability_analysis.py:159-184 (tucker_phi_padded)
        skips columns with denom == 0                         -> NOT deflated.
  (iii) src/alignment/reviewer3_additions.py:20-45 (tucker_phi)
        also skips columns with denom == 0                    -> NOT deflated.

This script establishes, numerically:
  1. which matrix is padded in every real call path;
  2. that (i) equals (k/D) x (the informative-column mean), exactly;
  3. that the permutation p-value is invariant to that rescaling;
  4. that the estimator is NOT symmetric in its arguments;
  5. which published numbers sit on which scale.
"""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

import common as C

OUTF = Path(__file__).parent / "results_07_padding.json"


def informative_mean_abs(A, B):
    """mean |phi| over the columns of the padded target that are NOT zero."""
    A, B = C.center(A), C.center(B)
    Ap, Bp = C.pad_to(A, B)
    Q, _ = orthogonal_procrustes(Ap, Bp)
    Ar = Ap @ Q
    vals = []
    for j in range(Bp.shape[1]):
        den = np.sqrt((Ar[:, j] @ Ar[:, j]) * (Bp[:, j] @ Bp[:, j]))
        if den > 0:
            vals.append(abs(float(Ar[:, j] @ Bp[:, j]) / den))
    return float(np.mean(vals)), len(vals), Bp.shape[1]


def main():
    z, ssym, _ = C.build_stats_matrix('base')
    claims, csym = C.load_claims(True)
    fac, fsym = C.load_factors()
    Cl, St, Fa, common = C.align_three(claims, csym, z, ssym, fac, fsym)

    res = {'n': len(common), 'pairs': {}}
    pairs = {'claims_vs_stats': (Cl, St), 'claims_vs_factors': (Cl, Fa),
             'stats_vs_factors': (St, Fa)}

    for name, (A, B) in pairs.items():
        padded = C.padded_phi(A, B)
        inf_mean, k, D = informative_mean_abs(A, B)
        pred = inf_mean * k / D
        # asymmetry: swap the argument order so the WIDER matrix is the target
        padded_swapped = C.padded_phi(B, A)
        inf_sw, k_sw, D_sw = informative_mean_abs(B, A)
        res['pairs'][name] = {
            'source_cols': A.shape[1], 'target_cols': B.shape[1],
            'padded_matrix': 'target (B)' if B.shape[1] < A.shape[1] else 'source (A)',
            'n_informative_cols': k, 'n_total_slots': D,
            'headline_padded_phi': float(padded),
            'informative_col_mean_abs_phi': inf_mean,
            'predicted_padded = inf_mean * k/D': float(pred),
            'exact_match': bool(abs(padded - pred) < 1e-12),
            'deflation_factor_k_over_D': k / D,
            'padded_phi_arguments_swapped': float(padded_swapped),
            'informative_mean_arguments_swapped': inf_sw,
            'perm_p_padded': float(C.padded_perm_p(A, B)),
        }
        r = res['pairs'][name]
        print(f"{name:20s} padded={padded:.4f}  informative-col mean|phi|={inf_mean:.4f} "
              f"x {k}/{D} = {pred:.4f}  exact={r['exact_match']}   "
              f"swapped order -> {padded_swapped:.4f} (informative {inf_sw:.4f})")

    # p-value invariance under the deflation rescaling ------------------------
    A, B = Cl, St
    from alignment.congruence import CongruenceCoefficient
    cc = CongruenceCoefficient()
    rng = np.random.default_rng(42)
    n = A.shape[0]
    obs_pad = cc.matrix_congruence(A, B)['mean_phi']
    obs_inf, k, D = informative_mean_abs(A, B)
    null_pad, null_inf = [], []
    for _ in range(1000):
        Bp = B[rng.permutation(n)]
        null_pad.append(cc.matrix_congruence(A, Bp)['mean_phi'])
        null_inf.append(informative_mean_abs(A, Bp)[0])
    p_pad = float(np.mean(np.array(null_pad) >= obs_pad))
    p_inf = float(np.mean(np.array(null_inf) >= obs_inf))
    res['p_value_invariance'] = {'p_padded': p_pad, 'p_informative': p_inf,
                                 'identical': p_pad == p_inf}
    print(f"\npermutation p, deflated scale = {p_pad:.4f}; undeflated scale = {p_inf:.4f} "
          f"(identical: {p_pad == p_inf})")

    # which published numbers live on which scale -----------------------------
    def r3_phi(A, B):
        A, B = C.center(A), C.center(B)
        A, B = C.pad_to(A, B)
        Q, _ = orthogonal_procrustes(A, B)
        Ar = A @ Q
        v = []
        for j in range(Ar.shape[1]):
            den = np.sqrt((Ar[:, j] @ Ar[:, j]) * (B[:, j] @ B[:, j]))
            if den > 0:
                v.append(float(Ar[:, j] @ B[:, j]) / den)
        return float(np.mean(v))

    res['scale_audit'] = {
        'congruence_py_deflated_claims_stats': float(C.padded_phi(Cl, St)),
        'skipzeros_signed_claims_stats (reviewer3/stability scale)': r3_phi(Cl, St),
        'matched_dimension_claims_stats': float(C.matched_phi(Cl, St, 7)),
        'cross_sectional_phi_full_published': json.load(
            open(C.OUT / 'analysis' / 'cross_sectional_analysis.json'))['phi_full'],
        'reviewer3_partial_published': json.load(
            open(C.OUT / 'alignment' / 'reviewer3_additions.json')
        )['partial_correlations']['comparisons'][1]['phi_partial'],
        'reviewer3_raw_published': json.load(
            open(C.OUT / 'alignment' / 'reviewer3_additions.json')
        )['partial_correlations']['comparisons'][1]['phi_raw'],
        'stability_jackknife_published_n': json.load(
            open(C.OUT / 'alignment' / 'stability_analysis.json')
        )['jackknife_stability']['n_assets'],
        'stability_jackknife_published_phi_full': json.load(
            open(C.OUT / 'alignment' / 'stability_analysis.json')
        )['jackknife_stability']['phi_full'],
    }
    for k_, v in res['scale_audit'].items():
        print(f"  {k_:60s} {v}")

    json.dump(res, open(OUTF, 'w'), indent=2)
    print(f"\nwrote {OUTF}")


if __name__ == '__main__':
    main()
