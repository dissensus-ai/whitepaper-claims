#!/usr/bin/env python3
"""STEP 6 (Task 2c) -- like-for-like contamination test.

The paper contrasts a contaminated n=37 corpus with a clean n=43 corpus and
concludes that contamination "manufactured a spurious cross-sectional result".
Cleanliness, sample size and composition all move together in that contrast.

Design:
  L = (contaminated common set) INTERSECT (clean common set)
  All four legs use the SAME market statistics and the SAME CP factors, so the
  only thing that varies is (a) which claims matrix, (b) which asset set.

  A  contaminated, full  (n=37)   <- the paper's left panel
  B  contaminated, on L
  C  clean, on L
  D  clean, full         (n=43)   <- the paper's right panel

  A->B = composition effect (dropping the assets with no clean counterpart)
  B->C = PURE contamination effect, composition held fixed
  C->D = composition effect (adding the newly collected assets)
"""
import json
from pathlib import Path

import numpy as np

import common as C
from alignment.procrustes import ProcrustesAlignment
from alignment.congruence import CongruenceCoefficient

OUTF = Path(__file__).parent / "results_06_contamination.json"
_AL, _CC = ProcrustesAlignment(), CongruenceCoefficient()


def loo_impacts(claims, stats, symbols):
    """Verbatim src/analysis/cross_sectional.py::compute_entity_alignment."""
    n = len(symbols)
    r = _AL.align_matrices(claims, stats)
    phi_full = _CC.matrix_congruence(r['source_rotated'], r['target_centered'])['mean_phi']
    rows = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        rl = _AL.align_matrices(claims[m], stats[m])
        phi_loo = _CC.matrix_congruence(rl['source_rotated'], rl['target_centered'])['mean_phi']
        imp = float(phi_full - phi_loo)
        rows.append({'symbol': symbols[i], 'phi_without': float(phi_loo), 'impact': imp,
                     'interpretation': 'helps' if imp > 0.01 else 'hurts' if imp < -0.01 else 'neutral'})
    rows.sort(key=lambda x: x['impact'], reverse=True)
    return float(phi_full), rows


def leg(claims, csym, stats, ssym, fac, fsym, restrict, label):
    Cl, St, Fa, common = C.align_three(claims, csym, stats, ssym, fac, fsym, restrict)
    phi_full, rows = loo_impacts(Cl, St, common)
    helpers = [r['symbol'] for r in rows if r['interpretation'] == 'helps']
    hurters = [r['symbol'] for r in rows if r['interpretation'] == 'hurts']
    out = {
        'label': label, 'n': len(common), 'symbols': common,
        'padded_phi': float(C.padded_phi(Cl, St)),
        'padded_p': float(C.padded_perm_p(Cl, St)),
        'matched_phi': float(C.matched_phi(Cl, St, 7)),
        'matched_p': float(C.matched_perm_p(Cl, St, 7)),
        'loo_phi_full': phi_full,
        'loo_top5': rows[:5], 'loo_bottom5': rows[-5:],
        'helpers': helpers, 'hurters': hurters,
        'n_helpers': len(helpers), 'n_hurters': len(hurters),
        'max_impact': float(max(r['impact'] for r in rows)),
        'min_impact': float(min(r['impact'] for r in rows)),
        'loo_all': rows,
    }
    print(f"[{label:26s}] n={out['n']:2d} padded={out['padded_phi']:.4f}(p={out['padded_p']:.3f}) "
          f"matched={out['matched_phi']:.4f}(p={out['matched_p']:.3f}) "
          f"helpers={helpers} max_impact={out['max_impact']:+.4f}")
    return out


def main():
    z, ssym, _ = C.build_stats_matrix('base')
    fac, fsym = C.load_factors()
    cl_clean, sym_clean = C.load_claims(clean=True)
    cl_dirty, sym_dirty = C.load_claims(clean=False)

    common_clean = sorted(set(sym_clean) & set(ssym) & set(fsym))
    common_dirty = sorted(set(sym_dirty) & set(ssym) & set(fsym))
    L = sorted(set(common_clean) & set(common_dirty))

    res = {
        'clean_common': common_clean, 'dirty_common': common_dirty,
        'L': L, 'n_L': len(L),
        'only_in_dirty': sorted(set(common_dirty) - set(L)),
        'only_in_clean': sorted(set(common_clean) - set(L)),
    }
    print(f"contaminated common n={len(common_dirty)}, clean common n={len(common_clean)}, "
          f"like-for-like L n={len(L)}")
    print(f"  only in contaminated: {res['only_in_dirty']}")
    print(f"  only in clean       : {res['only_in_clean']}\n")

    # which rows of the claims matrix actually changed on L?
    di = {s: cl_dirty[sym_dirty.index(s)] for s in L}
    ci = {s: cl_clean[sym_clean.index(s)] for s in L}
    changed = {s: float(np.abs(di[s] - ci[s]).max()) for s in L}
    res['row_max_abs_change'] = changed
    res['rows_changed'] = sorted([s for s, v in changed.items() if v > 1e-9])
    res['rows_identical'] = sorted([s for s, v in changed.items() if v <= 1e-9])
    print(f"claims rows that CHANGED on L ({len(res['rows_changed'])}): {res['rows_changed']}")
    print(f"claims rows identical on L ({len(res['rows_identical'])})\n")

    res['legs'] = {
        'A_contaminated_full': leg(cl_dirty, sym_dirty, z, ssym, fac, fsym, None,
                                   'A contaminated full'),
        'B_contaminated_on_L': leg(cl_dirty, sym_dirty, z, ssym, fac, fsym, L,
                                   'B contaminated on L'),
        'C_clean_on_L':        leg(cl_clean, sym_clean, z, ssym, fac, fsym, L,
                                   'C clean on L'),
        'D_clean_full':        leg(cl_clean, sym_clean, z, ssym, fac, fsym, None,
                                   'D clean full'),
    }

    A, B, Cc, D = (res['legs'][k] for k in
                   ['A_contaminated_full', 'B_contaminated_on_L', 'C_clean_on_L', 'D_clean_full'])
    res['decomposition'] = {
        'padded_phi': {'A': A['padded_phi'], 'B': B['padded_phi'],
                       'C': Cc['padded_phi'], 'D': D['padded_phi'],
                       'composition_A_to_B': B['padded_phi'] - A['padded_phi'],
                       'contamination_B_to_C': Cc['padded_phi'] - B['padded_phi'],
                       'composition_C_to_D': D['padded_phi'] - Cc['padded_phi']},
        'n_helpers': {'A': A['n_helpers'], 'B': B['n_helpers'],
                      'C': Cc['n_helpers'], 'D': D['n_helpers']},
        'max_impact': {'A': A['max_impact'], 'B': B['max_impact'],
                       'C': Cc['max_impact'], 'D': D['max_impact']},
    }
    print("\nDecomposition of the zero-padded phi:")
    d = res['decomposition']['padded_phi']
    print(f"  A contaminated n={A['n']}  phi={d['A']:.4f}")
    print(f"  B contaminated n={B['n']}  phi={d['B']:.4f}   (composition {d['composition_A_to_B']:+.4f})")
    print(f"  C clean        n={Cc['n']}  phi={d['C']:.4f}   (CONTAMINATION {d['contamination_B_to_C']:+.4f})")
    print(f"  D clean        n={D['n']}  phi={d['D']:.4f}   (composition {d['composition_C_to_D']:+.4f})")
    print(f"\n  helpers: A={A['n_helpers']} B={B['n_helpers']} C={Cc['n_helpers']} D={D['n_helpers']}")
    print(f"  max LOO impact: A={A['max_impact']:+.4f} B={B['max_impact']:+.4f} "
          f"C={Cc['max_impact']:+.4f} D={D['max_impact']:+.4f}")

    # rank correlation of LOO impacts between B and C (same assets)
    imB = {r['symbol']: r['impact'] for r in B['loo_all']}
    imC = {r['symbol']: r['impact'] for r in Cc['loo_all']}
    from scipy.stats import spearmanr, pearsonr
    xs = [imB[s] for s in L]; ys = [imC[s] for s in L]
    res['loo_agreement_B_vs_C'] = {
        'spearman': float(spearmanr(xs, ys).statistic),
        'pearson': float(pearsonr(xs, ys)[0]),
    }
    print(f"\n  LOO-impact agreement B vs C (same {len(L)} assets): "
          f"Spearman={res['loo_agreement_B_vs_C']['spearman']:.3f}, "
          f"Pearson={res['loo_agreement_B_vs_C']['pearson']:.3f}")

    json.dump(res, open(OUTF, 'w'), indent=2)
    print(f"\nwrote {OUTF}")


if __name__ == '__main__':
    main()
