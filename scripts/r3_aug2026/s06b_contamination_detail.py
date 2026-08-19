#!/usr/bin/env python3
"""STEP 6b -- which part of the contaminated->clean move kills the 'helpers'?

Drops assets one group at a time from the CONTAMINATED corpus without changing
any document, then swaps in the re-collected documents.
"""
import json
from pathlib import Path

import numpy as np
import common as C
from s06_contamination_likeforlike import loo_impacts

OUTF = Path(__file__).parent / "results_06b_contamination_detail.json"


def run(claims, csym, z, ssym, fac, fsym, restrict, label):
    Cl, St, Fa, common = C.align_three(claims, csym, z, ssym, fac, fsym, restrict)
    phi, rows = loo_impacts(Cl, St, common)
    imp = {r['symbol']: r['impact'] for r in rows}
    helpers = [r['symbol'] for r in rows if r['impact'] > 0.01]
    print(f"[{label:34s}] n={len(common):2d} padded={C.padded_phi(Cl, St):.4f} "
          f"helpers={helpers}  XMR={imp.get('XMR', float('nan')):+.4f} "
          f"CRV={imp.get('CRV', float('nan')):+.4f} SOL={imp.get('SOL', float('nan')):+.4f} "
          f"YFI={imp.get('YFI', float('nan')):+.4f}")
    return {'label': label, 'n': len(common), 'padded_phi': float(C.padded_phi(Cl, St)),
            'helpers': helpers, 'impacts': imp}


def main():
    z, ssym, _ = C.build_stats_matrix('base')
    fac, fsym = C.load_factors()
    cl_clean, sym_clean = C.load_claims(clean=True)
    cl_dirty, sym_dirty = C.load_claims(clean=False)
    common_dirty = sorted(set(sym_dirty) & set(ssym) & set(fsym))
    common_clean = sorted(set(sym_clean) & set(ssym) & set(fsym))
    L = sorted(set(common_dirty) & set(common_clean))

    res = {}
    res['A_full37'] = run(cl_dirty, sym_dirty, z, ssym, fac, fsym, common_dirty,
                          'contaminated n=37 (paper left panel)')
    for drop, lab in [(['YFI'], 'contaminated minus YFI'),
                      (['AXS'], 'contaminated minus AXS'),
                      (['SUSHI'], 'contaminated minus SUSHI'),
                      (['AXS', 'SUSHI'], 'contaminated minus AXS,SUSHI'),
                      (['YFI', 'AXS', 'SUSHI'], 'contaminated minus all 3 stubs (=L)')]:
        key = 'drop_' + '_'.join(drop)
        res[key] = run(cl_dirty, sym_dirty, z, ssym, fac, fsym,
                       [s for s in common_dirty if s not in drop], lab)

    # documents-only swap: same 34 assets, re-collected documents
    res['C_clean_on_L'] = run(cl_clean, sym_clean, z, ssym, fac, fsym, L,
                              'clean documents on the same 34')
    res['D_clean43'] = run(cl_clean, sym_clean, z, ssym, fac, fsym, common_clean,
                           'clean n=43 (paper right panel)')

    # which of the 34 shared documents actually changed
    di = {s: cl_dirty[sym_dirty.index(s)] for s in L}
    ci = {s: cl_clean[sym_clean.index(s)] for s in L}
    ch = {s: float(np.abs(di[s] - ci[s]).max()) for s in L}
    res['documents_replaced'] = sorted([s for s, v in ch.items() if v > 1e-4])
    res['documents_unchanged'] = sorted([s for s, v in ch.items() if v <= 1e-4])
    print(f"\ndocuments actually replaced on L ({len(res['documents_replaced'])}): "
          f"{res['documents_replaced']}")
    print(f"documents byte-identical on L ({len(res['documents_unchanged'])})")

    json.dump(res, open(OUTF, 'w'), indent=2)
    print(f"wrote {OUTF}")


if __name__ == '__main__':
    main()
