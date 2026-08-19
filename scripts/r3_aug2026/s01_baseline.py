#!/usr/bin/env python3
"""STEP 1 -- reproduce the published headline numbers from the live market data.

Expected: matched-dimension claims-vs-stats phi = 0.303, zero-padded phi = 0.223.
"""
import json
import numpy as np
from pathlib import Path

import common as C

OUTF = Path(__file__).parent / "results_01_baseline.json"


def main():
    res = {}

    # --- (a) does the committed stats matrix reproduce from data/market? -----
    z, syms, raw = C.build_stats_matrix('base')
    pub = np.load(C.OUT / 'market' / 'stats_matrix.npy')
    pub_sym = C.load_meta(C.OUT / 'market' / 'stats_matrix_meta.json')['symbols']
    res['market_dir'] = str(C.MARKET_DIR)
    res['n_parquet'] = len(list(C.MARKET_DIR.glob('*.parquet')))
    res['symbols_match'] = (syms == pub_sym)
    res['stats_max_abs_diff'] = float(np.abs(z - pub).max())
    print(f"market dir      : {C.MARKET_DIR}")
    print(f"parquet files   : {res['n_parquet']}  symbols match published: {res['symbols_match']}")
    print(f"stats matrix max|diff| vs committed .npy: {res['stats_max_abs_diff']:.3e}")

    # --- (b) headline alignment ---------------------------------------------
    claims, csym = C.load_claims(clean=True)
    fac, fsym = C.load_factors()
    Cl, St, Fa, common = C.align_three(claims, csym, z, syms, fac, fsym)
    res['n_common'] = len(common)
    res['common'] = common
    print(f"common assets   : {len(common)}")

    pairs = {
        'claims_vs_stats':   (Cl, St, 7),
        'claims_vs_factors': (Cl, Fa, 2),
        'stats_vs_factors':  (St, Fa, 2),
    }
    res['alignment'] = {}
    for name, (X, Y, tgt) in pairs.items():
        pp = C.padded_phi(X, Y)
        pcols = C.padded_columns(X, Y)
        pp_p = C.padded_perm_p(X, Y)
        mp = C.matched_phi(X, Y, tgt)
        mp_p = C.matched_perm_p(X, Y, tgt)
        res['alignment'][name] = {
            'padded_phi': float(pp), 'padded_p': float(pp_p),
            'padded_columns': [float(v) for v in pcols],
            'matched_phi': float(mp), 'matched_p': float(mp_p),
            'matched_target_dim': tgt,
        }
        print(f"{name:20s} padded phi={pp:.4f} (p={pp_p:.3f})   "
              f"matched phi={mp:.4f} (p={mp_p:.3f})")

    # --- (c) alternative metrics --------------------------------------------
    res['alt_metrics'] = {}
    for name, (X, Y, _t) in pairs.items():
        res['alt_metrics'][name] = C.alt_metrics(X, Y)
        m = res['alt_metrics'][name]
        print(f"{name:20s} RV={m['rv']['value']:.3f}(p={m['rv']['p_value']:.3f}) "
              f"dCor={m['dcor']['value']:.3f}(p={m['dcor']['p_value']:.3f}) "
              f"CCA={m['cca']['value']:.3f}(p={m['cca']['p_value']:.3f}) "
              f"PLS={m['pls']['value']:.3f}(p={m['pls']['p_value']:.3f})")

    json.dump(res, open(OUTF, 'w'), indent=2)
    print(f"\nwrote {OUTF}")


if __name__ == '__main__':
    main()
