#!/usr/bin/env python3
"""STEP 2 -- re-run the headline alignment with USD-notional volume.

Variants
--------
base            : published pipeline (CCXT `volume`, BASE-ASSET units)
usd_close_avgonly: avg_volume from sum(volume*close); vol_volatility LEFT on the
                  base-asset series  -> isolates the avg_volume change exactly
usd_close       : both volume-derived stats from sum(volume*close)
usd_typical     : both volume-derived stats from sum(volume*(H+L+C)/3)
"""
import json
from pathlib import Path

import numpy as np
import common as C

OUTF = Path(__file__).parent / "results_02_usd_volume.json"

VARIANTS = [
    ('base',              dict(volume_basis='base',        cv_follows_basis=True)),
    ('usd_close_avgonly', dict(volume_basis='usd_close',   cv_follows_basis=False)),
    ('usd_close',         dict(volume_basis='usd_close',   cv_follows_basis=True)),
    ('usd_typical',       dict(volume_basis='usd_typical', cv_follows_basis=True)),
]

PAIRS = [('claims_vs_stats', 7), ('claims_vs_factors', 2), ('stats_vs_factors', 2)]


def main():
    claims, csym = C.load_claims(clean=True)
    fac, fsym = C.load_factors()
    res = {'variants': {}}

    for vname, kw in VARIANTS:
        z, syms, raw = C.build_stats_matrix(**kw)
        Cl, St, Fa, common = C.align_three(claims, csym, z, syms, fac, fsym)
        entry = {'n': len(common), 'kw': kw, 'alignment': {}, 'alt_metrics': {}}

        # descriptive: how much does the volume column actually move?
        iv = C.STATISTICS.index('avg_volume')
        entry['avg_volume_raw_range'] = [float(raw[:, iv].min()), float(raw[:, iv].max())]
        entry['avg_volume_raw_sd'] = float(raw[:, iv].std())

        mats = {'claims_vs_stats': (Cl, St), 'claims_vs_factors': (Cl, Fa),
                'stats_vs_factors': (St, Fa)}
        for name, tgt in PAIRS:
            X, Y = mats[name]
            pp = C.padded_phi(X, Y)
            mp = C.matched_phi(X, Y, tgt)
            entry['alignment'][name] = {
                'padded_phi': float(pp),
                'padded_p': float(C.padded_perm_p(X, Y)),
                'padded_ci': [float(v) for v in C.padded_boot(X, Y)],
                'padded_columns': [float(v) for v in C.padded_columns(X, Y)],
                'matched_phi': float(mp),
                'matched_p': float(C.matched_perm_p(X, Y, tgt)),
                'matched_ci': [float(v) for v in C.matched_boot(X, Y, tgt)],
            }
            entry['alt_metrics'][name] = C.alt_metrics(X, Y)

        res['variants'][vname] = entry
        a = entry['alignment']['claims_vs_stats']
        m = entry['alt_metrics']['claims_vs_stats']
        print(f"[{vname:18s}] n={len(common)} "
              f"matched={a['matched_phi']:.4f}(p={a['matched_p']:.3f}) "
              f"padded={a['padded_phi']:.4f}(p={a['padded_p']:.3f}) | "
              f"RV={m['rv']['value']:.3f}(p={m['rv']['p_value']:.3f}) "
              f"dCor={m['dcor']['value']:.3f}(p={m['dcor']['p_value']:.3f}) "
              f"CCA={m['cca']['value']:.3f}(p={m['cca']['p_value']:.3f}) "
              f"PLS={m['pls']['value']:.3f}(p={m['pls']['p_value']:.3f})")

    # correlation between the two volume measures across assets
    zb, sb, rb = C.build_stats_matrix('base')
    zu, su, ru = C.build_stats_matrix('usd_close')
    iv = C.STATISTICS.index('avg_volume')
    ic = C.STATISTICS.index('vol_volatility')
    res['volume_measure_diagnostics'] = {
        'pearson_r_log_base_vs_log_usd': float(np.corrcoef(rb[:, iv], ru[:, iv])[0, 1]),
        'spearman_rank_r': float(
            np.corrcoef(np.argsort(np.argsort(rb[:, iv])),
                        np.argsort(np.argsort(ru[:, iv])))[0, 1]),
        'cv_pearson_r_base_vs_usd': float(np.corrcoef(rb[:, ic], ru[:, ic])[0, 1]),
        'symbols': sb,
        'log1p_base': rb[:, iv].tolist(),
        'log1p_usd': ru[:, iv].tolist(),
    }
    print("\navg_volume: corr(log base, log USD) = "
          f"{res['volume_measure_diagnostics']['pearson_r_log_base_vs_log_usd']:.3f}; "
          f"rank corr = {res['volume_measure_diagnostics']['spearman_rank_r']:.3f}")
    print("vol_volatility: corr(base, USD) = "
          f"{res['volume_measure_diagnostics']['cv_pearson_r_base_vs_usd']:.3f}")

    json.dump(res, open(OUTF, 'w'), indent=2)
    print(f"\nwrote {OUTF}")


if __name__ == '__main__':
    main()
