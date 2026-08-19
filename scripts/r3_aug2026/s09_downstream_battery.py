#!/usr/bin/env python3
"""STEP 9 -- every OTHER published number that touches the volume variable.

  * subsample stability          src/analysis/robustness.py::subsample_stability
  * feature importance           src/analysis/robustness.py::feature_importance
  * rolling-window temporal phi  src/analysis/temporal.py  (line 77 uses the raw
                                 base-asset `volume` for the window avg_volume)
  * Bitcoin-exclusion robustness
  * the volume ranking itself (is base-asset volume even a size proxy?)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from analysis.robustness import RobustnessChecker
from alignment.procrustes import ProcrustesAlignment
from alignment.congruence import CongruenceCoefficient

OUTF = Path(__file__).parent / "results_09_downstream.json"
_AL, _CC = ProcrustesAlignment(), CongruenceCoefficient()


def rolling_stats(volume_basis, window_months=6, step_months=3):
    """src/analysis/temporal.py::compute_rolling_stats with a volume basis."""
    all_data = {}
    for f in C.MARKET_DIR.glob("*.parquet"):
        sym = f.stem.replace("_ohlcv", "").upper()
        df = pd.read_parquet(f)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        if volume_basis == 'base':
            df['vol_use'] = df['volume']
        elif volume_basis == 'usd_close':
            df['vol_use'] = df['volume'] * df['close']
        all_data[sym] = df
    omin = min(d.index.min() for d in all_data.values())
    omax = max(d.index.max() for d in all_data.values())
    rng_days = (omax - omin).days
    all_data = {s: d for s, d in all_data.items()
                if (d.index.max() - d.index.min()).days / rng_days > 0.9}
    mn = max(d.index.min() for d in all_data.values())
    mx = min(d.index.max() for d in all_data.values())
    windows, cur = [], mn
    while cur + pd.DateOffset(months=window_months) <= mx:
        windows.append((cur, cur + pd.DateOffset(months=window_months)))
        cur += pd.DateOffset(months=step_months)
    out = []
    for i, (s, e) in enumerate(windows):
        ws = {}
        for sym, df in all_data.items():
            w = df[(df.index >= s) & (df.index < e)]
            if len(w) < 100:
                continue
            r = w['close'].pct_change().dropna()
            if len(r) == 0:
                continue
            cummax = w['close'].cummax()
            dd = ((w['close'] - cummax) / cummax).min()
            ws[sym] = {'mean_return': float(r.mean()), 'volatility': float(r.std()),
                       'sharpe': float(r.mean() / r.std()) if r.std() > 0 else 0,
                       'max_drawdown': float(dd),
                       'avg_volume': float(w['vol_use'].mean())}
        if ws:
            out.append({'window_id': i, 'start': s.isoformat(), 'end': e.isoformat(),
                        'stats': ws})
    return out


def temporal_phi(claims_full, sym_full, windows):
    res = []
    for w in windows:
        common = [s for s in w['stats'] if s in sym_full]
        if len(common) < 5:
            continue
        S = np.array([[w['stats'][s]['mean_return'], w['stats'][s]['volatility'],
                       w['stats'][s]['sharpe'], w['stats'][s]['max_drawdown'],
                       w['stats'][s]['avg_volume']] for s in common])
        S = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-8)
        Cm = claims_full[[sym_full.index(s) for s in common]]
        r = _AL.align_matrices(Cm, S)
        res.append(float(_CC.matrix_congruence(r['source_rotated'],
                                               r['target_centered'])['mean_phi']))
    return {'n_windows': len(res), 'mean_phi': float(np.mean(res)),
            'std_phi': float(np.std(res)), 'per_window': res}


def main():
    claims, csym = C.load_claims(True)
    fac, fsym = C.load_factors()
    cats = C.load_meta(C.OUT / 'nlp' / 'claims_matrix_meta.json')['categories']
    res = {}

    for vb in ['base', 'usd_close']:
        z, ssym, raw = C.build_stats_matrix(vb)
        Cl, St, Fa, common = C.align_three(claims, csym, z, ssym, fac, fsym)
        rc = RobustnessChecker(C.OUT / 'analysis')
        sub = rc.subsample_stability(Cl, St, common)
        fi = rc.feature_importance(Cl, St, cats, C.STATISTICS)
        # Bitcoin exclusion
        keep = [i for i, s in enumerate(common) if s != 'BTC']
        nb = {
            'n': len(keep),
            'padded_phi': float(C.padded_phi(Cl[keep], St[keep])),
            'padded_p': float(C.padded_perm_p(Cl[keep], St[keep])),
            'matched_phi': float(C.matched_phi(Cl[keep], St[keep], 7)),
            'matched_p': float(C.matched_perm_p(Cl[keep], St[keep], 7)),
        }
        tw = temporal_phi(claims, csym, rolling_stats(vb))
        res[vb] = {
            'subsample_stability': {k: sub[k] for k in
                                    ['mean_phi', 'std_phi', 'ci_lower', 'ci_upper']},
            'feature_importance_max_abs': float(max(
                abs(c['importance']) for c in fi['claims_importance'])),
            'feature_importance': fi['claims_importance'],
            'no_btc': nb,
            'temporal': {k: tw[k] for k in ['n_windows', 'mean_phi', 'std_phi']},
        }
        print(f"[{vb}] subsample phi={sub['mean_phi']:.3f}+-{sub['std_phi']:.3f}  "
              f"max|feature importance|={res[vb]['feature_importance_max_abs']:.4f}  "
              f"no-BTC padded={nb['padded_phi']:.4f}(p={nb['padded_p']:.3f}) "
              f"matched={nb['matched_phi']:.4f}(p={nb['matched_p']:.3f})  "
              f"temporal phi={tw['mean_phi']:.3f}+-{tw['std_phi']:.3f} "
              f"({tw['n_windows']} windows)")

    # is base-asset volume even a size proxy?
    zb, sb, rb = C.build_stats_matrix('base')
    zu, su, ru = C.build_stats_matrix('usd_close')
    iv = C.STATISTICS.index('avg_volume')
    ob = np.argsort(-rb[:, iv]); ou = np.argsort(-ru[:, iv])
    res['volume_ranking'] = {
        'top10_base_units': [sb[i] for i in ob[:10]],
        'bottom10_base_units': [sb[i] for i in ob[-10:]],
        'top10_usd_notional': [su[i] for i in ou[:10]],
        'bottom10_usd_notional': [su[i] for i in ou[-10:]],
        'btc_rank_base': int(np.where(ob == sb.index('BTC'))[0][0]) + 1,
        'btc_rank_usd': int(np.where(ou == su.index('BTC'))[0][0]) + 1,
        'eth_rank_base': int(np.where(ob == sb.index('ETH'))[0][0]) + 1,
        'eth_rank_usd': int(np.where(ou == su.index('ETH'))[0][0]) + 1,
    }
    v = res['volume_ranking']
    print(f"\nvolume ranking (n=49): BTC rank {v['btc_rank_base']} (base units) "
          f"-> {v['btc_rank_usd']} (USD);  ETH {v['eth_rank_base']} -> {v['eth_rank_usd']}")
    print(f"  top-10 by base-asset units : {v['top10_base_units']}")
    print(f"  top-10 by USD notional     : {v['top10_usd_notional']}")

    json.dump(res, open(OUTF, 'w'), indent=2)
    print(f"\nwrote {OUTF}")


if __name__ == '__main__':
    main()
