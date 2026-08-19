#!/usr/bin/env python3
"""TASK 2 (review finding M13) -- window-length heterogeneity.

Seven of 49 assets have materially shorter estimation windows (POL 2,630 hourly
rows vs a full panel of 17,543 -- a 6.7x spread; main-blinded.tex:184).  Four of
the seven market statistics are window-length dependent.  They are then z-scored
cross-sectionally (common.py:112 / src/market/summary_statistics.py:112) and
treated as commensurable coordinates of S.

This script:
  A. quantifies the window-length dependence of each statistic directly, by
     recomputing the FULL-window assets' statistics on each truncated window and
     measuring how far each statistic moves for an unchanged asset;
  B. locates the seven short-window assets in the z-scored cross-section, per
     statistic (mean z, Welch t, Mann-Whitney, Cliff's delta), and correlates
     each z-scored statistic with log window length;
  C. re-runs the primary matched-dimension alignment
       (a) excluding the seven short-window assets,
       (b) restricting all assets to a common overlapping window
           -- note the STRICT common window over all 49 is EMPTY, so this is a
           ladder of nested windows, each requiring further exclusions;
       (c) under a minimum-coverage screen;
     against the current primary (matched phi = 0.280, p = 0.507, usd_close).

Nothing under code/src or code/outputs is written to.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).parent))
import common as C                                            # noqa: E402

OUTF = Path(__file__).parent / "results_t02_window_length.json"
BASIS = 'usd_close'                # the manuscript's primary volume specification
FULL_ROWS = 17543


# --------------------------------------------------------------------------- #
def load_panel():
    panel = {}
    for f in sorted(C.MARKET_DIR.glob("*.parquet")):
        sym = f.stem.replace('_ohlcv', '').upper()
        df = pd.read_parquet(f)
        panel[sym] = df
    return panel


def stats_matrix_from(panel, symbols, start=None, end=None, basis=BASIS,
                      min_days=5):
    """Build the z-scored M x 7 matrix over `symbols`, optionally windowed."""
    rows, keep = [], []
    for s in symbols:
        df = panel[s]
        if start is not None:
            df = df[df['timestamp'] >= start]
        if end is not None:
            df = df[df['timestamp'] <= end]
        if df.empty or df['timestamp'].dt.date.nunique() < min_days:
            continue
        rows.append([C.compute_stats_for_df(df, basis)[k] for k in C.STATISTICS])
        keep.append(s)
    raw = np.asarray(rows, float)
    z = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-8)
    return z, keep, raw


def align2(claims, csym, z, ssym, restrict=None):
    common = sorted(set(csym) & set(ssym))
    if restrict is not None:
        common = sorted(set(common) & set(restrict))
    return (claims[[csym.index(s) for s in common]],
            z[[ssym.index(s) for s in common]], common)


def report(Cl, St, label, n_perm=1000):
    out = {
        'label': label, 'n': int(Cl.shape[0]),
        'matched_phi': float(C.matched_phi(Cl, St, 7)),
        'matched_p': float(C.matched_perm_p(Cl, St, 7, n_perm=n_perm)),
        'padded_phi': float(C.padded_phi(Cl, St)),
        'padded_p': float(C.padded_perm_p(Cl, St, n_perm=n_perm)),
    }
    print(f"[{label:52s}] n={out['n']:2d}  matched phi={out['matched_phi']:.4f} "
          f"(p={out['matched_p']:.3f})   padded phi={out['padded_phi']:.4f} "
          f"(p={out['padded_p']:.3f})")
    return out


def main():
    out = {}
    panel = load_panel()
    claims, csym = C.load_claims(clean=True)
    fac, fsym = C.load_factors()

    coverage = {s: {'n_rows': int(len(df)),
                    'start': str(df['timestamp'].min()),
                    'end': str(df['timestamp'].max())} for s, df in panel.items()}
    out['coverage'] = coverage
    short = sorted([s for s, v in coverage.items() if v['n_rows'] < FULL_ROWS],
                   key=lambda s: coverage[s]['n_rows'])
    out['short_window_assets'] = short
    print("short-window assets (n_rows < 17,543):")
    for s in short:
        print(f"   {s:7s} {coverage[s]['n_rows']:6d} rows  "
              f"{coverage[s]['n_rows']/FULL_ROWS*100:5.1f}%  "
              f"{coverage[s]['start'][:10]} -> {coverage[s]['end'][:10]}")

    # the strict common window over all 49
    latest_start = max(pd.Timestamp(v['start']) for v in coverage.values())
    earliest_end = min(pd.Timestamp(v['end']) for v in coverage.values())
    out['strict_common_window'] = {
        'latest_start': str(latest_start), 'earliest_end': str(earliest_end),
        'is_empty': bool(latest_start > earliest_end)}
    print(f"\nSTRICT common window over all 49: [{latest_start}, {earliest_end}] "
          f"-> {'EMPTY' if latest_start > earliest_end else 'non-empty'}")

    # ------------------------------------------------------------------- A --
    print("\n" + "=" * 82)
    print("A. HOW WINDOW-DEPENDENT IS EACH STATISTIC?  (full-window assets only,")
    print("   recomputed on each truncated window; the asset is unchanged, only the window)")
    print("=" * 82)
    fullsyms = sorted([s for s in panel if coverage[s]['n_rows'] == FULL_ROWS])
    _, _, raw_full = stats_matrix_from(panel, fullsyms)
    A = {}
    for s in short:
        st, en = pd.Timestamp(coverage[s]['start']), pd.Timestamp(coverage[s]['end'])
        _, keep, raw_w = stats_matrix_from(panel, fullsyms, start=st, end=en)
        assert keep == fullsyms
        blk = {}
        for j, name in enumerate(C.STATISTICS):
            a, b = raw_full[:, j], raw_w[:, j]
            # z-score both within their own cross-section, then compare
            za = (a - a.mean()) / (a.std() + 1e-12)
            zb = (b - b.mean()) / (b.std() + 1e-12)
            blk[name] = {
                'pearson_raw': float(sps.pearsonr(a, b)[0]),
                'spearman_raw': float(sps.spearmanr(a, b).statistic),
                'spearman_z': float(sps.spearmanr(za, zb).statistic),
                'mean_abs_z_shift': float(np.abs(za - zb).mean()),
                'median_ratio_raw': float(np.median(b / np.where(np.abs(a) < 1e-12, np.nan, a))),
            }
        A[s] = {'window_rows': coverage[s]['n_rows'], 'per_statistic': blk}
        print(f"\n  window of {s} ({coverage[s]['n_rows']} rows, "
              f"{coverage[s]['start'][:10]}..{coverage[s]['end'][:10]}), "
              f"applied to the {len(fullsyms)} full-window assets:")
        for name in C.STATISTICS:
            v = blk[name]
            print(f"    {name:15s} rank-corr(full, windowed) = {v['spearman_raw']:+.3f}   "
                  f"mean |dz| = {v['mean_abs_z_shift']:.3f}")
    out['window_dependence'] = A

    # ------------------------------------------------------------------- B --
    print("\n" + "=" * 82)
    print("B. WHERE DO THE SEVEN SIT IN THE Z-SCORED CROSS-SECTION?")
    print("=" * 82)
    z49, sym49, raw49 = stats_matrix_from(panel, sorted(panel.keys()))
    isshort = np.array([s in short for s in sym49])
    nrows = np.array([coverage[s]['n_rows'] for s in sym49], float)

    # restrict to the 43-asset alignment sample too
    align_syms = sorted(set(csym) & set(sym49) & set(fsym))
    out['alignment_sample_n'] = len(align_syms)
    out['short_in_alignment_sample'] = sorted(set(short) & set(align_syms))
    idx43 = [sym49.index(s) for s in align_syms]
    z43 = z49[idx43]
    short43 = np.array([s in short for s in align_syms])
    nrows43 = np.array([coverage[s]['n_rows'] for s in align_syms], float)

    B = {}
    print(f"\n{'statistic':16s} {'meanZ short':>11s} {'meanZ full':>10s} {'diff':>7s} "
          f"{'Welch t':>8s} {'p':>7s} {'MWU p':>7s} {'Cliff d':>8s} {'rho(z,log n)':>12s} {'p':>7s}")
    for j, name in enumerate(C.STATISTICS):
        a, b = z43[short43, j], z43[~short43, j]
        t, pt = sps.ttest_ind(a, b, equal_var=False)
        try:
            u, pu = sps.mannwhitneyu(a, b, alternative='two-sided')
        except ValueError:
            u, pu = np.nan, np.nan
        # Cliff's delta
        gt = sum((x > y) for x in a for y in b)
        lt = sum((x < y) for x in a for y in b)
        cliff = (gt - lt) / (len(a) * len(b))
        rho = sps.spearmanr(np.log(nrows43), z43[:, j])
        B[name] = {
            'mean_z_short': float(a.mean()), 'mean_z_full': float(b.mean()),
            'diff': float(a.mean() - b.mean()),
            'welch_t': float(t), 'welch_p': float(pt),
            'mwu_p': float(pu), 'cliffs_delta': float(cliff),
            'spearman_z_vs_log_rows': float(rho.statistic),
            'spearman_p': float(rho.pvalue),
            'short_z_values': {s: float(z43[i, j]) for i, s in enumerate(align_syms)
                               if short43[i]},
        }
        print(f"{name:16s} {a.mean():11.3f} {b.mean():10.3f} {a.mean()-b.mean():7.3f} "
              f"{t:8.2f} {pt:7.4f} {pu:7.4f} {cliff:8.3f} {rho.statistic:12.3f} "
              f"{rho.pvalue:7.4f}")
    out['zscore_position'] = B

    # joint test: is the 7-vector of short assets displaced in R^7?
    mu_s = z43[short43].mean(axis=0)
    print(f"\n  joint mean-z vector of the seven: "
          + ", ".join(f"{n}={v:+.2f}" for n, v in zip(C.STATISTICS, mu_s)))
    print(f"  norm of that displacement = {np.linalg.norm(mu_s):.3f}  "
          f"(a permutation reference follows)")
    rng = np.random.default_rng(20260808)
    nullnorm = []
    for _ in range(20000):
        pick = rng.choice(len(align_syms), size=int(short43.sum()), replace=False)
        nullnorm.append(np.linalg.norm(z43[pick].mean(axis=0)))
    nullnorm = np.array(nullnorm)
    p_joint = float((nullnorm >= np.linalg.norm(mu_s)).mean())
    out['joint_displacement'] = {
        'mean_z_vector': [float(v) for v in mu_s],
        'norm': float(np.linalg.norm(mu_s)),
        'perm_null_mean': float(nullnorm.mean()),
        'perm_null_q95': float(np.percentile(nullnorm, 95)),
        'perm_p': p_joint}
    print(f"  permutation null mean={nullnorm.mean():.3f} q95={np.percentile(nullnorm,95):.3f} "
          f"-> p = {p_joint:.4f}")

    # ------------------------------------------------------------------- C --
    print("\n" + "=" * 82)
    print("C. RE-RUNNING THE PRIMARY ALIGNMENT")
    print("=" * 82)
    Cl0, St0, common0 = align2(claims, csym, z49, sym49, align_syms)
    runs = {'P0_published_primary': report(Cl0, St0, 'P0 published primary (n=43, available window)')}

    # (a) exclude the seven, z-scoring retained as published (over all 49)
    keep36 = [s for s in align_syms if s not in short]
    Cl, St, _ = align2(claims, csym, z49, sym49, keep36)
    runs['V1a_exclude7_z_over_49'] = report(Cl, St, 'V1a exclude 7 short (z from the 49-asset build)')

    # (a') exclude the seven and re-z-score on the 42 full-window market assets
    z42, sym42, _ = stats_matrix_from(panel, fullsyms)
    Cl, St, _ = align2(claims, csym, z42, sym42, keep36)
    runs['V1b_exclude7_rezscored'] = report(Cl, St, 'V1b exclude 7 short (re-z-scored on the 42)')

    # (b) common-window ladder
    ladder = [
        ('W1_drop_XMR_OCEAN', ['XMR', 'OCEAN']),
        ('W2_drop_XMR_OCEAN_POL', ['XMR', 'OCEAN', 'POL']),
        ('W3_drop_XMR_OCEAN_POL_RENDER', ['XMR', 'OCEAN', 'POL', 'RENDER']),
        ('W4_drop_XMR_OCEAN_POL_RENDER_SUI', ['XMR', 'OCEAN', 'POL', 'RENDER', 'SUI']),
        ('W5_drop_all_7_short', short),
    ]
    for key, drop in ladder:
        mkeep = [s for s in panel if s not in drop]
        st = max(pd.Timestamp(coverage[s]['start']) for s in mkeep)
        en = min(pd.Timestamp(coverage[s]['end']) for s in mkeep)
        if st > en:
            print(f"[{key}] window EMPTY -- skipped")
            continue
        zw, symw, _ = stats_matrix_from(panel, sorted(mkeep), start=st, end=en)
        akeep = [s for s in align_syms if s not in drop]
        Cl, St, com = align2(claims, csym, zw, symw, akeep)
        ndays = int(pd.Series(pd.date_range(st, en, freq='h')).dt.date.nunique())
        r = report(Cl, St, f'{key} common window {str(st)[:10]}..{str(en)[:10]} ({ndays}d)')
        r['window_start'] = str(st); r['window_end'] = str(en)
        r['window_days'] = ndays
        r['dropped'] = list(drop)
        runs[key] = r

    # (c) minimum-coverage screens
    for frac in [0.50, 0.80, 0.90]:
        thr = frac * FULL_ROWS
        keep = [s for s in align_syms if coverage[s]['n_rows'] >= thr]
        Cl, St, _ = align2(claims, csym, z49, sym49, keep)
        runs[f'S_mincov_{int(frac*100)}pct'] = report(
            Cl, St, f'S min-coverage >= {int(frac*100)}% of the panel')

    # (d) drop the two most truncated only (the paper's jackknife pair)
    for drop, key in [(['POL'], 'J_drop_POL'), (['POL', 'RENDER'], 'J_drop_POL_RENDER')]:
        keep = [s for s in align_syms if s not in drop]
        Cl, St, _ = align2(claims, csym, z49, sym49, keep)
        runs[key] = report(Cl, St, f'jackknife-style: drop {drop}')

    out['runs'] = runs

    # ------------------------------------------------------------------- D --
    # does the jackknife see it?  compare the 7 short assets' LOO impacts with the rest
    print("\n" + "=" * 82)
    print("D. CAN THE JACKKNIFE SEE IT?")
    print("=" * 82)
    from s06_contamination_likeforlike import loo_impacts
    _, rows = loo_impacts(Cl0, St0, common0)
    imp = {r['symbol']: r['impact'] for r in rows}
    a = np.array([imp[s] for s in align_syms if s in short])
    b = np.array([imp[s] for s in align_syms if s not in short])
    t, pt = sps.ttest_ind(a, b, equal_var=False)
    out['loo_short_vs_full'] = {
        'mean_impact_short': float(a.mean()), 'mean_impact_full': float(b.mean()),
        'welch_t': float(t), 'welch_p': float(pt),
        'short_impacts': {s: float(imp[s]) for s in align_syms if s in short},
        'spearman_impact_vs_log_rows': float(
            sps.spearmanr(np.log(nrows43), [imp[s] for s in align_syms]).statistic),
    }
    print(f"  LOO impact, seven short: mean {a.mean():+.4f}  "
          f"({', '.join(f'{s} {imp[s]:+.4f}' for s in align_syms if s in short)})")
    print(f"  LOO impact, 36 full    : mean {b.mean():+.4f}")
    print(f"  Welch t = {t:.2f}, p = {pt:.4f}; "
          f"rho(impact, log rows) = {out['loo_short_vs_full']['spearman_impact_vs_log_rows']:+.3f}")
    print(f"  For reference the published primary is phi = 0.2801 (p = 0.507); "
          f"the group deletion is the V1 rows above.")

    json.dump(out, open(OUTF, 'w'), indent=2)
    print(f"\nwrote {OUTF}")


if __name__ == '__main__':
    main()
