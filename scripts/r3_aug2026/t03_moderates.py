#!/usr/bin/env python3
"""TASK 3 -- the two moderate findings.

(a) MODERATE M7.  "distance correlation (0.48) and the mean canonical correlation
    (0.45) are moderate, not small -- they simply fail significance at n = 43
    (p = 0.93 and 0.17)"  (main-blinded.tex:1184).
    A dCor of 0.477 with permutation p = 0.928 sits BELOW its own null.  This
    script characterises the null for all four alternative metrics and computes
    the bias-corrected (U-centred) distance correlation of Szekely & Rizzo
    (2013, Ann. Statist. 41:2382), which is an unbiased estimator of dCov^2 and
    may be negative.

(b) MODERATE M9.  The market-capitalisation control
    (src/alignment/reviewer3_additions.py:305,324) sets
        market_cap_proxy = stats_aligned[:, 4]   # avg_volume
    and then residualises `stats_aligned` on it -- i.e. it residualises S on one
    of S's own seven columns, which annihilates that column exactly.  This
    script quantifies the resulting rank loss and separates the mechanical
    column-deletion component from any genuine partialling effect.

Nothing under code/src or code/outputs is written to.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).parent))
import common as C                                            # noqa: E402
from alignment.alternative_metrics import (                    # noqa: E402
    rv_coefficient, distance_correlation, cca_correlation, pls_score)

OUTF = Path(__file__).parent / "results_t03_moderates.json"
BASIS = 'usd_close'


# --------------------------------------------------------------------------- #
# (a) bias-corrected distance correlation (U-centring)
# --------------------------------------------------------------------------- #
def u_centre(D):
    n = D.shape[0]
    r = D.sum(axis=1, keepdims=True)
    c = D.sum(axis=0, keepdims=True)
    g = D.sum()
    A = D - r / (n - 2) - c / (n - 2) + g / ((n - 1) * (n - 2))
    np.fill_diagonal(A, 0.0)
    return A


def dcor_unbiased(X, Y):
    """Szekely & Rizzo (2013) bias-corrected dCor*; may be negative."""
    n = X.shape[0]
    A = u_centre(squareform(pdist(X, 'euclidean')))
    B = u_centre(squareform(pdist(Y, 'euclidean')))
    denom = n * (n - 3)
    dcov = (A * B).sum() / denom
    vx = (A * A).sum() / denom
    vy = (B * B).sum() / denom
    if vx <= 0 or vy <= 0:
        return {'dcov_star': float(dcov), 'dcor_star': np.nan,
                'dvar_x_star': float(vx), 'dvar_y_star': float(vy),
                't_stat': np.nan, 'p_one_sided': np.nan, 'df': np.nan}
    R = dcov / np.sqrt(vx * vy)
    v = n * (n - 3) / 2.0
    # Szekely-Rizzo t-test for high-dimensional independence
    if abs(R) < 1:
        T = np.sqrt(v - 1) * R / np.sqrt(1 - R ** 2)
    else:
        T = np.inf * np.sign(R)
    p = float(sps.t.sf(T, df=v - 1))
    return {'dcov_star': float(dcov), 'dcor_star': float(R),
            'dvar_x_star': float(vx), 'dvar_y_star': float(vy),
            't_stat': float(T), 'p_one_sided': p, 'df': float(v - 1)}


def null_profile(X, Y, fn, obs, n_perm=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    vals = []
    for _ in range(n_perm):
        v = fn(X, Y[rng.permutation(n)])
        if isinstance(v, dict):
            v = v.get('mean_correlation', v.get('dcor_star', 0.0))
        vals.append(v)
    vals = np.asarray(vals, float)
    return {
        'n_perm': n_perm,
        'null_mean': float(vals.mean()), 'null_sd': float(vals.std(ddof=1)),
        'null_q01': float(np.percentile(vals, 1)),
        'null_q05': float(np.percentile(vals, 5)),
        'null_q50': float(np.percentile(vals, 50)),
        'null_q95': float(np.percentile(vals, 95)),
        'null_q99': float(np.percentile(vals, 99)),
        'null_min': float(vals.min()), 'null_max': float(vals.max()),
        'observed': float(obs),
        'percentile_of_observed': float((vals < obs).mean() * 100),
        'p_upper_tail': float((vals >= obs).mean()),
        'p_lower_tail': float((vals <= obs).mean()),
        'p_as_published_two_sided_abs': float((np.abs(vals) >= abs(obs)).mean()),
        'excess_over_null_mean': float(obs - vals.mean()),
        'z_vs_null': float((obs - vals.mean()) / vals.std(ddof=1)),
    }


# --------------------------------------------------------------------------- #
# (b) helpers
# --------------------------------------------------------------------------- #
def residualize(X, control):
    Xr = np.zeros_like(X)
    for j in range(X.shape[1]):
        s, i, *_ = sps.linregress(control, X[:, j])
        Xr[:, j] = X[:, j] - (s * control + i)
    return Xr


def eff_rank(M):
    """Participation-ratio and entropy effective rank of a centred matrix."""
    Mc = M - M.mean(axis=0)
    s = np.linalg.svd(Mc, compute_uv=False)
    lam = s ** 2
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    p = lam / lam.sum()
    p = p[p > 0]
    ent = float(np.exp(-(p * np.log(p)).sum()))
    return {'participation_ratio': pr, 'entropy_effective_rank': ent,
            'numerical_rank_1e-10': int((s > s.max() * 1e-10).sum()),
            'singular_values': [float(v) for v in s]}


def r3_tucker_phi(A, B, return_cols=False):
    """Verbatim reviewer3_additions.py::tucker_phi (pads, SKIPS zero columns)."""
    from scipy.linalg import orthogonal_procrustes
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    A, B = C.pad_to(A, B)
    R, _ = orthogonal_procrustes(A, B)
    Ar = A @ R
    phis, used = [], []
    for j in range(Ar.shape[1]):
        num = float(Ar[:, j] @ B[:, j])
        den = float(np.sqrt((Ar[:, j] @ Ar[:, j]) * (B[:, j] @ B[:, j])))
        if den > 0:
            phis.append(num / den)
            used.append(j)
    v = float(np.mean(phis)) if phis else 0.0
    return (v, used) if return_cols else v


def matched_phi_cols(X, Y, target):
    """C.matched_phi, additionally reporting how many columns entered the mean."""
    from alignment.matched_dimension_analysis import reduce_dimensions
    from scipy.linalg import orthogonal_procrustes
    A = reduce_dimensions(X, target)
    A = A - A.mean(axis=0)
    B = Y - Y.mean(axis=0)
    R, _ = orthogonal_procrustes(A, B)
    Ar = A @ R
    phis = []
    for j in range(Ar.shape[1]):
        num = float(Ar[:, j] @ B[:, j])
        den = float(np.sqrt((Ar[:, j] @ Ar[:, j]) * (B[:, j] @ B[:, j])))
        if den > 0:
            phis.append(num / den)
    return (float(np.mean(phis)) if phis else 0.0), len(phis)


# --------------------------------------------------------------------------- #
def main():
    out = {}
    claims, csym = C.load_claims(clean=True)
    fac, fsym = C.load_factors()
    z, ssym, raw = C.build_stats_matrix(BASIS)
    Cl, St, Fa, common = C.align_three(claims, csym, z, ssym, fac, fsym)
    n = len(common)
    out['n'] = n
    out['volume_basis'] = BASIS
    print(f"n = {n}, claims {Cl.shape}, stats {St.shape}, basis = {BASIS}")

    # ================================================================= (a) ==
    print("\n" + "=" * 84)
    print("(a) MODERATE M7 -- dCor 0.477 with p = 0.928 lies BELOW its own null")
    print("=" * 84)
    metrics = {
        'rv': (rv_coefficient, rv_coefficient(Cl, St)),
        'dcor': (distance_correlation, distance_correlation(Cl, St)),
        'cca': (lambda a, b: cca_correlation(a, b)['mean_correlation'],
                cca_correlation(Cl, St)['mean_correlation']),
        'pls': (lambda a, b: pls_score(a, b)['mean_correlation'],
                pls_score(Cl, St)['mean_correlation']),
    }
    A = {}
    print(f"\n{'metric':6s} {'observed':>9s} {'null mean':>10s} {'null sd':>8s} "
          f"{'null q05':>9s} {'null q95':>9s} {'pctile':>7s} {'excess':>8s} "
          f"{'z':>6s} {'p(pub)':>7s} {'p(lower)':>9s}")
    for name, (fn, obs) in metrics.items():
        prof = null_profile(Cl, St, fn, obs, n_perm=5000)
        A[name] = prof
        print(f"{name:6s} {obs:9.4f} {prof['null_mean']:10.4f} {prof['null_sd']:8.4f} "
              f"{prof['null_q05']:9.4f} {prof['null_q95']:9.4f} "
              f"{prof['percentile_of_observed']:6.1f}% {prof['excess_over_null_mean']:+8.4f} "
              f"{prof['z_vs_null']:+6.2f} {prof['p_as_published_two_sided_abs']:7.3f} "
              f"{prof['p_lower_tail']:9.3f}")
    out['alt_metric_nulls'] = A

    # the same three legs, so the reader can see which nulls are informative
    print("\n  the two other legs, for context:")
    legs = {'claims_vs_stats': (Cl, St), 'claims_vs_factors': (Cl, Fa),
            'stats_vs_factors': (St, Fa)}
    L = {}
    for lname, (X, Y) in legs.items():
        d = distance_correlation(X, Y)
        prof = null_profile(X, Y, distance_correlation, d, n_perm=2000)
        ub = dcor_unbiased(X, Y)
        L[lname] = {'dcor': float(d), 'null': prof, 'unbiased': ub}
        print(f"    {lname:20s} dCor={d:.4f} null mean={prof['null_mean']:.4f} "
              f"pctile={prof['percentile_of_observed']:5.1f}%  "
              f"dCor*={ub['dcor_star']:+.4f} (SR t-test p={ub['p_one_sided']:.4f})")
    out['dcor_by_leg'] = L

    # bias-corrected dCor, claims vs stats, with its own permutation reference
    ub = dcor_unbiased(Cl, St)
    ubprof = null_profile(Cl, St, lambda a, b: dcor_unbiased(a, b)['dcor_star'],
                          ub['dcor_star'], n_perm=5000)
    out['dcor_unbiased_claims_stats'] = {'estimate': ub, 'null': ubprof}
    print(f"\n  bias-corrected (U-centred) dCor*, claims vs stats:")
    print(f"    dCov* = {ub['dcov_star']:+.6f}   dCor* = {ub['dcor_star']:+.4f}")
    print(f"    Szekely-Rizzo t = {ub['t_stat']:+.3f} on {ub['df']:.0f} df, "
          f"one-sided p = {ub['p_one_sided']:.4f}")
    print(f"    permutation null for dCor*: mean {ubprof['null_mean']:+.4f} "
          f"sd {ubprof['null_sd']:.4f}, observed at the "
          f"{ubprof['percentile_of_observed']:.1f}th percentile")

    # how much of the raw 0.477 is finite-sample floor?
    print(f"\n  decomposition of the raw statistic:")
    print(f"    raw dCor                       = {metrics['dcor'][1]:.4f}")
    print(f"    permutation-null mean (floor)  = {A['dcor']['null_mean']:.4f}")
    print(f"    excess over the floor          = {A['dcor']['excess_over_null_mean']:+.4f}")
    print(f"    raw CCA mean canonical corr    = {metrics['cca'][1]:.4f}")
    print(f"    its null mean (floor)          = {A['cca']['null_mean']:.4f}")
    print(f"    excess over the floor          = {A['cca']['excess_over_null_mean']:+.4f}")
    print(f"    raw RV                         = {metrics['rv'][1]:.4f}  "
          f"null {A['rv']['null_mean']:.4f}  excess {A['rv']['excess_over_null_mean']:+.4f}")
    print(f"    raw PLS                        = {metrics['pls'][1]:.4f}  "
          f"null {A['pls']['null_mean']:.4f}  excess {A['pls']['excess_over_null_mean']:+.4f}")

    # ================================================================= (b) ==
    print("\n" + "=" * 84)
    print("(b) MODERATE M9 -- the market-cap control residualises S on S[:, 4]")
    print("=" * 84)
    proxy = St[:, 4].copy()
    proxy = (proxy - proxy.mean()) / (proxy.std() + 1e-10)
    Clr = residualize(Cl, proxy)
    Str = residualize(St, proxy)
    Far = residualize(Fa, proxy)

    print(f"\n  after residualising S on its own column 4 (avg_volume):")
    colnorm = np.abs(Str).max(axis=0)
    for j, nm in enumerate(C.STATISTICS):
        print(f"    col {j} {nm:16s} max|value| = {colnorm[j]:.3e}"
              + ("   <-- ANNIHILATED" if colnorm[j] < 1e-10 else ""))
    out['residualised_column_maxabs'] = {nm: float(colnorm[j])
                                         for j, nm in enumerate(C.STATISTICS)}

    er_raw, er_res = eff_rank(St), eff_rank(Str)
    print(f"\n  effective rank of S    : participation ratio {er_raw['participation_ratio']:.3f}, "
          f"entropy rank {er_raw['entropy_effective_rank']:.3f}, "
          f"numerical rank {er_raw['numerical_rank_1e-10']}")
    print(f"  effective rank of S~   : participation ratio {er_res['participation_ratio']:.3f}, "
          f"entropy rank {er_res['entropy_effective_rank']:.3f}, "
          f"numerical rank {er_res['numerical_rank_1e-10']}")
    out['effective_rank'] = {'S': er_raw, 'S_residualised': er_res}

    # how much of each matrix does the proxy explain?
    r2_s = [float(sps.pearsonr(proxy, St[:, j])[0] ** 2) for j in range(St.shape[1])]
    r2_c = [float(sps.pearsonr(proxy, Cl[:, j])[0] ** 2) for j in range(Cl.shape[1])]
    cats = json.load(open(C.OUT / 'nlp' / 'claims_matrix_meta.json'))['categories']
    print(f"\n  R^2 of the proxy on each column of S: "
          + ", ".join(f"{nm}={v:.3f}" for nm, v in zip(C.STATISTICS, r2_s)))
    print(f"  R^2 of the proxy on each column of C: "
          + ", ".join(f"{nm}={v:.3f}" for nm, v in zip(cats, r2_c)))
    print(f"  mean R^2 on C = {np.mean(r2_c):.4f}  (max {max(r2_c):.4f} on "
          f"{cats[int(np.argmax(r2_c))]})")
    out['proxy_r2'] = {'stats': dict(zip(C.STATISTICS, r2_s)),
                       'claims': dict(zip(cats, r2_c)),
                       'mean_r2_claims': float(np.mean(r2_c))}

    # --- the estimator's column bookkeeping -------------------------------- #
    v_raw, k_raw = matched_phi_cols(Cl, St, 7)
    v_res, k_res = matched_phi_cols(Clr, Str, 7)
    print(f"\n  matched-dimension estimator, columns entering the mean:")
    print(f"    raw          phi = {v_raw:.4f} over {k_raw} columns")
    print(f"    residualised phi = {v_res:.4f} over {k_res} columns  "
          f"<-- {'NOT like-for-like' if k_res != k_raw else 'like-for-like'}")
    pr_raw, u_raw = r3_tucker_phi(Cl, St, return_cols=True)
    pr_res, u_res = r3_tucker_phi(Clr, Str, return_cols=True)
    print(f"  un-reduced (reviewer3_additions) estimator:")
    print(f"    raw          phi = {pr_raw:.4f} over {len(u_raw)} columns")
    print(f"    residualised phi = {pr_res:.4f} over {len(u_res)} columns")
    out['column_bookkeeping'] = {
        'matched_raw': {'phi': v_raw, 'n_cols': k_raw},
        'matched_residualised': {'phi': v_res, 'n_cols': k_res},
        'r3_raw': {'phi': pr_raw, 'n_cols': len(u_raw)},
        'r3_residualised': {'phi': pr_res, 'n_cols': len(u_res)},
    }

    # --- decomposition ------------------------------------------------------ #
    keep6 = [j for j in range(7) if j != 4]
    S6 = St[:, keep6]
    S6r = Str[:, keep6]
    variants = {
        'A_baseline_C_vs_S7':                (Cl,  St,  7),
        'B_published_control_Cres_vs_Sres7': (Clr, Str, 7),
        'C_drop_col4_only_C_vs_S6':          (Cl,  S6,  6),
        'D_drop_and_partial_Cres_vs_S6res':  (Clr, S6r, 6),
        'E_residualise_S_only_C_vs_Sres7':   (Cl,  Str, 7),
        'F_residualise_C_only_Cres_vs_S7':   (Clr, St,  7),
        'G_residualise_S_only_C_vs_S6res':   (Cl,  S6r, 6),
    }
    print(f"\n  {'variant':36s} {'matched phi':>12s} {'p':>7s} {'#cols':>6s} "
          f"{'r3 phi':>8s} {'padded phi':>11s}")
    D = {}
    for k, (X, Y, tgt) in variants.items():
        mv, mc = matched_phi_cols(X, Y, tgt)
        p = C.matched_perm_p(X, Y, tgt, n_perm=1000)
        D[k] = {'matched_phi': float(mv), 'matched_p': float(p), 'n_cols': mc,
                'r3_phi': float(r3_tucker_phi(X, Y)),
                'padded_phi': float(C.padded_phi(X, Y))}
        print(f"  {k:36s} {mv:12.4f} {p:7.3f} {mc:6d} {D[k]['r3_phi']:8.4f} "
              f"{D[k]['padded_phi']:11.4f}")
    out['marketcap_decomposition'] = D

    a, b = D['A_baseline_C_vs_S7'], D['B_published_control_Cres_vs_Sres7']
    c, d = D['C_drop_col4_only_C_vs_S6'], D['D_drop_and_partial_Cres_vs_S6res']
    print(f"\n  published move (A -> B), matched : {a['matched_phi']:.4f} -> "
          f"{b['matched_phi']:.4f}  ({b['matched_phi']-a['matched_phi']:+.4f})")
    print(f"  mechanical column deletion (A -> C): {a['matched_phi']:.4f} -> "
          f"{c['matched_phi']:.4f}  ({c['matched_phi']-a['matched_phi']:+.4f})")
    print(f"  genuine partialling  (C -> D)      : {c['matched_phi']:.4f} -> "
          f"{d['matched_phi']:.4f}  ({d['matched_phi']-c['matched_phi']:+.4f})")
    out['marketcap_attribution'] = {
        'published_move': b['matched_phi'] - a['matched_phi'],
        'column_deletion_component': c['matched_phi'] - a['matched_phi'],
        'genuine_partial_component': d['matched_phi'] - c['matched_phi'],
    }

    # does residualising S on col4 differ from dropping col4?  (S~ restricted to
    # the 6 live columns vs the raw 6 columns)
    diffs = {C.STATISTICS[j]: float(np.abs(S6[:, i] - S6r[:, i]).max() /
                                    (np.abs(S6[:, i]).max() + 1e-12))
             for i, j in enumerate(keep6)}
    print(f"\n  relative change in the six surviving columns of S after partialling: "
          + ", ".join(f"{k}={v:.3f}" for k, v in diffs.items()))
    out['surviving_column_relative_change'] = diffs
    out['external_marketcap_available'] = False

    json.dump(out, open(OUTF, 'w'), indent=2)
    print(f"\nwrote {OUTF}")


if __name__ == '__main__':
    main()
