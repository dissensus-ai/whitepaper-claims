#!/usr/bin/env python3
"""TASK 1 (review finding M7) -- does the stub-row leverage mechanism exist?

The manuscript (main-blinded.tex:360) generalises the AXS/SUSHI/YFI episode into
a field-level rule:

    "at n ~ 40 the more dangerous failure is that a few say almost nothing,
     since near-empty rows carry no signal but do carry leverage."

The referee objection: Eq.(1) row-normalises every document profile
(src/nlp/zero_shot_classifier.py:76-77), so NO row of C is near-empty -- every
row sums to 1 regardless of source-document length.  The mechanism as stated is
therefore impossible in the matrix the estimator actually sees.

This script establishes, empirically:
  A. what the three stub rows look like in C after row-normalisation
     (row sums, entropy, centroid distance, hat-matrix leverage, per-column z)
  B. source-document word / chunk counts, and what survives normalisation
  C. whether "few chunks" predicts outlying composition (the candidate REAL
     mechanism: the profile is a MEAN over chunk-level score vectors, so its
     sampling variance scales ~1/n_chunks)
  D. controlled perturbation: inject 3 rows carrying the stubs' PROFILE into the
     clean corpus and see whether the helper ordering reappears; then inject
     3 rows with uniform / centroid / random / resampled-real profiles as
     controls, holding the market side identical in every arm.

Nothing under code/src or code/outputs is written to.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).parent))
import common as C                                            # noqa: E402
from s06_contamination_likeforlike import loo_impacts         # noqa: E402

OUTF = Path(__file__).parent / "results_t01_stub_leverage.json"
STUBS = ['AXS', 'SUSHI', 'YFI']
REPLACED = ['ADA', 'ALGO', 'ATOM', 'BAND', 'DOT', 'GRT', 'NEAR', 'RPL']
CATS = json.load(open(C.OUT / 'nlp' / 'claims_matrix_meta.json'))['categories']


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def entropy(p):
    p = np.clip(np.asarray(p, float), 1e-12, None)
    return float(-(p * np.log(p)).sum())


def hat_leverage(X):
    """h_ii of the centred configuration -- the standard leverage diagnostic."""
    Xc = X - X.mean(axis=0)
    H = Xc @ np.linalg.pinv(Xc.T @ Xc) @ Xc.T
    return np.diag(H).copy()


def row_diagnostics(M, syms, label):
    """Per-row diagnostics for a row-normalised claims matrix."""
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=1)
    lev = hat_leverage(M)
    # Mahalanobis on the centred matrix via pseudo-inverse (closure => singular)
    Xc = M - mu
    Sinv = np.linalg.pinv(np.cov(Xc, rowvar=False))
    maha = np.sqrt(np.maximum(0.0, np.einsum('ij,jk,ik->i', Xc, Sinv, Xc)))
    # distance from the MODAL profile: medoid = row minimising summed L1 distance
    D1 = np.abs(M[:, None, :] - M[None, :, :]).sum(-1)
    medoid = int(np.argmin(D1.sum(1)))
    out = {}
    for i, s in enumerate(syms):
        z = (M[i] - mu) / sd
        out[s] = {
            'row_sum': float(M[i].sum()),
            'entropy': entropy(M[i]),
            'euclid_from_centroid': float(np.linalg.norm(M[i] - mu)),
            'cosine_from_centroid': float(1 - (M[i] @ mu) /
                                          (np.linalg.norm(M[i]) * np.linalg.norm(mu))),
            'l1_from_medoid': float(np.abs(M[i] - M[medoid]).sum()),
            'mahalanobis': float(maha[i]),
            'hat_leverage': float(lev[i]),
            'max_abs_z': float(np.abs(z).max()),
            'argmax_abs_z_category': CATS[int(np.argmax(np.abs(z)))],
            'signed_z': [float(v) for v in z],
        }
    # ranks (1 = most extreme)
    for key in ['entropy', 'euclid_from_centroid', 'mahalanobis', 'hat_leverage',
                'max_abs_z', 'l1_from_medoid']:
        vals = np.array([out[s][key] for s in syms])
        order = (-vals).argsort()            # descending
        rk = np.empty(len(syms), int)
        rk[order] = np.arange(1, len(syms) + 1)
        for i, s in enumerate(syms):
            out[s][key + '_rank'] = int(rk[i])
    return {'label': label, 'n': len(syms), 'medoid': syms[medoid],
            'centroid': [float(v) for v in mu],
            'colsd': [float(v) for v in sd], 'rows': out}


# --------------------------------------------------------------------------- #
# injection machinery
# --------------------------------------------------------------------------- #
def make_injected(base_claims, base_syms, add_syms, add_profiles):
    """Return (claims, symbols) with `add_profiles` appended for `add_syms`."""
    Mc = np.vstack([base_claims, np.asarray(add_profiles, float)])
    return Mc, list(base_syms) + list(add_syms)


def arm(claims, csym, z, ssym, fac, fsym, restrict, label, verbose=True):
    Cl, St, Fa, common = C.align_three(claims, csym, z, ssym, fac, fsym, restrict)
    phi_full, rows = loo_impacts(Cl, St, common)
    imp = {r['symbol']: r['impact'] for r in rows}
    helpers = [r['symbol'] for r in rows if r['impact'] > 0.01]
    res = {
        'label': label, 'n': len(common),
        'padded_phi': float(C.padded_phi(Cl, St)),
        'matched_phi': float(C.matched_phi(Cl, St, 7)),
        'loo_phi_full': phi_full,
        'helpers': helpers, 'n_helpers': len(helpers),
        'max_impact': float(max(imp.values())),
        'min_impact': float(min(imp.values())),
        'impacts': {k: float(v) for k, v in imp.items()},
    }
    if verbose:
        print(f"[{label:38s}] n={res['n']:2d} padded={res['padded_phi']:.4f} "
              f"matched={res['matched_phi']:.4f} helpers({res['n_helpers']})={helpers} "
              f"max={res['max_impact']:+.4f}")
    return res


def main():
    rng = np.random.default_rng(20260808)
    out = {}

    cl_dirty, sym_dirty = C.load_claims(clean=False)
    cl_clean, sym_clean = C.load_claims(clean=True)
    fac, fsym = C.load_factors()
    z_base, ssym, _ = C.build_stats_matrix('base')
    z_usd, _, _ = C.build_stats_matrix('usd_close')

    common_dirty = sorted(set(sym_dirty) & set(ssym) & set(fsym))
    common_clean = sorted(set(sym_clean) & set(ssym) & set(fsym))
    L = sorted(set(common_dirty) & set(common_clean))
    out['sets'] = {'contaminated_n': len(common_dirty), 'clean_n': len(common_clean),
                   'L_n': len(L), 'L': L}

    # ---------------------------------------------------------------- A + B --
    print("=" * 78)
    print("A. WHAT THE STUB ROWS LOOK LIKE IN C AFTER ROW-NORMALISATION")
    print("=" * 78)
    print(f"row sums, contaminated C: min={cl_dirty.sum(1).min():.15f} "
          f"max={cl_dirty.sum(1).max():.15f}   (Eq.1, zero_shot_classifier.py:76-77)")
    out['row_sum_check'] = {
        'contaminated_min': float(cl_dirty.sum(1).min()),
        'contaminated_max': float(cl_dirty.sum(1).max()),
        'clean_min': float(cl_clean.sum(1).min()),
        'clean_max': float(cl_clean.sum(1).max()),
    }

    idx_d = [sym_dirty.index(s) for s in common_dirty]
    diag_dirty = row_diagnostics(cl_dirty[idx_d], common_dirty, 'contaminated n=37')
    idx_c = [sym_clean.index(s) for s in common_clean]
    diag_clean = row_diagnostics(cl_clean[idx_c], common_clean, 'clean n=43')
    out['diagnostics_contaminated'] = diag_dirty
    out['diagnostics_clean'] = diag_clean

    print(f"\ncontaminated-corpus centroid: "
          + ", ".join(f"{c}={v:.3f}" for c, v in zip(CATS, diag_dirty['centroid'])))
    print(f"medoid (modal profile) row: {diag_dirty['medoid']}\n")
    hdr = f"{'sym':7s} {'entropy':>8s} {'rk':>3s} {'d_cent':>7s} {'rk':>3s} {'maha':>6s} {'rk':>3s} {'lev':>6s} {'rk':>3s} {'max|z|':>7s} {'rk':>3s}  on"
    print(hdr)
    for s in common_dirty:
        r = diag_dirty['rows'][s]
        star = ' <<<' if s in STUBS else (' (repl)' if s in REPLACED else '')
        print(f"{s:7s} {r['entropy']:8.4f} {r['entropy_rank']:3d} "
              f"{r['euclid_from_centroid']:7.4f} {r['euclid_from_centroid_rank']:3d} "
              f"{r['mahalanobis']:6.3f} {r['mahalanobis_rank']:3d} "
              f"{r['hat_leverage']:6.4f} {r['hat_leverage_rank']:3d} "
              f"{r['max_abs_z']:7.3f} {r['max_abs_z_rank']:3d}  "
              f"{r['argmax_abs_z_category']}{star}")

    # ------------------------------------------------------------------- B --
    print("\n" + "=" * 78)
    print("B. SOURCE-DOCUMENT SIZE (extracted_chunks.json) AND WHAT SURVIVES Eq.(1)")
    print("=" * 78)
    ch_d = json.load(open(C.OUT / '_pre_rerun_backup_jun27-nlp' / 'extracted_chunks.json'))
    ch_c = json.load(open(C.OUT / 'nlp' / 'extracted_chunks.json'))
    wc_d = {k: v['word_count'] for k, v in ch_d.items()}
    nc_d = {k: len(v['chunks']) for k, v in ch_d.items()}
    pg_d = {k: v.get('page_count') for k, v in ch_d.items()}
    out['contaminated_doc_sizes'] = {
        s: {'word_count': wc_d.get(s), 'n_chunks': nc_d.get(s), 'page_count': pg_d.get(s)}
        for s in common_dirty}
    out['clean_doc_sizes'] = {
        s: {'word_count': ch_c[s]['word_count'], 'n_chunks': len(ch_c[s]['chunks'])}
        for s in common_clean if s in ch_c}

    srt = sorted(((wc_d[s], s) for s in common_dirty))
    print(f"{'rank':>4s} {'sym':7s} {'words':>7s} {'chunks':>6s} {'pages':>5s}  status")
    for i, (w, s) in enumerate(srt[:12], 1):
        st = 'STUB (dropped)' if s in STUBS else ('wrong-doc (replaced)' if s in REPLACED else '')
        print(f"{i:4d} {s:7s} {w:7d} {nc_d[s]:6d} {str(pg_d[s]):>5s}  {st}")
    print(f"  ... median word count over the 37 = {int(np.median(list(wc_d[s] for s in common_dirty)))}")

    # ------------------------------------------------------------------- C --
    print("\n" + "=" * 78)
    print("C. DOES DOCUMENT SIZE PREDICT AN OUTLYING COMPOSITION?")
    print("=" * 78)
    corr = {}
    for corpus, diag, wc, nc, syms in [
            ('contaminated37', diag_dirty, wc_d, nc_d, common_dirty),
            ('clean43', diag_clean,
             {s: ch_c[s]['word_count'] for s in common_clean},
             {s: len(ch_c[s]['chunks']) for s in common_clean}, common_clean)]:
        w = np.array([wc[s] for s in syms], float)
        k = np.array([nc[s] for s in syms], float)
        block = {}
        for key in ['euclid_from_centroid', 'mahalanobis', 'hat_leverage',
                    'max_abs_z', 'entropy']:
            y = np.array([diag['rows'][s][key] for s in syms])
            block[key] = {
                'spearman_vs_words': float(sps.spearmanr(w, y).statistic),
                'spearman_vs_chunks': float(sps.spearmanr(k, y).statistic),
                'pearson_vs_inv_sqrt_chunks': float(sps.pearsonr(1 / np.sqrt(k), y)[0]),
                'p_pearson_vs_inv_sqrt_chunks': float(sps.pearsonr(1 / np.sqrt(k), y)[1]),
            }
            print(f"  [{corpus:14s}] {key:22s} rho(words)={block[key]['spearman_vs_words']:+.3f} "
                  f"rho(chunks)={block[key]['spearman_vs_chunks']:+.3f} "
                  f"r(1/sqrt(chunks))={block[key]['pearson_vs_inv_sqrt_chunks']:+.3f} "
                  f"(p={block[key]['p_pearson_vs_inv_sqrt_chunks']:.4f})")
        corr[corpus] = block
    out['size_vs_outlyingness'] = corr

    # LOO impact vs the diagnostics, on the contaminated 37 (base build, as in the paper)
    A37 = arm(cl_dirty, sym_dirty, z_base, ssym, fac, fsym, common_dirty,
              'contaminated 37 (base build)')
    out['contaminated37_base'] = A37
    imp = A37['impacts']
    blk = {}
    for key in ['euclid_from_centroid', 'mahalanobis', 'hat_leverage', 'max_abs_z', 'entropy']:
        y = np.array([diag_dirty['rows'][s][key] for s in common_dirty])
        x = np.array([abs(imp[s]) for s in common_dirty])
        xs = np.array([imp[s] for s in common_dirty])
        blk[key] = {'spearman_vs_abs_impact': float(sps.spearmanr(y, x).statistic),
                    'spearman_vs_signed_impact': float(sps.spearmanr(y, xs).statistic)}
        print(f"  [LOO |impact|  ] {key:22s} rho={blk[key]['spearman_vs_abs_impact']:+.3f} "
              f"(signed {blk[key]['spearman_vs_signed_impact']:+.3f})")
    wn = np.array([wc_d[s] for s in common_dirty], float)
    blk['word_count'] = {
        'spearman_vs_abs_impact': float(sps.spearmanr(wn, [abs(imp[s]) for s in common_dirty]).statistic),
        'spearman_vs_signed_impact': float(sps.spearmanr(wn, [imp[s] for s in common_dirty]).statistic)}
    print(f"  [LOO |impact|  ] {'word_count':22s} rho={blk['word_count']['spearman_vs_abs_impact']:+.3f} "
          f"(signed {blk['word_count']['spearman_vs_signed_impact']:+.3f})")
    out['loo_vs_diagnostics_contaminated37'] = blk

    # ------------------------------------------------------------------- D --
    print("\n" + "=" * 78)
    print("D. CONTROLLED PERTURBATION -- inject 3 rows, market side held identical")
    print("=" * 78)
    stub_profiles = np.array([cl_dirty[sym_dirty.index(s)] for s in STUBS])
    out['stub_profiles'] = {s: [float(v) for v in cl_dirty[sym_dirty.index(s)]] for s in STUBS}

    for build, zmat in [('base', z_base), ('usd_close', z_usd)]:
        print(f"\n--- volume build: {build} ---")
        block = {}
        for basename, base_claims, base_syms, base_restrict in [
                ('cleanL34', cl_clean, sym_clean, L),
                ('clean43', cl_clean, sym_clean, common_clean)]:
            bidx = [base_syms.index(s) for s in base_restrict]
            B = cl_clean[bidx]
            centroid = B.mean(axis=0)
            print(f"\n  base corpus = {basename} (n={len(base_restrict)})")
            sub = {}
            sub['baseline_no_injection'] = arm(
                base_claims, base_syms, zmat, ssym, fac, fsym, base_restrict,
                f'{basename} baseline (no injection)')

            det = {
                'stub_actual': stub_profiles,
                'uniform': np.tile(np.full(10, 0.1), (3, 1)),
                'centroid': np.tile(centroid, (3, 1)),
            }
            for nm, prof in det.items():
                Mc, syms2 = make_injected(B, base_restrict, STUBS, prof)
                sub[nm] = arm(Mc, syms2, zmat, ssym, fac, fsym, base_restrict + STUBS,
                              f'{basename} + 3x {nm}')

            # stochastic arms
            for nm, ndraw in [('dirichlet_flat', 200), ('resample_real', 200),
                              ('stub_shuffled', 200), ('dirichlet_corpus', 200)]:
                counts, maxima, padded, helpsets = [], [], [], []
                for d in range(ndraw):
                    if nm == 'dirichlet_flat':
                        prof = rng.dirichlet(np.ones(10), size=3)
                    elif nm == 'resample_real':
                        prof = B[rng.choice(len(B), size=3, replace=False)]
                    elif nm == 'stub_shuffled':
                        prof = np.array([sp[rng.permutation(10)] for sp in stub_profiles])
                    else:   # dirichlet_corpus: match corpus mean & dispersion
                        m = centroid
                        v = B.var(axis=0, ddof=1)
                        # method-of-moments concentration for a Dirichlet
                        s0 = float(np.median((m * (1 - m) / np.maximum(v, 1e-12)) - 1))
                        s0 = max(s0, 1.0)
                        prof = rng.dirichlet(m * s0, size=3)
                    Mc, syms2 = make_injected(B, base_restrict, STUBS, prof)
                    r = arm(Mc, syms2, zmat, ssym, fac, fsym, base_restrict + STUBS,
                            f'{nm}#{d}', verbose=False)
                    counts.append(r['n_helpers'])
                    maxima.append(r['max_impact'])
                    padded.append(r['padded_phi'])
                    helpsets.append(r['helpers'])
                counts = np.array(counts); maxima = np.array(maxima)
                sub[nm] = {
                    'n_draws': ndraw,
                    'n_helpers_mean': float(counts.mean()),
                    'n_helpers_median': float(np.median(counts)),
                    'n_helpers_p95': float(np.percentile(counts, 95)),
                    'n_helpers_max': int(counts.max()),
                    'frac_draws_with_ge1_helper': float((counts >= 1).mean()),
                    'frac_draws_with_ge4_helpers': float((counts >= 4).mean()),
                    'max_impact_mean': float(maxima.mean()),
                    'max_impact_p95': float(np.percentile(maxima, 95)),
                    'max_impact_max': float(maxima.max()),
                    'padded_phi_mean': float(np.mean(padded)),
                    'frac_draws_XMR_helper': float(np.mean(['XMR' in h for h in helpsets])),
                    'frac_draws_CRV_helper': float(np.mean(['CRV' in h for h in helpsets])),
                    'frac_draws_SOL_helper': float(np.mean(['SOL' in h for h in helpsets])),
                    'frac_draws_injected_helper': float(np.mean(
                        [any(s in h for s in STUBS) for h in helpsets])),
                }
                print(f"  [{basename} + 3x {nm:17s}] draws={ndraw} "
                      f"E[#helpers]={counts.mean():.2f} med={np.median(counts):.0f} "
                      f"p95={np.percentile(counts, 95):.0f} max={counts.max()} "
                      f"P(>=1)={(counts >= 1).mean():.3f} P(>=4)={(counts >= 4).mean():.3f} "
                      f"E[max impact]={maxima.mean():+.4f}")
            block[basename] = sub
        out[f'injection_{build}'] = block

    json.dump(out, open(OUTF, 'w'), indent=2)
    print(f"\nwrote {OUTF}")


if __name__ == '__main__':
    main()
