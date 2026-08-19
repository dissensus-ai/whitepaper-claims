#!/usr/bin/env python3
"""STEP 5 (Task 2b) -- matched-geometry positive control on OUTLIER-ROBUST bases.

The paper's MDE ~0.29 (ideal) / ~0.44 (realistic) come from
scripts/positive_control_matched_statsbase.py -> outputs/positive_control_matched_statsbase.json
which uses the real 43x7 z-scored statistics matrix as the carrier.  That matrix
is heavy-tailed (BTC and a handful of high-volume assets are multi-sigma on
several columns).  We repeat the identical simulation on four carriers:

  published        the real 43x7 z-scored stats matrix (reproduction check)
  no_btc           same, BTC row dropped (n = 42)
  winsor05         each column winsorised at the 5th/95th percentile
  rank_inv_normal  each column replaced by its inverse-normal rank score

Config (reps, perms, grid, seed) is identical to the published script.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps

import common as C
from alignment.matched_dimension_analysis import tucker_phi, reduce_dimensions

OUTF = Path(__file__).parent / "results_05_poscontrol_statsbase.json"
RHO_FACTORS = 0.95
REPS, PERMS = 500, 200
GRID = [0.2, 0.3, 0.5, 0.65, 0.8]
SEED = 20260627


def load_stats_base(volume_basis='base'):
    z, syms, _ = C.build_stats_matrix(volume_basis)
    csym = C.load_meta(C.OUT / 'nlp' / 'claims_matrix_meta.json')['symbols']
    fsym = C.load_meta(C.OUT / 'tensor' / 'cp_factors_meta.json')['symbols']
    common = sorted(set(syms) & set(csym) & set(fsym))
    return z[[syms.index(s) for s in common]].astype(float), common


def winsorise(M, lo=5, hi=95):
    W = M.copy()
    for j in range(M.shape[1]):
        a, b = np.percentile(M[:, j], [lo, hi])
        W[:, j] = np.clip(M[:, j], a, b)
    return W


def rank_inv_normal(M):
    R = np.empty_like(M)
    n = M.shape[0]
    for j in range(M.shape[1]):
        r = sps.rankdata(M[:, j])
        R[:, j] = sps.norm.ppf((r - 0.5) / n)
    return R


def synth_claims_like(S, true_phi, rng, embed_dim=10, reliability=1.0):
    n, k = S.shape
    phi = true_phi * np.sqrt(reliability * RHO_FACTORS) if reliability < 1.0 else true_phi
    phi = float(np.clip(phi, 1e-6, 0.999999))
    H = np.empty((n, k))
    for j in range(k):
        fc = S[:, j] - S[:, j].mean()
        v = float(np.var(fc))
        if v <= 0:
            H[:, j] = rng.standard_normal(n); continue
        H[:, j] = fc + rng.standard_normal(n) * np.sqrt(v * (1 - phi ** 2) / phi ** 2)
    G = rng.standard_normal((embed_dim, k))
    U, _ = np.linalg.qr(G)
    return H @ U.T


def perm_p(A_red, S, observed, rng, perms=PERMS):
    n = S.shape[0]
    null = np.empty(perms)
    for i in range(perms):
        null[i] = tucker_phi(A_red, S[rng.permutation(n)])
    return float(np.mean(np.abs(null) >= abs(observed)))


def run_cell(S, true_phi, reliability, rng, reps=REPS):
    det = np.empty(reps, bool); rec = np.empty(reps)
    for r in range(reps):
        A = synth_claims_like(S, true_phi, rng, 10, reliability)
        A_red = reduce_dimensions(A, S.shape[1])
        obs = tucker_phi(A_red, S)
        rec[r] = obs
        det[r] = perm_p(A_red, S, obs, rng) < 0.05
    p = float(det.mean())
    return {'true_phi': true_phi, 'reliability': reliability, 'power': p,
            'power_se': float(np.sqrt(p * (1 - p) / reps)),
            'mean_recovered_phi': float(rec.mean())}


def interp_mde(phis, powers, target=0.80):
    phis = np.asarray(phis, float); powers = np.asarray(powers, float)
    o = np.argsort(phis); phis, powers = phis[o], powers[o]
    if powers.max() < target:
        return None
    for i in range(1, len(phis)):
        if powers[i - 1] < target <= powers[i]:
            p0, p1 = powers[i - 1], powers[i]; x0, x1 = phis[i - 1], phis[i]
            return float(x1) if p1 == p0 else float(x0 + (target - p0) * (x1 - x0) / (p1 - p0))
    return float(phis[0]) if powers[0] >= target else None


def base_diagnostics(S, name):
    mx = np.abs(S).max(axis=0)
    kurt = sps.kurtosis(S, axis=0, fisher=True)
    return {'name': name, 'shape': list(S.shape),
            'max_abs_z_per_col': mx.tolist(),
            'excess_kurtosis_per_col': kurt.tolist(),
            'max_leverage_row': int(np.argmax((S ** 2).sum(axis=1)))}


def main():
    S0, common = load_stats_base('base')
    btc = common.index('BTC')
    bases = {
        'published': (S0, common),
        'no_btc': (np.delete(S0, btc, axis=0), [s for s in common if s != 'BTC']),
        'winsor05': (winsorise(S0), common),
        'rank_inv_normal': (rank_inv_normal(S0), common),
    }
    out = {'config': {'reps': REPS, 'perms': PERMS, 'grid': GRID, 'seed': SEED},
           'bases': {}}
    for bname, (S, syms) in bases.items():
        t0 = time.time()
        rng = np.random.default_rng(SEED)
        entry = {'n': S.shape[0], 'diagnostics': base_diagnostics(S, bname),
                 'scenarios': {}}
        entry['diagnostics']['max_leverage_symbol'] = syms[entry['diagnostics']['max_leverage_row']]
        for sname, rel in [('ideal_1.0', 1.0), ('rho_0.50', 0.50), ('rho_0.314', 0.314)]:
            cells = [run_cell(S, phi, rel, rng) for phi in GRID]
            mde = interp_mde([c['true_phi'] for c in cells], [c['power'] for c in cells])
            entry['scenarios'][sname] = {'reliability': rel, 'cells': cells,
                                         'mde_power80': mde}
            print(f"[{bname:16s}] {sname:10s} power={[round(c['power'],3) for c in cells]} "
                  f"MDE={mde if mde is None else round(mde,3)}")
        entry['seconds'] = round(time.time() - t0, 1)
        out['bases'][bname] = entry
        print(f"  ({entry['seconds']}s, max|z| per column: "
              f"{[round(v,1) for v in entry['diagnostics']['max_abs_z_per_col']]}, "
              f"highest-leverage asset = {entry['diagnostics']['max_leverage_symbol']})\n")

    json.dump(out, open(OUTF, 'w'), indent=2)
    print(f"wrote {OUTF}")


if __name__ == '__main__':
    main()
