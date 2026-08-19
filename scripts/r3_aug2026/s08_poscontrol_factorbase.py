#!/usr/bin/env python3
"""STEP 8 (Task 2b, second leg) -- the FACTOR-carrier positive control on
outlier-robust bases.

Reproduces scripts/positive_control_sim.py (padded estimator + congruence.py
permutation test, real 43x2 CP asset factors as carrier) and repeats it with
BTC excluded / winsorised / rank-inverse-normal carriers.  The paper describes
this carrier as degenerate ("Bitcoin's level dominates the leading factor",
main-blinded.tex:211) and reports its ideal MDE as phi ~ 0.40.
"""
import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps

import common as C
from alignment.congruence import CongruenceCoefficient

logging.disable(logging.INFO)   # the pipeline logs an INFO line per Procrustes call

OUTF = Path(__file__).parent / "results_08_poscontrol_factorbase.json"
RHO_FACTORS = 0.95
REPS, PERMS = 500, 200
GRID = [0.2, 0.3, 0.5, 0.65, 0.8]
SEED = 20260627


def load_factor_base():
    fac, fsym = C.load_factors()
    csym = C.load_meta(C.OUT / 'nlp' / 'claims_matrix_meta.json')['symbols']
    z, ssym, _ = C.build_stats_matrix('base')
    common = sorted(set(fsym) & set(csym) & set(ssym))
    return fac[[fsym.index(s) for s in common]].astype(float), common


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


def synth_claims_like(B, true_phi, rng, embed_dim=10, reliability=1.0):
    n, k = B.shape
    phi = true_phi * np.sqrt(reliability * RHO_FACTORS) if reliability < 1.0 else true_phi
    phi = float(np.clip(phi, 1e-6, 0.999999))
    H = np.empty((n, k))
    for j in range(k):
        fc = B[:, j] - B[:, j].mean()
        v = float(np.var(fc))
        if v <= 0:
            H[:, j] = rng.standard_normal(n); continue
        H[:, j] = fc + rng.standard_normal(n) * np.sqrt(v * (1 - phi ** 2) / phi ** 2)
    G = rng.standard_normal((embed_dim, k))
    U, _ = np.linalg.qr(G)
    return H @ U.T


def run_cell(B, true_phi, reliability, cc, rng, reps=REPS):
    nk = B.shape[1]
    det = np.empty(reps, bool); obs = np.empty(reps); real = np.empty(reps)
    for r in range(reps):
        A = synth_claims_like(B, true_phi, rng, 10, reliability)
        pt = cc.permutation_test(A, B, n_permutations=PERMS)
        det[r] = pt['p_value'] < 0.05
        obs[r] = pt['observed_phi']
        cols = cc.matrix_congruence(A, B)['column_phis']
        real[r] = float(np.mean(np.abs(cols[:nk])))
    p = float(det.mean())
    return {'true_phi': true_phi, 'reliability': reliability, 'power': p,
            'power_se': float(np.sqrt(p * (1 - p) / reps)),
            'mean_recovered_phi_pipeline': float(obs.mean()),
            'mean_recovered_phi_factor_cols': float(real.mean())}


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


def main():
    B0, common = load_factor_base()
    btc = common.index('BTC')
    # degeneracy diagnostics on the published carrier
    diag = {
        'btc_z_on_factor1': float((B0[btc, 0] - B0[:, 0].mean()) / B0[:, 0].std()),
        'btc_z_on_factor2': float((B0[btc, 1] - B0[:, 1].mean()) / B0[:, 1].std()),
        'factor1_share_of_squared_norm_from_btc':
            float(B0[btc, 0] ** 2 / (B0[:, 0] ** 2).sum()),
        'corr_f1_f2': float(np.corrcoef(B0[:, 0], B0[:, 1])[0, 1]),
        'excess_kurtosis': sps.kurtosis(B0, axis=0).tolist(),
    }
    print("published carrier degeneracy:", json.dumps(diag, indent=1))

    bases = {
        'published': B0,
        'no_btc': np.delete(B0, btc, axis=0),
        'winsor05': winsorise(B0),
        'rank_inv_normal': rank_inv_normal(B0),
    }
    cc = CongruenceCoefficient()
    out = {'config': {'reps': REPS, 'perms': PERMS, 'grid': GRID, 'seed': SEED},
           'published_carrier_diagnostics': diag, 'bases': {}}
    for bname, B in bases.items():
        t0 = time.time()
        rng = np.random.default_rng(SEED)
        entry = {'n': B.shape[0], 'scenarios': {}}
        for sname, rel in [('ideal_1.0', 1.0), ('rho_0.314', 0.314)]:
            cells = [run_cell(B, phi, rel, cc, rng) for phi in GRID]
            mde = interp_mde([c['true_phi'] for c in cells], [c['power'] for c in cells])
            entry['scenarios'][sname] = {'reliability': rel, 'cells': cells,
                                         'mde_power80': mde}
            print(f"[{bname:16s}] {sname:10s} power={[round(c['power'],3) for c in cells]} "
                  f"MDE={mde if mde is None else round(mde,3)} "
                  f"recovered(factor cols)={[round(c['mean_recovered_phi_factor_cols'],3) for c in cells]}")
        entry['seconds'] = round(time.time() - t0, 1)
        out['bases'][bname] = entry
        print(f"  ({entry['seconds']}s)\n")

    json.dump(out, open(OUTF, 'w'), indent=2)
    print(f"wrote {OUTF}")


if __name__ == '__main__':
    main()
