#!/usr/bin/env python3
"""
VERIFICATION: positive-control power for the *matched-dimension* estimator
(the paper's PRIMARY estimator, phi=0.303) using the real 43x7 STATISTICS
matrix as the calibration base -- mirroring the headline test's
10D -> reduce-to-7D -> recover-phi geometry EXACTLY.

This checks whether the MDE / power story the paper reports (derived from the
factor-base + zero-padded pipeline) also holds for the matched estimator with
the statistics base, as a referee asked.
"""
import json, sys, time
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from alignment.matched_dimension_analysis import tucker_phi, reduce_dimensions

RHO_FACTORS = 0.95

def load_stats_base():
    out = CODE / "outputs"
    stats = np.load(out / "market" / "stats_matrix.npy")
    ssym = json.load(open(out / "market" / "stats_matrix_meta.json"))["symbols"]
    csym = json.load(open(out / "nlp" / "claims_matrix_meta.json"))["symbols"]
    fsym = json.load(open(out / "tensor" / "cp_factors_meta.json"))["symbols"]
    common = sorted(set(ssym) & set(csym) & set(fsym))
    S = stats[[ssym.index(s) for s in common]].astype(float)
    return S, common

def synth_claims_like(S, true_phi, rng, embed_dim=10, reliability=1.0):
    """43x10 claims-like: each of the 7 stats columns corrupted to per-column
    correlation phi, embedded into a 7-dim subspace of the 10-dim claims space
    via a random orthonormal map."""
    n, k = S.shape  # 43 x 7
    phi = true_phi * np.sqrt(reliability * RHO_FACTORS) if reliability < 1.0 else true_phi
    phi = float(np.clip(phi, 1e-6, 0.999999))
    H = np.empty((n, k))
    for j in range(k):
        fc = S[:, j] - S[:, j].mean()
        var_fc = float(np.var(fc))
        if var_fc <= 0:
            H[:, j] = rng.standard_normal(n); continue
        noise_var = var_fc * (1.0 - phi**2) / (phi**2)
        H[:, j] = fc + rng.standard_normal(n) * np.sqrt(noise_var)
    G = rng.standard_normal((embed_dim, k))
    U, _ = np.linalg.qr(G)        # embed_dim x k, U^T U = I_k
    return H @ U.T                # n x embed_dim

def perm_p(A_red, S, observed, rng, perms=200):
    n = S.shape[0]
    null = np.empty(perms)
    for i in range(perms):
        null[i] = tucker_phi(A_red, S[rng.permutation(n)])
    return float(np.mean(np.abs(null) >= abs(observed)))

def run_cell(S, true_phi, reps, perms, reliability, rng):
    det = np.empty(reps, bool); rec = np.empty(reps)
    for r in range(reps):
        A = synth_claims_like(S, true_phi, rng, embed_dim=10, reliability=reliability)
        A_red = reduce_dimensions(A, 7)          # 10D -> 7D, same as headline
        obs = tucker_phi(A_red, S)               # matched estimator
        rec[r] = obs
        det[r] = perm_p(A_red, S, obs, rng, perms) < 0.05
    power = float(det.mean())
    return {"true_phi": float(true_phi), "reliability": float(reliability),
            "power": power, "power_se": float(np.sqrt(power*(1-power)/reps)),
            "mean_recovered_phi": float(rec.mean())}

def interp_mde(phis, powers, target=0.80):
    phis=np.asarray(phis,float); powers=np.asarray(powers,float)
    o=np.argsort(phis); phis,powers=phis[o],powers[o]
    if powers.max()<target: return None
    for i in range(1,len(phis)):
        if powers[i-1]<target<=powers[i]:
            p0,p1=powers[i-1],powers[i]; x0,x1=phis[i-1],phis[i]
            return float(x1) if p1==p0 else float(x0+(target-p0)*(x1-x0)/(p1-p0))
    return float(phis[0]) if powers[0]>=target else None

def main():
    t=time.time()
    rng=np.random.default_rng(20260627)
    S, common = load_stats_base()
    print(f"[pc-matched] base = real 43x{S.shape[1]} stats, n={S.shape[0]}")
    reps, perms = 500, 200
    grid=[0.2,0.3,0.5,0.65,0.8]
    scenarios={"ideal_1.0":1.0, "rho_0.50":0.50, "rho_0.314":0.314, "rho_0.07":0.07}
    results={}
    for sname,rel in scenarios.items():
        cells=[run_cell(S,phi,reps,perms,rel,rng) for phi in grid]
        mde=interp_mde([c["true_phi"] for c in cells],[c["power"] for c in cells])
        results[sname]={"reliability":rel,"cells":cells,"mde_power80":mde}
        print(f"\n[{sname}] rel={rel}")
        for c in cells:
            print(f"  phi={c['true_phi']:.2f} power={c['power']:.3f} recovered={c['mean_recovered_phi']:.3f}")
        print(f"  MDE(80%) = {('%.3f'%mde) if mde else 'NOT REACHED'}")
    out={"description":"matched-estimator positive control, real 43x7 stats base, mirrors headline 10D->7D->recover",
         "config":{"reps":reps,"perms":perms,"grid":grid,"seed":20260627,"base":"real 43x7 stats","n":S.shape[0]},
         "scenarios":results}
    p=CODE / "outputs" / "positive_control_matched_statsbase.json"
    json.dump(out, open(p,"w"), indent=2)
    print(f"\nwrote {p}  ({time.time()-t:.1f}s)")

if __name__=="__main__":
    main()
