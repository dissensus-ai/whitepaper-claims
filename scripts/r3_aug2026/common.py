#!/usr/bin/env python3
"""
Shared helpers for the Aug-2026 R3 re-run (whitepaper-claims / FINI-D-26-01252).

Everything here reuses the paper's own estimators from code/src/. Nothing is
re-implemented except where explicitly noted (Brokken congruence rotation).

Volume bases
------------
'base'        : daily sum of CCXT `volume` (BASE-ASSET units)  -- the published pipeline
'usd_close'   : daily sum of volume * close                    (USD notional, close price)
'usd_typical' : daily sum of volume * (high+low+close)/3       (USD notional, VWAP-ish)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.linalg import orthogonal_procrustes

CODE = Path(__file__).resolve().parents[2]          # .../whitepaper-claims/code
sys.path.insert(0, str(CODE / "src"))

from alignment.congruence import CongruenceCoefficient          # noqa: E402
from alignment.matched_dimension_analysis import (               # noqa: E402
    tucker_phi as matched_tucker_phi,
    reduce_dimensions,
)
from alignment.alternative_metrics import (                      # noqa: E402
    rv_coefficient, distance_correlation, cca_correlation, pls_score,
    bootstrap_metric, permutation_test as alt_permutation_test,
)

OUT = CODE / "outputs"
MARKET_DIR = CODE / "data" / "market"

STATISTICS = ['mean_return', 'volatility', 'sharpe', 'max_drawdown',
              'avg_volume', 'vol_volatility', 'trend']


# --------------------------------------------------------------------------- #
# 1. market summary statistics with a configurable volume basis
# --------------------------------------------------------------------------- #
def _compute_max_drawdown(prices: pd.Series) -> float:
    cummax = prices.cummax()
    return abs(((prices - cummax) / cummax).min())


def _compute_trend(prices: pd.Series) -> float:
    x = np.arange(len(prices))
    slope, *_ = sps.linregress(x, prices.values)
    return slope / prices.mean()


def compute_stats_for_df(df: pd.DataFrame, volume_basis: str = 'base',
                         cv_follows_basis: bool = True) -> dict:
    """Mirror src/market/summary_statistics.py::compute_stats with a switchable
    volume basis.  `cv_follows_basis=False` keeps vol_volatility on the
    published base-asset series (isolates the avg_volume change)."""
    df = df.copy()
    df['date'] = df['timestamp'].dt.date

    if volume_basis == 'base':
        df['vol_use'] = df['volume']
    elif volume_basis == 'usd_close':
        df['vol_use'] = df['volume'] * df['close']
    elif volume_basis == 'usd_typical':
        df['vol_use'] = df['volume'] * (df['high'] + df['low'] + df['close']) / 3.0
    else:
        raise ValueError(volume_basis)

    daily = df.groupby('date').agg({'close': 'last',
                                    'volume': 'sum',
                                    'vol_use': 'sum'}).reset_index()

    daily_returns = np.log(daily['close'] / daily['close'].shift(1)).dropna()
    cv_series = daily['vol_use'] if cv_follows_basis else daily['volume']

    return {
        'mean_return': daily_returns.mean() * 252,
        'volatility': daily_returns.std() * np.sqrt(252),
        'sharpe': (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
                  if daily_returns.std() > 0 else 0,
        'max_drawdown': _compute_max_drawdown(daily['close']),
        'avg_volume': np.log1p(daily['vol_use'].mean()),
        'vol_volatility': cv_series.std() / cv_series.mean()
                          if cv_series.mean() > 0 else 0,
        'trend': _compute_trend(daily['close']),
    }


def build_stats_matrix(volume_basis: str = 'base', market_dir: Path = MARKET_DIR,
                       cv_follows_basis: bool = True):
    """Returns (z-scored N x 7 matrix, symbols, raw N x 7 matrix)."""
    files = sorted(Path(market_dir).glob("*.parquet"))
    if not files:
        raise ValueError(f"no parquet in {market_dir}")
    symbols, rows = [], []
    for f in files:
        sym = f.stem.replace('_ohlcv', '').upper()
        df = pd.read_parquet(f)
        if df.empty:
            continue
        s = compute_stats_for_df(df, volume_basis, cv_follows_basis)
        symbols.append(sym)
        rows.append([s[k] for k in STATISTICS])
    raw = np.asarray(rows, float)
    z = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-8)
    return z, symbols, raw


# --------------------------------------------------------------------------- #
# 2. matrix loading / alignment to the common asset set
# --------------------------------------------------------------------------- #
def load_meta(p: Path):
    return json.load(open(p))


def load_claims(clean: bool = True):
    if clean:
        M = np.load(OUT / 'nlp' / 'claims_matrix.npy')
        sym = load_meta(OUT / 'nlp' / 'claims_matrix_meta.json')['symbols']
    else:  # pre-Jun-27 contaminated build
        b = OUT / '_pre_rerun_backup_jun27-nlp'
        M = np.load(b / 'claims_matrix.npy')
        sym = load_meta(b / 'claims_matrix_meta.json')['symbols']
    return M, sym


def load_factors():
    M = np.load(OUT / 'tensor' / 'cp_asset_factors.npy')
    sym = load_meta(OUT / 'tensor' / 'cp_factors_meta.json')['symbols']
    return M, sym


def align_three(claims, claims_sym, stats, stats_sym, factors, factors_sym,
                restrict=None):
    common = sorted(set(claims_sym) & set(stats_sym) & set(factors_sym))
    if restrict is not None:
        common = sorted(set(common) & set(restrict))
    return (claims[[claims_sym.index(s) for s in common]],
            stats[[stats_sym.index(s) for s in common]],
            factors[[factors_sym.index(s) for s in common]],
            common)


# --------------------------------------------------------------------------- #
# 3. the two headline estimators, exactly as published
# --------------------------------------------------------------------------- #
_CC = CongruenceCoefficient()


def padded_phi(A: np.ndarray, B: np.ndarray) -> float:
    """The paper's 'zero-padded' phi: src/alignment/congruence.py
    (mean ABSOLUTE column congruence over max(dA,dB) slots; padded columns
    score exactly 0.0 and ARE included in the mean)."""
    return _CC.matrix_congruence(A, B)['mean_phi']


def padded_columns(A, B):
    return _CC.matrix_congruence(A, B)['column_phis']


def padded_perm_p(A, B, n_perm=1000, seed=42) -> float:
    """One-sided permutation p, as in congruence.py::permutation_test."""
    return _CC.permutation_test(A, B, n_permutations=n_perm, seed=seed)['p_value']


def padded_boot(A, B, n_boot=1000, seed=42) -> tuple:
    r = _CC.bootstrap_confidence_interval(A, B, n_bootstrap=n_boot, seed=seed)
    return r['ci_lower'], r['ci_upper']


def matched_phi(X: np.ndarray, Y: np.ndarray, target: int) -> float:
    """The paper's PRIMARY 'dimension-matched' phi:
    src/alignment/matched_dimension_analysis.py (PCA-reduce the wider matrix to
    `target` dims, then mean SIGNED column congruence)."""
    return matched_tucker_phi(reduce_dimensions(X, target), Y)


def matched_perm_p(X, Y, target, n_perm=1000, seed=42) -> float:
    """Two-sided permutation p, as in matched_dimension_analysis.py."""
    Xr = reduce_dimensions(X, target)
    obs = matched_tucker_phi(Xr, Y)
    rng = np.random.default_rng(seed)
    n = Xr.shape[0]
    null = [matched_tucker_phi(Xr, Y[rng.permutation(n)]) for _ in range(n_perm)]
    return float(np.mean(np.abs(null) >= abs(obs)))


def matched_boot(X, Y, target, n_boot=1000, seed=42) -> tuple:
    Xr = reduce_dimensions(X, target)
    rng = np.random.default_rng(seed)
    n = Xr.shape[0]
    vals = [matched_tucker_phi(Xr[i], Y[i])
            for i in (rng.choice(n, size=n, replace=True) for _ in range(n_boot))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------- #
# 4. alternative metrics bundle (RV / dCor / CCA / PLS -- there is NO HSIC)
# --------------------------------------------------------------------------- #
def alt_metrics(X, Y, n_boot=1000, n_perm=1000):
    out = {}
    rv = rv_coefficient(X, Y)
    out['rv'] = {'value': float(rv),
                 'p_value': float(alt_permutation_test(X, Y, rv_coefficient, rv, n_perm))}
    dc = distance_correlation(X, Y)
    out['dcor'] = {'value': float(dc),
                   'p_value': float(alt_permutation_test(X, Y, distance_correlation, dc, n_perm))}
    cc = cca_correlation(X, Y)
    out['cca'] = {'value': float(cc['mean_correlation']),
                  'p_value': float(alt_permutation_test(
                      X, Y, lambda a, b: cca_correlation(a, b),
                      cc['mean_correlation'], n_perm))}
    pl = pls_score(X, Y)
    out['pls'] = {'value': float(pl['mean_correlation']),
                  'p_value': float(alt_permutation_test(
                      X, Y, lambda a, b: pls_score(a, b),
                      pl['mean_correlation'], n_perm))}
    return out


# --------------------------------------------------------------------------- #
# 5. Brokken (1983) direct congruence rotation
# --------------------------------------------------------------------------- #
def _colwise_congruence(Arot, B, use_abs=False, only_first=None):
    k = B.shape[1] if only_first is None else only_first
    vals = []
    for j in range(k):
        num = float(Arot[:, j] @ B[:, j])
        den = float(np.sqrt((Arot[:, j] @ Arot[:, j]) * (B[:, j] @ B[:, j])))
        if den > 0:
            vals.append(abs(num / den) if use_abs else num / den)
    return float(np.mean(vals)) if vals else 0.0


def brokken_rotation(A: np.ndarray, B: np.ndarray, n_iter: int = 500,
                     tol: float = 1e-12, seed: int = 0):
    """Brokken (1983) orthogonal direct-congruence rotation.

    Maximises  sum_j  <A q_j, b_j> / ||A q_j||   over orthonormal Q.
    Implemented as the standard majorisation / iterative-Procrustes scheme:
    at each step build the target T with columns  b_j / ||b_j||  scaled by the
    current column norms of A Q, then take the Procrustes solution towards T.
    Guaranteed monotone non-decreasing in the congruence criterion.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    p = A.shape[1]
    # start from the Frobenius-optimal (Schoenemann) rotation
    Q, _ = orthogonal_procrustes(A, B)
    bn = np.linalg.norm(B, axis=0)
    bn[bn == 0] = 1.0
    Bhat = B / bn                       # unit-norm columns
    prev = -np.inf
    for _ in range(n_iter):
        AQ = A @ Q
        s = np.linalg.norm(AQ, axis=0)   # column norms of the rotated A
        s[s == 0] = 1e-12
        # gradient of sum_j <a_j(Q), bhat_j> / ||a_j(Q)|| wrt AQ, expressed as a
        # Procrustes target
        T = np.empty_like(AQ)
        for j in range(p):
            u = AQ[:, j] / s[j]
            T[:, j] = (Bhat[:, j] - (u @ Bhat[:, j]) * u) / s[j] + u
        Qn, _ = orthogonal_procrustes(A, T)
        crit = _colwise_congruence(A @ Qn, B)
        if crit <= prev + tol:
            if crit > prev:
                Q = Qn
            break
        prev, Q = crit, Qn
    return Q


def congruence_max_rotation(A, B, use_abs=False, n_restart=40, seed=0,
                            n_iter=800):
    """Numerical congruence-maximising orthogonal rotation with random restarts.
    Returns (best_Q, best_criterion).  `use_abs` scores mean |phi| (the padded
    estimator's statistic); otherwise mean signed phi (the matched estimator's).
    """
    from scipy.optimize import minimize
    from scipy.linalg import expm

    A = np.asarray(A, float)
    B = np.asarray(B, float)
    p = A.shape[1]
    idx = np.triu_indices(p, 1)
    rng = np.random.default_rng(seed)

    def to_Q(theta, Q0):
        S = np.zeros((p, p))
        S[idx] = theta
        S = S - S.T
        return Q0 @ expm(S)

    def neg(theta, Q0):
        return -_colwise_congruence(A @ to_Q(theta, Q0), B, use_abs=use_abs)

    Q0_frob, _ = orthogonal_procrustes(A, B)
    best_Q, best_c = Q0_frob, _colwise_congruence(A @ Q0_frob, B, use_abs=use_abs)
    starts = [Q0_frob, brokken_rotation(A, B)]
    for _ in range(n_restart):
        G = rng.standard_normal((p, p))
        Qr, _r = np.linalg.qr(G)
        starts.append(Qr)
    for Q0 in starts:
        r = minimize(neg, np.zeros(len(idx[0])), args=(Q0,), method='L-BFGS-B',
                     options={'maxiter': n_iter, 'ftol': 1e-14, 'gtol': 1e-12})
        Q = to_Q(r.x, Q0)
        c = _colwise_congruence(A @ Q, B, use_abs=use_abs)
        if c > best_c:
            best_Q, best_c = Q, c
    return best_Q, float(best_c)


def center(M):
    return M - M.mean(axis=0)


def pad_to(A, B):
    m = max(A.shape[1], B.shape[1])
    if A.shape[1] < m:
        A = np.hstack([A, np.zeros((A.shape[0], m - A.shape[1]))])
    if B.shape[1] < m:
        B = np.hstack([B, np.zeros((B.shape[0], m - B.shape[1]))])
    return A, B
