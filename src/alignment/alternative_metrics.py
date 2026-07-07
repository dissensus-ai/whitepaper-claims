#!/usr/bin/env python3
"""
Alternative alignment metrics for cross-space comparison.

Implements:
- RV coefficient (Robert & Escoufier, 1976)
- Distance correlation (Székely et al., 2007)
- Canonical Correlation Analysis (CCA)
- Partial Least Squares (PLS)

These supplement Tucker's φ from Procrustes to address reviewer concerns
about reliance on a single alignment measure.
"""

import numpy as np
import json
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def rv_coefficient(X: np.ndarray, Y: np.ndarray) -> float:
    """
    RV coefficient (Robert & Escoufier, 1976).

    Measures similarity between two configuration matrices.
    RV ∈ [0, 1], with 1 indicating identical configurations.

    RV(X, Y) = trace(X'Y Y'X) / sqrt(trace(X'X X'X) * trace(Y'Y Y'Y))
    """
    # Center the matrices
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Compute cross-products
    XX = X @ X.T
    YY = Y @ Y.T

    # RV coefficient
    numerator = np.trace(XX @ YY)
    denominator = np.sqrt(np.trace(XX @ XX) * np.trace(YY @ YY))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Distance correlation (Székely, Rizzo, & Bakirov, 2007).

    Measures dependence between random vectors of arbitrary dimensions.
    dCor ∈ [0, 1], with 0 iff X and Y are independent.
    """
    n = X.shape[0]

    # Compute pairwise distance matrices
    a = squareform(pdist(X, 'euclidean'))
    b = squareform(pdist(Y, 'euclidean'))

    # Double-center the distance matrices
    a_row_mean = a.mean(axis=1, keepdims=True)
    a_col_mean = a.mean(axis=0, keepdims=True)
    a_grand_mean = a.mean()
    A = a - a_row_mean - a_col_mean + a_grand_mean

    b_row_mean = b.mean(axis=1, keepdims=True)
    b_col_mean = b.mean(axis=0, keepdims=True)
    b_grand_mean = b.mean()
    B = b - b_row_mean - b_col_mean + b_grand_mean

    # Distance covariance and variances
    dCov_sq = (A * B).sum() / (n * n)
    dVar_X_sq = (A * A).sum() / (n * n)
    dVar_Y_sq = (B * B).sum() / (n * n)

    # Distance correlation
    if dVar_X_sq <= 0 or dVar_Y_sq <= 0:
        return 0.0

    dCor_sq = dCov_sq / np.sqrt(dVar_X_sq * dVar_Y_sq)

    # dCor is sqrt of dCor^2, but handle numerical issues
    return np.sqrt(max(0, dCor_sq))


def cca_correlation(X: np.ndarray, Y: np.ndarray, n_components: int = None) -> dict:
    """
    Canonical Correlation Analysis.

    Returns canonical correlations and the mean canonical correlation.
    """
    # Determine max components
    n_samples = X.shape[0]
    max_components = min(X.shape[1], Y.shape[1], n_samples - 1)

    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)

    # Standardize
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    Y_scaled = scaler_y.fit_transform(Y)

    # Fit CCA
    cca = CCA(n_components=n_components, max_iter=1000)
    try:
        X_c, Y_c = cca.fit_transform(X_scaled, Y_scaled)

        # Compute canonical correlations
        correlations = []
        for i in range(n_components):
            r, _ = stats.pearsonr(X_c[:, i], Y_c[:, i])
            correlations.append(r)

        return {
            'canonical_correlations': correlations,
            'mean_correlation': np.mean(correlations),
            'first_correlation': correlations[0] if correlations else 0.0,
            'n_components': n_components
        }
    except Exception as e:
        return {
            'canonical_correlations': [],
            'mean_correlation': 0.0,
            'first_correlation': 0.0,
            'n_components': 0,
            'error': str(e)
        }


def pls_score(X: np.ndarray, Y: np.ndarray, n_components: int = None) -> dict:
    """
    Partial Least Squares correlation.

    Returns explained variance and correlation in latent space.
    """
    n_samples = X.shape[0]
    max_components = min(X.shape[1], Y.shape[1], n_samples - 1)

    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)

    # Standardize
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    Y_scaled = scaler_y.fit_transform(Y)

    try:
        pls = PLSCanonical(n_components=n_components, max_iter=1000)
        X_scores, Y_scores = pls.fit_transform(X_scaled, Y_scaled)

        # Correlations in latent space
        correlations = []
        for i in range(n_components):
            r, _ = stats.pearsonr(X_scores[:, i], Y_scores[:, i])
            correlations.append(r)

        return {
            'latent_correlations': correlations,
            'mean_correlation': np.mean(correlations),
            'first_correlation': correlations[0] if correlations else 0.0,
            'n_components': n_components
        }
    except Exception as e:
        return {
            'latent_correlations': [],
            'mean_correlation': 0.0,
            'first_correlation': 0.0,
            'n_components': 0,
            'error': str(e)
        }


def bootstrap_metric(X: np.ndarray, Y: np.ndarray, metric_fn: callable,
                     n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42) -> dict:
    """Bootstrap confidence intervals for any metric.

    Reproducibility: draws come from a local ``np.random.default_rng(seed)``
    (default seed 42), so results are deterministic regardless of global
    NumPy RNG state.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    bootstrap_values = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        X_boot = X[idx]
        Y_boot = Y[idx]

        try:
            val = metric_fn(X_boot, Y_boot)
            if isinstance(val, dict):
                val = val.get('mean_correlation', val.get('first_correlation', 0))
            bootstrap_values.append(val)
        except:
            continue

    if len(bootstrap_values) < 100:
        return {'ci_lower': np.nan, 'ci_upper': np.nan, 'se': np.nan}

    alpha = (1 - ci) / 2
    ci_lower = np.percentile(bootstrap_values, alpha * 100)
    ci_upper = np.percentile(bootstrap_values, (1 - alpha) * 100)
    se = np.std(bootstrap_values)

    return {
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se,
        'n_valid': len(bootstrap_values)
    }


def permutation_test(X: np.ndarray, Y: np.ndarray, metric_fn: callable,
                     observed: float, n_permutations: int = 1000, seed: int = 42) -> float:
    """Permutation test for significance of any metric.

    Reproducibility: permutations come from a local ``np.random.default_rng(seed)``
    (default seed 42), so the reported p-value is deterministic regardless of
    global NumPy RNG state.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    null_values = []

    for _ in range(n_permutations):
        perm_idx = rng.permutation(n)
        Y_perm = Y[perm_idx]

        try:
            val = metric_fn(X, Y_perm)
            if isinstance(val, dict):
                val = val.get('mean_correlation', val.get('first_correlation', 0))
            null_values.append(val)
        except:
            continue

    if len(null_values) < 100:
        return np.nan

    # Two-tailed p-value
    p_value = np.mean(np.abs(null_values) >= np.abs(observed))
    return p_value


def load_and_align_matrices(base_path: Path) -> tuple:
    """Load matrices and align to common assets."""
    # Load matrices
    claims = np.load(base_path / 'outputs/nlp/claims_matrix.npy')
    stats = np.load(base_path / 'outputs/market/stats_matrix.npy')
    factors = np.load(base_path / 'outputs/tensor/cp_asset_factors.npy')

    # Load metadata
    with open(base_path / 'outputs/nlp/claims_matrix_meta.json') as f:
        claims_meta = json.load(f)
    with open(base_path / 'outputs/market/stats_matrix_meta.json') as f:
        stats_meta = json.load(f)
    with open(base_path / 'outputs/tensor/cp_factors_meta.json') as f:
        factors_meta = json.load(f)

    claims_symbols = claims_meta['symbols']
    stats_symbols = stats_meta['symbols']
    factors_symbols = factors_meta['symbols']

    # Find common assets
    common = sorted(set(claims_symbols) & set(stats_symbols) & set(factors_symbols))

    # Align matrices
    claims_idx = [claims_symbols.index(s) for s in common]
    stats_idx = [stats_symbols.index(s) for s in common]
    factors_idx = [factors_symbols.index(s) for s in common]

    claims_aligned = claims[claims_idx]
    stats_aligned = stats[stats_idx]
    factors_aligned = factors[factors_idx]

    return claims_aligned, stats_aligned, factors_aligned, common


def compute_all_metrics(X: np.ndarray, Y: np.ndarray, name: str,
                        n_bootstrap: int = 1000, n_permutations: int = 1000) -> dict:
    """Compute all alternative metrics for a pair of matrices."""
    print(f"\n  Computing metrics for {name}...")

    results = {'comparison': name}

    # 1. RV coefficient
    print(f"    RV coefficient...", end=' ')
    rv = rv_coefficient(X, Y)
    rv_boot = bootstrap_metric(X, Y, rv_coefficient, n_bootstrap)
    rv_p = permutation_test(X, Y, rv_coefficient, rv, n_permutations)
    results['rv'] = {
        'value': rv,
        'ci_lower': rv_boot['ci_lower'],
        'ci_upper': rv_boot['ci_upper'],
        'se': rv_boot['se'],
        'p_value': rv_p
    }
    print(f"RV={rv:.4f}")

    # 2. Distance correlation
    print(f"    Distance correlation...", end=' ')
    dcor = distance_correlation(X, Y)
    dcor_boot = bootstrap_metric(X, Y, distance_correlation, n_bootstrap)
    dcor_p = permutation_test(X, Y, distance_correlation, dcor, n_permutations)
    results['dcor'] = {
        'value': dcor,
        'ci_lower': dcor_boot['ci_lower'],
        'ci_upper': dcor_boot['ci_upper'],
        'se': dcor_boot['se'],
        'p_value': dcor_p
    }
    print(f"dCor={dcor:.4f}")

    # 3. CCA
    print(f"    CCA...", end=' ')
    cca_result = cca_correlation(X, Y)
    cca_boot = bootstrap_metric(X, Y, lambda a, b: cca_correlation(a, b), n_bootstrap)
    cca_p = permutation_test(X, Y, lambda a, b: cca_correlation(a, b),
                             cca_result['mean_correlation'], n_permutations)
    results['cca'] = {
        'canonical_correlations': cca_result['canonical_correlations'],
        'mean_correlation': cca_result['mean_correlation'],
        'first_correlation': cca_result['first_correlation'],
        'ci_lower': cca_boot['ci_lower'],
        'ci_upper': cca_boot['ci_upper'],
        'se': cca_boot['se'],
        'p_value': cca_p,
        'n_components': cca_result['n_components']
    }
    print(f"CCA={cca_result['mean_correlation']:.4f}")

    # 4. PLS
    print(f"    PLS...", end=' ')
    pls_result = pls_score(X, Y)
    pls_boot = bootstrap_metric(X, Y, lambda a, b: pls_score(a, b), n_bootstrap)
    pls_p = permutation_test(X, Y, lambda a, b: pls_score(a, b),
                             pls_result['mean_correlation'], n_permutations)
    results['pls'] = {
        'latent_correlations': pls_result['latent_correlations'],
        'mean_correlation': pls_result['mean_correlation'],
        'first_correlation': pls_result['first_correlation'],
        'ci_lower': pls_boot['ci_lower'],
        'ci_upper': pls_boot['ci_upper'],
        'se': pls_boot['se'],
        'p_value': pls_p,
        'n_components': pls_result['n_components']
    }
    print(f"PLS={pls_result['mean_correlation']:.4f}")

    return results


def main():
    base_path = Path(__file__).parent.parent.parent
    output_path = base_path / 'outputs' / 'alignment'
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ALTERNATIVE ALIGNMENT METRICS")
    print("=" * 60)

    # Load and align
    print("\nLoading and aligning matrices...")
    claims, stats, factors, common_assets = load_and_align_matrices(base_path)

    print(f"  Common assets: {len(common_assets)}")
    print(f"  Claims shape: {claims.shape}")
    print(f"  Stats shape: {stats.shape}")
    print(f"  Factors shape: {factors.shape}")

    # Compute for all pairs
    all_results = {
        'n_assets': len(common_assets),
        'common_assets': common_assets,
        'matrix_shapes': {
            'claims': list(claims.shape),
            'stats': list(stats.shape),
            'factors': list(factors.shape)
        },
        'comparisons': []
    }

    # 1. Claims vs Factors
    print("\n" + "-" * 40)
    print("Claims vs Factors")
    print("-" * 40)
    cf_results = compute_all_metrics(claims, factors, 'claims_vs_factors')
    all_results['comparisons'].append(cf_results)

    # 2. Claims vs Statistics
    print("\n" + "-" * 40)
    print("Claims vs Statistics")
    print("-" * 40)
    cs_results = compute_all_metrics(claims, stats, 'claims_vs_stats')
    all_results['comparisons'].append(cs_results)

    # 3. Statistics vs Factors (sanity check - should be significant)
    print("\n" + "-" * 40)
    print("Statistics vs Factors (Sanity Check)")
    print("-" * 40)
    sf_results = compute_all_metrics(stats, factors, 'stats_vs_factors')
    all_results['comparisons'].append(sf_results)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "Comparison", "RV", "dCor", "CCA", "PLS"))
    print("-" * 65)

    for comp in all_results['comparisons']:
        name = comp['comparison'].replace('_', ' ').title()
        rv = comp['rv']['value']
        dcor = comp['dcor']['value']
        cca = comp['cca']['mean_correlation']
        pls = comp['pls']['mean_correlation']

        rv_sig = "*" if comp['rv']['p_value'] < 0.05 else ""
        dcor_sig = "*" if comp['dcor']['p_value'] < 0.05 else ""
        cca_sig = "*" if comp['cca']['p_value'] < 0.05 else ""
        pls_sig = "*" if comp['pls']['p_value'] < 0.05 else ""

        print("{:<25} {:>9.3f}{} {:>9.3f}{} {:>9.3f}{} {:>9.3f}{}".format(
            name, rv, rv_sig, dcor, dcor_sig, cca, cca_sig, pls, pls_sig))

    print("\n* p < 0.05")

    # Save results
    output_file = output_path / 'alternative_metrics.json'

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == '__main__':
    main()
