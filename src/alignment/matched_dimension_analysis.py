#!/usr/bin/env python3
"""
Matched-dimension alignment analysis.

Reduces higher-dimensional matrices to lower dimensions via SVD/PCA
before computing Procrustes alignment, avoiding zero-padding bias.
Addresses reviewer question about non-padded significance testing.
"""

import numpy as np
import json
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def tucker_phi(A: np.ndarray, B: np.ndarray) -> float:
    """Tucker's congruence coefficient after Procrustes rotation."""
    # Center matrices
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)

    # Procrustes rotation
    R, _ = orthogonal_procrustes(A, B)
    A_rot = A @ R

    # Tucker's phi for each dimension
    phis = []
    for j in range(A_rot.shape[1]):
        num = np.dot(A_rot[:, j], B[:, j])
        denom = np.sqrt(np.dot(A_rot[:, j], A_rot[:, j]) * np.dot(B[:, j], B[:, j]))
        if denom > 0:
            phis.append(num / denom)

    return np.mean(phis) if phis else 0.0


def reduce_dimensions(X: np.ndarray, target_dim: int) -> np.ndarray:
    """Reduce matrix to target dimensions using PCA/SVD."""
    if X.shape[1] <= target_dim:
        return X

    pca = PCA(n_components=target_dim)
    return pca.fit_transform(X)


def permutation_test(X: np.ndarray, Y: np.ndarray, observed: float,
                     n_permutations: int = 1000, seed: int = 42) -> float:
    """Permutation test for significance (deterministic; local default_rng(seed))."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    null_values = []

    for _ in range(n_permutations):
        perm_idx = rng.permutation(n)
        Y_perm = Y[perm_idx]
        phi = tucker_phi(X, Y_perm)
        null_values.append(phi)

    p_value = np.mean(np.abs(null_values) >= np.abs(observed))
    return p_value


def bootstrap_ci(X: np.ndarray, Y: np.ndarray, n_bootstrap: int = 1000,
                 ci: float = 0.95, seed: int = 42) -> dict:
    """Bootstrap confidence intervals (deterministic; local default_rng(seed))."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    bootstrap_values = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        X_boot = X[idx]
        Y_boot = Y[idx]
        phi = tucker_phi(X_boot, Y_boot)
        bootstrap_values.append(phi)

    alpha = (1 - ci) / 2
    ci_lower = np.percentile(bootstrap_values, alpha * 100)
    ci_upper = np.percentile(bootstrap_values, (1 - alpha) * 100)

    return {
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': np.std(bootstrap_values)
    }


def load_and_align_matrices(base_path: Path) -> tuple:
    """Load matrices and align to common assets."""
    claims = np.load(base_path / 'outputs/nlp/claims_matrix.npy')
    stats = np.load(base_path / 'outputs/market/stats_matrix.npy')
    factors = np.load(base_path / 'outputs/tensor/cp_asset_factors.npy')

    with open(base_path / 'outputs/nlp/claims_matrix_meta.json') as f:
        claims_meta = json.load(f)
    with open(base_path / 'outputs/market/stats_matrix_meta.json') as f:
        stats_meta = json.load(f)
    with open(base_path / 'outputs/tensor/cp_factors_meta.json') as f:
        factors_meta = json.load(f)

    claims_symbols = claims_meta['symbols']
    stats_symbols = stats_meta['symbols']
    factors_symbols = factors_meta['symbols']

    common = sorted(set(claims_symbols) & set(stats_symbols) & set(factors_symbols))

    claims_idx = [claims_symbols.index(s) for s in common]
    stats_idx = [stats_symbols.index(s) for s in common]
    factors_idx = [factors_symbols.index(s) for s in common]

    return (claims[claims_idx], stats[stats_idx], factors[factors_idx], common)


def main():
    base_path = Path(__file__).parent.parent.parent
    output_path = base_path / 'outputs' / 'alignment'
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MATCHED-DIMENSION ALIGNMENT ANALYSIS")
    print("=" * 60)

    # Load data
    print("\nLoading matrices...")
    claims, stats, factors, common = load_and_align_matrices(base_path)

    print(f"  Assets: {len(common)}")
    print(f"  Claims: {claims.shape} (10D)")
    print(f"  Stats: {stats.shape} (7D)")
    print(f"  Factors: {factors.shape} (2D)")

    results = {
        'n_assets': len(common),
        'original_dims': {
            'claims': claims.shape[1],
            'stats': stats.shape[1],
            'factors': factors.shape[1]
        },
        'comparisons': []
    }

    # Define comparisons with different dimension matching strategies
    comparisons = [
        # Claims vs Factors
        {
            'name': 'claims_vs_factors',
            'X': claims,
            'Y': factors,
            'target_dim': 2,
            'reduce': 'X'  # Reduce claims to match factors
        },
        # Claims vs Statistics
        {
            'name': 'claims_vs_stats',
            'X': claims,
            'Y': stats,
            'target_dim': 7,
            'reduce': 'X'  # Reduce claims to match stats
        },
        # Statistics vs Factors
        {
            'name': 'stats_vs_factors',
            'X': stats,
            'Y': factors,
            'target_dim': 2,
            'reduce': 'X'  # Reduce stats to match factors
        }
    ]

    print("\n" + "-" * 60)
    print("MATCHED-DIMENSION RESULTS (via SVD reduction)")
    print("-" * 60)

    for comp in comparisons:
        print(f"\n{comp['name'].replace('_', ' ').title()}:")

        X = comp['X']
        Y = comp['Y']
        target = comp['target_dim']

        # Reduce to matched dimensions
        if comp['reduce'] == 'X':
            X_matched = reduce_dimensions(X, target)
            Y_matched = Y
        else:
            X_matched = X
            Y_matched = reduce_dimensions(Y, target)

        print(f"  Reduced: {X.shape[1]}D → {X_matched.shape[1]}D (matched to {target}D)")

        # Compute phi
        phi = tucker_phi(X_matched, Y_matched)
        print(f"  Tucker φ = {phi:.4f}")

        # Bootstrap CI
        boot = bootstrap_ci(X_matched, Y_matched, n_bootstrap=1000)
        print(f"  95% CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

        # Permutation test
        p_val = permutation_test(X_matched, Y_matched, phi, n_permutations=1000)
        sig = "*" if p_val < 0.05 else ""
        print(f"  p-value = {p_val:.4f}{sig}")

        results['comparisons'].append({
            'name': comp['name'],
            'original_dims': [X.shape[1], Y.shape[1]],
            'matched_dim': target,
            'phi': float(phi),
            'ci_lower': float(boot['ci_lower']),
            'ci_upper': float(boot['ci_upper']),
            'se': float(boot['se']),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05)
        })

    # Summary comparison with padded results
    print("\n" + "=" * 60)
    print("COMPARISON: PADDED vs MATCHED DIMENSIONS")
    print("=" * 60)
    print("\n{:<25} {:>15} {:>15}".format("Comparison", "Padded φ", "Matched φ"))
    print("-" * 55)

    # Load original padded results for comparison
    try:
        with open(base_path / 'outputs/alignment/alignment_results.json') as f:
            padded = json.load(f)

        padded_map = {
            'claims_vs_factors': padded.get('claims_factors', {}).get('phi', 'N/A'),
            'claims_vs_stats': padded.get('claims_stats', {}).get('phi', 'N/A'),
            'stats_vs_factors': padded.get('stats_factors', {}).get('phi', 'N/A')
        }

        for comp in results['comparisons']:
            name = comp['name'].replace('_', ' ').title()
            padded_phi = padded_map.get(comp['name'], 'N/A')
            matched_phi = comp['phi']
            if isinstance(padded_phi, float):
                print(f"{name:<25} {padded_phi:>15.4f} {matched_phi:>15.4f}")
            else:
                print(f"{name:<25} {'N/A':>15} {matched_phi:>15.4f}")
    except FileNotFoundError:
        print("(Original padded results not found)")

    # Save results
    with open(output_path / 'matched_dimension_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path / 'matched_dimension_results.json'}")

    return results


if __name__ == '__main__':
    main()
