#!/usr/bin/env python3
"""
Alternative Alignment Methods for TENSOR-DEFI Expansion

Implements:
1. Canonical Correlation Analysis (CCA) - handles dimension mismatch naturally
2. RV Coefficient - rotation-invariant similarity between configuration matrices

These methods avoid the zero-padding bias of Procrustes when matrices have
different column dimensions.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CCAAlignment:
    """
    Canonical Correlation Analysis for cross-modal alignment.

    CCA finds linear combinations of variables in each matrix that are
    maximally correlated. Unlike Procrustes, it handles dimension mismatch
    naturally without zero-padding.

    Interpretation:
    - ρ₁, ρ₂: First two canonical correlations (0-1 scale)
    - ρ₁ > 0.7: Strong alignment of primary dimension
    - Mean canonical correlation gives overall alignment strength
    """

    def __init__(self, n_components: int = None, scale: bool = True):
        """
        Args:
            n_components: Number of canonical components (default: min dimensions)
            scale: Whether to standardize inputs
        """
        self.n_components = n_components
        self.scale = scale

    def fit_transform(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        source_name: str = "X",
        target_name: str = "Y"
    ) -> dict:
        """
        Compute CCA between two matrices.

        Args:
            X: First matrix (N × D1)
            Y: Second matrix (N × D2)

        Returns:
            Dictionary with canonical correlations and projections
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Row mismatch: {X.shape[0]} vs {Y.shape[0]}")

        # Determine components
        n_components = self.n_components or min(X.shape[1], Y.shape[1])
        n_components = min(n_components, X.shape[0] - 1)  # Can't exceed n-1

        logger.info(
            f"CCA: {source_name} {X.shape} ↔ {target_name} {Y.shape}, "
            f"n_components={n_components}"
        )

        # Standardize if requested
        if self.scale:
            X_scaled = StandardScaler().fit_transform(X)
            Y_scaled = StandardScaler().fit_transform(Y)
        else:
            X_scaled = X.copy()
            Y_scaled = Y.copy()

        # Fit CCA
        cca = CCA(n_components=n_components)
        X_c, Y_c = cca.fit_transform(X_scaled, Y_scaled)

        # Compute canonical correlations for each component
        canonical_correlations = []
        for i in range(n_components):
            rho, _ = stats.pearsonr(X_c[:, i], Y_c[:, i])
            canonical_correlations.append(rho)

        # Wilks' Lambda for significance testing
        wilks_lambda = self._compute_wilks_lambda(canonical_correlations)

        # Mean and RMS canonical correlation
        mean_rho = np.mean(canonical_correlations)
        rms_rho = np.sqrt(np.mean(np.array(canonical_correlations) ** 2))

        return {
            'source_name': source_name,
            'target_name': target_name,
            'source_shape': list(X.shape),
            'target_shape': list(Y.shape),
            'n_components': n_components,
            'canonical_correlations': [float(r) for r in canonical_correlations],
            'rho_1': float(canonical_correlations[0]) if canonical_correlations else 0.0,
            'rho_2': float(canonical_correlations[1]) if len(canonical_correlations) > 1 else 0.0,
            'mean_rho': float(mean_rho),
            'rms_rho': float(rms_rho),
            'wilks_lambda': float(wilks_lambda),
            'X_canonical': X_c,
            'Y_canonical': Y_c,
            'x_loadings': cca.x_loadings_,
            'y_loadings': cca.y_loadings_,
            'interpretation': self._interpret_rho(mean_rho)
        }

    def _compute_wilks_lambda(self, correlations: list) -> float:
        """
        Compute Wilks' Lambda statistic.

        Lambda = product(1 - rho_i^2)
        Smaller Lambda = stronger relationship
        """
        wilks = 1.0
        for rho in correlations:
            wilks *= (1 - rho ** 2)
        return wilks

    def _interpret_rho(self, rho: float) -> str:
        """Interpret canonical correlation strength."""
        if rho >= 0.8:
            return 'very_strong'
        elif rho >= 0.6:
            return 'strong'
        elif rho >= 0.4:
            return 'moderate'
        elif rho >= 0.2:
            return 'weak'
        else:
            return 'negligible'

    def bootstrap_ci(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        seed: int = 42
    ) -> dict:
        """Bootstrap confidence interval for first canonical correlation.

        Reproducibility: draws come from a local ``np.random.default_rng(seed)``
        (default seed 42), deterministic regardless of global NumPy state.
        """
        rng = np.random.default_rng(seed)
        n_samples = X.shape[0]
        bootstrap_rhos = []

        for _ in range(n_bootstrap):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            Y_boot = Y[indices]

            try:
                result = self.fit_transform(X_boot, Y_boot)
                bootstrap_rhos.append(result['rho_1'])
            except Exception:
                continue  # Skip failed fits

        if not bootstrap_rhos:
            return {'ci_lower': 0.0, 'ci_upper': 0.0, 'mean': 0.0}

        bootstrap_rhos = np.array(bootstrap_rhos)
        alpha = 1 - ci_level

        return {
            'mean': float(np.mean(bootstrap_rhos)),
            'std': float(np.std(bootstrap_rhos)),
            'ci_lower': float(np.percentile(bootstrap_rhos, 100 * alpha / 2)),
            'ci_upper': float(np.percentile(bootstrap_rhos, 100 * (1 - alpha / 2))),
            'ci_level': ci_level,
            'n_bootstrap': n_bootstrap
        }


class RVCoefficient:
    """
    RV Coefficient for comparing configuration matrices.

    The RV coefficient is a multivariate generalization of the squared
    Pearson correlation. It measures similarity between two sets of
    centered/scaled variables without requiring alignment.

    Key properties:
    - Rotation-invariant (doesn't need Procrustes)
    - Handles different dimensions naturally
    - Ranges from 0 to 1

    Interpretation:
    - RV > 0.9: Nearly identical configurations
    - RV > 0.7: Strong similarity
    - RV > 0.5: Moderate similarity
    - RV < 0.3: Weak similarity
    """

    def __init__(self, center: bool = True, scale: bool = True):
        self.center = center
        self.scale = scale

    def compute(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        source_name: str = "X",
        target_name: str = "Y"
    ) -> dict:
        """
        Compute RV coefficient between two matrices.

        RV = trace(XX'YY') / sqrt(trace(XX'XX') * trace(YY'YY'))

        This is the inner product between configuration matrices, normalized.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Row mismatch: {X.shape[0]} vs {Y.shape[0]}")

        logger.info(f"RV: {source_name} {X.shape} ↔ {target_name} {Y.shape}")

        # Preprocess
        if self.center:
            X = X - X.mean(axis=0)
            Y = Y - Y.mean(axis=0)

        if self.scale:
            X_norm = np.linalg.norm(X, 'fro')
            Y_norm = np.linalg.norm(Y, 'fro')
            if X_norm > 0:
                X = X / X_norm
            if Y_norm > 0:
                Y = Y / Y_norm

        # Compute cross-product matrices
        XX = X @ X.T  # N × N configuration matrix
        YY = Y @ Y.T  # N × N configuration matrix

        # RV coefficient
        numerator = np.trace(XX @ YY)
        denominator = np.sqrt(np.trace(XX @ XX) * np.trace(YY @ YY))

        rv = numerator / denominator if denominator > 0 else 0.0

        return {
            'source_name': source_name,
            'target_name': target_name,
            'source_shape': list(X.shape),
            'target_shape': list(Y.shape),
            'rv_coefficient': float(rv),
            'numerator': float(numerator),
            'denominator': float(denominator),
            'interpretation': self._interpret_rv(rv)
        }

    def _interpret_rv(self, rv: float) -> str:
        """Interpret RV coefficient."""
        if rv >= 0.9:
            return 'nearly_identical'
        elif rv >= 0.7:
            return 'strong'
        elif rv >= 0.5:
            return 'moderate'
        elif rv >= 0.3:
            return 'weak'
        else:
            return 'negligible'

    def permutation_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_permutations: int = 1000,
        seed: int = 42
    ) -> dict:
        """Permutation test for RV significance.

        Reproducibility: permutations come from a local ``np.random.default_rng(seed)``
        (default seed 42), deterministic regardless of global NumPy state.
        """
        rng = np.random.default_rng(seed)
        observed = self.compute(X, Y)['rv_coefficient']

        null_rvs = []
        for _ in range(n_permutations):
            perm_idx = rng.permutation(Y.shape[0])
            Y_perm = Y[perm_idx]
            null_rv = self.compute(X, Y_perm)['rv_coefficient']
            null_rvs.append(null_rv)

        null_rvs = np.array(null_rvs)
        p_value = np.mean(null_rvs >= observed)

        return {
            'observed_rv': float(observed),
            'null_mean': float(np.mean(null_rvs)),
            'null_std': float(np.std(null_rvs)),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'effect_size': float((observed - np.mean(null_rvs)) / np.std(null_rvs)) if np.std(null_rvs) > 0 else 0.0
        }


class ExtendedAlignmentTester:
    """
    Extended alignment testing with multiple methods.

    Compares:
    1. Procrustes (original)
    2. CCA
    3. RV Coefficient
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cca = CCAAlignment()
        self.rv = RVCoefficient()

        # Import Procrustes from existing module
        from .congruence import CongruenceCoefficient
        self.procrustes = CongruenceCoefficient()

    def run_all_methods(
        self,
        claims: np.ndarray,
        factors: np.ndarray,
        source_name: str = "claims",
        target_name: str = "factors"
    ) -> dict:
        """Run all alignment methods on a pair of matrices."""
        results = {}

        # 1. Procrustes (original)
        logger.info("Running Procrustes alignment...")
        procrustes_result = self.procrustes.matrix_congruence(claims, factors)
        results['procrustes'] = {
            'phi': procrustes_result['mean_phi'],
            'interpretation': procrustes_result['interpretation']
        }

        # 2. CCA
        logger.info("Running CCA...")
        cca_result = self.cca.fit_transform(claims, factors, source_name, target_name)
        results['cca'] = {
            'rho_1': cca_result['rho_1'],
            'rho_2': cca_result['rho_2'],
            'mean_rho': cca_result['mean_rho'],
            'wilks_lambda': cca_result['wilks_lambda'],
            'interpretation': cca_result['interpretation']
        }

        # 3. RV Coefficient
        logger.info("Running RV coefficient...")
        rv_result = self.rv.compute(claims, factors, source_name, target_name)
        results['rv'] = {
            'rv_coefficient': rv_result['rv_coefficient'],
            'interpretation': rv_result['interpretation']
        }

        return results

    def run_comparison(
        self,
        claims: np.ndarray,
        stats_matrix: np.ndarray,
        factors: np.ndarray,
        n_bootstrap: int = 500
    ) -> dict:
        """
        Run full comparison across all three matrix pairs.

        Returns comprehensive comparison of alignment methods.
        """
        pairs = [
            ('claims', 'stats', claims, stats_matrix),
            ('claims', 'factors', claims, factors),
            ('stats', 'factors', stats_matrix, factors)
        ]

        results = {}

        for source_name, target_name, source, target in pairs:
            pair_key = f"{source_name}_{target_name}"
            logger.info(f"\n{'='*50}")
            logger.info(f"Comparing {source_name} ↔ {target_name}")
            logger.info(f"{'='*50}")

            results[pair_key] = self.run_all_methods(
                source, target, source_name, target_name
            )

            # Add bootstrap CI for CCA
            logger.info("Computing CCA bootstrap CI...")
            cca_bootstrap = self.cca.bootstrap_ci(source, target, n_bootstrap)
            results[pair_key]['cca']['bootstrap'] = cca_bootstrap

            # Add permutation test for RV
            logger.info("Computing RV permutation test...")
            rv_perm = self.rv.permutation_test(source, target, n_permutations=500)
            results[pair_key]['rv']['permutation'] = rv_perm

        return results

    def save_comparison(self, results: dict):
        """Save comparison results."""
        # Clean for JSON
        clean_results = {}
        for pair_key, pair_results in results.items():
            clean_results[pair_key] = {}
            for method, method_results in pair_results.items():
                clean_results[pair_key][method] = {
                    k: v for k, v in method_results.items()
                    if not isinstance(v, np.ndarray)
                }

        output_path = self.output_dir / "alignment_comparison.json"
        with open(output_path, 'w') as f:
            json.dump(clean_results, f, indent=2)

        logger.info(f"Saved comparison: {output_path}")
        self._print_comparison(results)

    def _print_comparison(self, results: dict):
        """Print comparison table."""
        print(f"\n{'='*80}")
        print("ALIGNMENT METHOD COMPARISON")
        print(f"{'='*80}")
        print(f"{'Pair':<20} {'Procrustes φ':>15} {'CCA ρ₁':>12} {'RV coef':>12}")
        print(f"{'-'*80}")

        for pair_key, pair_results in results.items():
            phi = pair_results['procrustes']['phi']
            rho = pair_results['cca']['rho_1']
            rv = pair_results['rv']['rv_coefficient']

            pair_name = pair_key.replace('_', ' ↔ ')
            print(f"{pair_name:<20} {phi:>15.3f} {rho:>12.3f} {rv:>12.3f}")

        print(f"{'='*80}")

        # Interpretation
        print("\nInterpretation Guide:")
        print("  Procrustes φ: |φ|>0.85 = similar, |φ|>0.95 = equivalent")
        print("  CCA ρ₁: >0.7 = strong alignment of primary dimension")
        print("  RV coef: >0.7 = strong similarity, >0.9 = nearly identical")
        print(f"{'='*80}")


def main():
    """Run alignment comparison."""
    base_path = Path(__file__).parent.parent.parent

    # Load matrices
    claims = np.load(base_path / "outputs" / "nlp" / "claims_matrix.npy")
    stats_matrix = np.load(base_path / "outputs" / "market" / "stats_matrix.npy")
    factors = np.load(base_path / "outputs" / "tensor" / "cp_asset_factors.npy")

    # Load metadata to align by symbol
    import json
    with open(base_path / "outputs" / "nlp" / "claims_matrix_meta.json") as f:
        claims_symbols = json.load(f)['symbols']
    with open(base_path / "outputs" / "market" / "stats_matrix_meta.json") as f:
        stats_symbols = json.load(f)['symbols']
    with open(base_path / "outputs" / "tensor" / "cp_factors_meta.json") as f:
        factors_symbols = json.load(f)['symbols']

    # Find common symbols
    common = sorted(set(claims_symbols) & set(stats_symbols) & set(factors_symbols))
    logger.info(f"Common entities: {len(common)}")

    # Align matrices
    claims_idx = [claims_symbols.index(s) for s in common]
    stats_idx = [stats_symbols.index(s) for s in common]
    factors_idx = [factors_symbols.index(s) for s in common]

    claims = claims[claims_idx]
    stats_matrix = stats_matrix[stats_idx]
    factors = factors[factors_idx]

    # Run comparison
    tester = ExtendedAlignmentTester(output_dir=base_path / "outputs" / "alignment")
    results = tester.run_comparison(claims, stats_matrix, factors)
    tester.save_comparison(results)


if __name__ == "__main__":
    main()
