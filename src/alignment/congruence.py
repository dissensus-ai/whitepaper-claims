#!/usr/bin/env python3
"""
Tucker's Congruence Coefficient for TENSOR-DEFI

Implements Tucker's φ coefficient for measuring factor similarity.
Ranges from -1 to +1, where |φ| > 0.85 indicates factor equivalence.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from .procrustes import ProcrustesAlignment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CongruenceCoefficient:
    """Computes Tucker's congruence coefficient for matrix alignment."""

    # Interpretation thresholds (Lorenzo-Seva & ten Berge, 2006)
    THRESHOLDS = {
        'equivalent': 0.95,  # Factors considered equal
        'similar': 0.85,     # Fair similarity
        'moderate': 0.65,    # Some similarity
        'weak': 0.0          # Below this = dissimilar
    }

    def __init__(self):
        self.aligner = ProcrustesAlignment()

    def tuckers_phi(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Tucker's congruence coefficient between two vectors.

        φ = Σ(x_i * y_i) / √(Σx_i² * Σy_i²)

        This is essentially the cosine similarity but without mean-centering,
        making it more suitable for factor comparison.
        """
        numerator = np.sum(x * y)
        denominator = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def matrix_congruence(
        self,
        A: np.ndarray,
        B: np.ndarray,
        per_column: bool = True
    ) -> dict:
        """
        Compute congruence between two matrices after Procrustes alignment.

        Returns overall φ and per-dimension φ values.
        """
        # First align via Procrustes
        alignment = self.aligner.align_matrices(A, B, "A", "B")
        A_rotated = alignment['source_rotated']
        B_centered = alignment['target_centered']

        # Overall congruence (average across columns)
        n_cols = A_rotated.shape[1]
        column_phis = []

        for i in range(n_cols):
            phi = self.tuckers_phi(A_rotated[:, i], B_centered[:, i])
            column_phis.append(phi)

        # Mean absolute φ (factors can have opposite signs)
        mean_phi = np.mean(np.abs(column_phis))

        # Root mean square φ (alternative metric)
        rms_phi = np.sqrt(np.mean(np.array(column_phis) ** 2))

        return {
            'mean_phi': float(mean_phi),
            'rms_phi': float(rms_phi),
            'column_phis': [float(p) for p in column_phis],
            'interpretation': self._interpret_phi(mean_phi),
            'alignment': alignment
        }

    def _interpret_phi(self, phi: float) -> str:
        """Interpret congruence coefficient."""
        abs_phi = abs(phi)
        if abs_phi >= self.THRESHOLDS['equivalent']:
            return 'equivalent'
        elif abs_phi >= self.THRESHOLDS['similar']:
            return 'similar'
        elif abs_phi >= self.THRESHOLDS['moderate']:
            return 'moderate'
        else:
            return 'weak'

    def bootstrap_confidence_interval(
        self,
        A: np.ndarray,
        B: np.ndarray,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        seed: int = 42
    ) -> dict:
        """
        Bootstrap confidence interval for congruence coefficient.

        Reproducibility: draws come from a local ``np.random.default_rng(seed)``
        (default seed 42), so results are deterministic regardless of global
        NumPy RNG state.
        """
        rng = np.random.default_rng(seed)
        n_entities = A.shape[0]
        bootstrap_phis = []

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = rng.choice(n_entities, size=n_entities, replace=True)
            A_boot = A[indices]
            B_boot = B[indices]

            result = self.matrix_congruence(A_boot, B_boot)
            bootstrap_phis.append(result['mean_phi'])

        bootstrap_phis = np.array(bootstrap_phis)

        # Confidence interval
        alpha = 1 - ci_level
        ci_lower = np.percentile(bootstrap_phis, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_phis, 100 * (1 - alpha / 2))

        return {
            'mean': float(np.mean(bootstrap_phis)),
            'std': float(np.std(bootstrap_phis)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'ci_level': ci_level,
            'n_bootstrap': n_bootstrap
        }

    def permutation_test(
        self,
        A: np.ndarray,
        B: np.ndarray,
        n_permutations: int = 1000,
        seed: int = 42
    ) -> dict:
        """
        Permutation test: is φ significantly greater than chance?

        Reproducibility: permutations come from a local
        ``np.random.default_rng(seed)`` (default seed 42), so the reported
        p-value is deterministic regardless of global NumPy RNG state.
        """
        rng = np.random.default_rng(seed)

        # Observed φ
        observed = self.matrix_congruence(A, B)['mean_phi']

        # Null distribution (permute rows of B)
        null_phis = []
        for _ in range(n_permutations):
            perm_indices = rng.permutation(B.shape[0])
            B_perm = B[perm_indices]
            null_phi = self.matrix_congruence(A, B_perm)['mean_phi']
            null_phis.append(null_phi)

        null_phis = np.array(null_phis)

        # One-tailed p-value (observed > null)
        p_value = np.mean(null_phis >= observed)

        return {
            'observed_phi': float(observed),
            'null_mean': float(np.mean(null_phis)),
            'null_std': float(np.std(null_phis)),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'effect_size': float((observed - np.mean(null_phis)) / np.std(null_phis))
        }


class AlignmentTester:
    """
    Runs full alignment testing for TENSOR-DEFI.

    Tests three alignments:
    1. Claims ↔ Stats (φ₁)
    2. Claims ↔ Factors (φ₂)
    3. Stats ↔ Factors (φ₃)
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.congruence = CongruenceCoefficient()

    def load_matrices(
        self,
        claims_path: Path,
        stats_path: Path,
        factors_path: Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Load all three matrices and verify entity alignment."""
        claims = np.load(claims_path)
        stats_matrix = np.load(stats_path)
        factors = np.load(factors_path)

        # Load metadata for symbols
        claims_meta_path = claims_path.parent / "claims_matrix_meta.json"
        stats_meta_path = stats_path.parent / "stats_matrix_meta.json"
        factors_meta_path = factors_path.parent / "cp_factors_meta.json"

        with open(claims_meta_path) as f:
            claims_symbols = json.load(f)['symbols']
        with open(stats_meta_path) as f:
            stats_symbols = json.load(f)['symbols']
        with open(factors_meta_path) as f:
            factors_symbols = json.load(f)['symbols']

        # Find common symbols
        common = set(claims_symbols) & set(stats_symbols) & set(factors_symbols)
        common = sorted(common)

        logger.info(f"Common entities: {len(common)}")

        # Filter to common symbols
        claims_idx = [claims_symbols.index(s) for s in common]
        stats_idx = [stats_symbols.index(s) for s in common]
        factors_idx = [factors_symbols.index(s) for s in common]

        claims = claims[claims_idx]
        stats_matrix = stats_matrix[stats_idx]
        factors = factors[factors_idx]

        return claims, stats_matrix, factors, common

    def run_alignment_tests(
        self,
        claims: np.ndarray,
        stats_matrix: np.ndarray,
        factors: np.ndarray,
        n_bootstrap: int = 1000
    ) -> dict:
        """Run all three alignment tests."""
        results = {}

        # Test 1: Claims ↔ Stats
        logger.info("Testing Claims ↔ Stats alignment...")
        results['claims_stats'] = {
            'congruence': self.congruence.matrix_congruence(claims, stats_matrix),
            'bootstrap': self.congruence.bootstrap_confidence_interval(
                claims, stats_matrix, n_bootstrap
            ),
            'permutation': self.congruence.permutation_test(claims, stats_matrix)
        }

        # Test 2: Claims ↔ Factors
        logger.info("Testing Claims ↔ Factors alignment...")
        results['claims_factors'] = {
            'congruence': self.congruence.matrix_congruence(claims, factors),
            'bootstrap': self.congruence.bootstrap_confidence_interval(
                claims, factors, n_bootstrap
            ),
            'permutation': self.congruence.permutation_test(claims, factors)
        }

        # Test 3: Stats ↔ Factors
        logger.info("Testing Stats ↔ Factors alignment...")
        results['stats_factors'] = {
            'congruence': self.congruence.matrix_congruence(stats_matrix, factors),
            'bootstrap': self.congruence.bootstrap_confidence_interval(
                stats_matrix, factors, n_bootstrap
            ),
            'permutation': self.congruence.permutation_test(stats_matrix, factors)
        }

        return results

    def save_results(self, results: dict, symbols: list[str]):
        """Save alignment results."""
        # Clean results for JSON (remove numpy arrays, convert numpy types)
        clean_results = {}
        for key, value in results.items():
            clean_results[key] = {
                'mean_phi': float(value['congruence']['mean_phi']),
                'interpretation': value['congruence']['interpretation'],
                'column_phis': [float(x) for x in value['congruence']['column_phis']],
                'bootstrap_ci': [
                    float(value['bootstrap']['ci_lower']),
                    float(value['bootstrap']['ci_upper'])
                ],
                'p_value': float(value['permutation']['p_value']),
                'significant': bool(value['permutation']['significant'])
            }

        clean_results['metadata'] = {
            'n_entities': len(symbols),
            'symbols': symbols
        }

        output_path = self.output_dir / "alignment_results.json"
        with open(output_path, 'w') as f:
            json.dump(clean_results, f, indent=2)

        logger.info(f"Saved results: {output_path}")

        self._print_summary(results)

    def _print_summary(self, results: dict):
        """Print alignment summary."""
        print(f"\n{'='*70}")
        print("ALIGNMENT TESTING RESULTS")
        print(f"{'='*70}")

        comparisons = [
            ('Claims ↔ Stats', 'claims_stats'),
            ('Claims ↔ Factors', 'claims_factors'),
            ('Stats ↔ Factors', 'stats_factors')
        ]

        for name, key in comparisons:
            r = results[key]
            phi = r['congruence']['mean_phi']
            ci_lo = r['bootstrap']['ci_lower']
            ci_hi = r['bootstrap']['ci_upper']
            p = r['permutation']['p_value']
            interp = r['congruence']['interpretation']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

            print(f"\n{name}:")
            print(f"  Tucker's φ = {phi:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] {sig}")
            print(f"  Interpretation: {interp}")
            print(f"  p-value: {p:.4f}")

        # Which alignment is strongest?
        phis = {
            'Claims-Stats': results['claims_stats']['congruence']['mean_phi'],
            'Claims-Factors': results['claims_factors']['congruence']['mean_phi'],
            'Stats-Factors': results['stats_factors']['congruence']['mean_phi']
        }
        best = max(phis, key=phis.get)

        print(f"\n{'='*70}")
        print(f"CONCLUSION: Strongest alignment is {best} (φ = {phis[best]:.3f})")
        print(f"{'='*70}")


def main():
    """Run alignment testing."""
    base_path = Path(__file__).parent.parent.parent

    tester = AlignmentTester(output_dir=base_path / "outputs" / "alignment")

    claims, stats_matrix, factors, symbols = tester.load_matrices(
        claims_path=base_path / "outputs" / "nlp" / "claims_matrix.npy",
        stats_path=base_path / "outputs" / "market" / "stats_matrix.npy",
        factors_path=base_path / "outputs" / "tensor" / "cp_asset_factors.npy"
    )

    results = tester.run_alignment_tests(claims, stats_matrix, factors)
    tester.save_results(results, symbols)


if __name__ == "__main__":
    main()
