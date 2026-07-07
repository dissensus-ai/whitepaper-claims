"""Robustness checks for TENSOR-DEFI."""

import logging
from pathlib import Path
import numpy as np
import json

logger = logging.getLogger(__name__)


class RobustnessChecker:
    """Robustness and sensitivity analysis."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def subsample_stability(self, claims_matrix: np.ndarray, stats_matrix: np.ndarray, symbols: list[str],
                           n_iterations: int = 100, subsample_frac: float = 0.8, seed: int = 42) -> dict:
        """Test alignment stability with random subsamples (deterministic; local default_rng(seed))."""
        import sys
        sys.path.insert(0, str(self.output_dir.parent.parent / "src"))
        from alignment.procrustes import ProcrustesAlignment
        from alignment.congruence import CongruenceCoefficient

        rng = np.random.default_rng(seed)
        aligner = ProcrustesAlignment()
        congruence = CongruenceCoefficient()
        n = len(symbols)
        subsample_size = int(n * subsample_frac)

        phis = []
        for _ in range(n_iterations):
            indices = rng.choice(n, size=subsample_size, replace=False)
            try:
                result = aligner.align_matrices(claims_matrix[indices], stats_matrix[indices])
                cong_result = congruence.matrix_congruence(result['source_rotated'], result['target_centered'])
                phis.append(float(cong_result['mean_phi']))
            except:
                continue

        return {
            'n_iterations': n_iterations,
            'subsample_frac': subsample_frac,
            'mean_phi': float(np.mean(phis)) if phis else 0,
            'std_phi': float(np.std(phis)) if phis else 0,
            'ci_lower': float(np.percentile(phis, 2.5)) if phis else 0,
            'ci_upper': float(np.percentile(phis, 97.5)) if phis else 0,
        }

    def feature_importance(self, claims_matrix: np.ndarray, stats_matrix: np.ndarray,
                          claim_categories: list[str], stat_features: list[str]) -> dict:
        """Analyze which features drive alignment."""
        import sys
        sys.path.insert(0, str(self.output_dir.parent.parent / "src"))
        from alignment.procrustes import ProcrustesAlignment
        from alignment.congruence import CongruenceCoefficient

        aligner = ProcrustesAlignment()
        congruence = CongruenceCoefficient()

        result_full = aligner.align_matrices(claims_matrix, stats_matrix)
        phi_full = congruence.matrix_congruence(result_full['source_rotated'], result_full['target_centered'])['mean_phi']

        claims_importance = []
        for i, cat in enumerate(claim_categories):
            claims_ablated = claims_matrix.copy()
            claims_ablated[:, i] = 0
            result = aligner.align_matrices(claims_ablated, stats_matrix)
            phi_ablated = congruence.matrix_congruence(result['source_rotated'], result['target_centered'])['mean_phi']
            claims_importance.append({
                'feature': cat,
                'phi_without': float(phi_ablated),
                'importance': float(phi_full - phi_ablated)
            })

        claims_importance.sort(key=lambda x: x['importance'], reverse=True)

        return {
            'phi_full': float(phi_full),
            'claims_importance': claims_importance,
            'most_important': [c['feature'] for c in claims_importance[:3]],
        }

    def save_results(self, results: dict):
        output_path = self.output_dir / "robustness_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved: {output_path}")

        print(f"\n{'='*60}")
        print("ROBUSTNESS ANALYSIS RESULTS")
        print(f"{'='*60}")

        if 'subsample_stability' in results:
            ss = results['subsample_stability']
            print(f"\nSubsample stability ({ss['n_iterations']} iters, {ss['subsample_frac']:.0%}):")
            print(f"  Mean φ: {ss['mean_phi']:.3f} ± {ss['std_phi']:.3f}")
            print(f"  95% CI: [{ss['ci_lower']:.3f}, {ss['ci_upper']:.3f}]")

        if 'feature_importance' in results:
            print("\nClaim category importance:")
            for c in results['feature_importance']['claims_importance']:
                print(f"  {c['feature']:20s}: {c['importance']:+.3f}")
