"""
FESTA Uncertainty Computation
Implements Algorithm 1 from the FESTA paper

U_FESTA = U_FES + U_FCS

Where:
- U_FES = -log(q_FES(y=ŷ|X))  # Consistency uncertainty
- U_FCS = -log(Σ_{y≠ŷ} q_FCS(y|X))  # Sensitivity uncertainty
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FESTAUncertainty:
    """FESTA uncertainty estimator"""

    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize FESTA uncertainty estimator

        Args:
            epsilon: Small value to avoid log(0)
        """
        self.epsilon = epsilon

    def compute_festa(
        self,
        fes_predictions: List[str],
        fcs_predictions: List[str],
        original_prediction: str
    ) -> Dict[str, float]:
        """
        Compute FESTA uncertainty score

        Args:
            fes_predictions: List of predictions on FES samples
            fcs_predictions: List of predictions on FCS samples
            original_prediction: Original model prediction

        Returns:
            Dictionary with U_FES, U_FCS, and U_FESTA scores
        """
        # Compute FES uncertainty
        u_fes = self.compute_u_fes(fes_predictions, original_prediction)

        # Compute FCS uncertainty
        u_fcs = self.compute_u_fcs(fcs_predictions, original_prediction)

        # Compute FESTA score
        u_festa = u_fes + u_fcs

        return {
            'U_FES': u_fes,
            'U_FCS': u_fcs,
            'U_FESTA': u_festa
        }

    def compute_u_fes(
        self,
        fes_predictions: List[str],
        original_prediction: str
    ) -> float:
        """
        Compute U_FES: consistency uncertainty from equivalent samples

        U_FES = -log(q_FES(y=ŷ|X))

        where q_FES(y|X) is the empirical distribution over FES samples

        Args:
            fes_predictions: Predictions on FES samples
            original_prediction: Original prediction ŷ

        Returns:
            U_FES score (higher = more uncertain)
        """
        if not fes_predictions:
            return 0.0

        # Compute empirical distribution q_FES(y|X)
        q_fes = self._compute_empirical_distribution(fes_predictions)

        # Get probability of original prediction
        prob_original = q_fes.get(original_prediction, 0.0)

        # Add epsilon to avoid log(0)
        prob_original = max(prob_original, self.epsilon)

        # Compute U_FES = -log(q_FES(y=ŷ|X))
        u_fes = -np.log(prob_original)

        return u_fes

    def compute_u_fcs(
        self,
        fcs_predictions: List[str],
        original_prediction: str
    ) -> float:
        """
        Compute U_FCS: sensitivity uncertainty from complementary samples

        U_FCS = -log(Σ_{y≠ŷ} q_FCS(y|X))

        where q_FCS(y|X) is the empirical distribution over FCS samples

        Args:
            fcs_predictions: Predictions on FCS samples
            original_prediction: Original prediction ŷ

        Returns:
            U_FCS score (higher = more uncertain)
        """
        if not fcs_predictions:
            return 0.0

        # Compute empirical distribution q_FCS(y|X)
        q_fcs = self._compute_empirical_distribution(fcs_predictions)

        # Compute probability of predictions different from original
        # Σ_{y≠ŷ} q_FCS(y|X)
        prob_complementary = sum(
            prob for option, prob in q_fcs.items()
            if option != original_prediction
        )

        # Add epsilon to avoid log(0)
        prob_complementary = max(prob_complementary, self.epsilon)

        # Compute U_FCS = -log(Σ_{y≠ŷ} q_FCS(y|X))
        u_fcs = -np.log(prob_complementary)

        return u_fcs

    def _compute_empirical_distribution(
        self,
        predictions: List[str]
    ) -> Dict[str, float]:
        """
        Compute empirical probability distribution from predictions

        Args:
            predictions: List of predictions

        Returns:
            Dictionary mapping options to probabilities
        """
        if not predictions:
            return {}

        # Count predictions
        counts = Counter(predictions)
        total = len(predictions)

        # Compute probabilities
        distribution = {
            option: count / total
            for option, count in counts.items()
        }

        return distribution

    def compute_confidence_score(
        self,
        uncertainty: float,
        method: str = "reciprocal"
    ) -> float:
        """
        Convert uncertainty to confidence score

        Args:
            uncertainty: Uncertainty value
            method: Conversion method ('reciprocal' or 'exp')

        Returns:
            Confidence score (higher = more confident)
        """
        if method == "reciprocal":
            return 1.0 / (1.0 + uncertainty)
        elif method == "exp":
            return np.exp(-uncertainty)
        else:
            raise ValueError(f"Unknown method: {method}")


class BaselineUncertaintyEstimator:
    """Baseline uncertainty estimation methods for comparison"""

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon

    def output_entropy(
        self,
        predictions: List[str],
        options: List[str] = ['A', 'B', 'C', 'D']
    ) -> float:
        """
        Compute output entropy (OE) from stochastic sampling

        H(q(y|x)) = -Σ q(y|x) log q(y|x)

        Args:
            predictions: List of predictions from stochastic sampling
            options: All possible options

        Returns:
            Entropy value
        """
        # Compute empirical distribution
        counts = Counter(predictions)
        total = len(predictions)

        probs = {option: counts.get(option, 0) / total for option in options}

        # Compute entropy
        entropy = 0.0
        for prob in probs.values():
            if prob > self.epsilon:
                entropy -= prob * np.log(prob)

        return entropy

    def prediction_consistency(
        self,
        predictions: List[str]
    ) -> float:
        """
        Compute prediction consistency (inverse of uncertainty)

        Args:
            predictions: List of predictions

        Returns:
            Consistency score (0-1, higher = more consistent)
        """
        if not predictions:
            return 0.0

        # Most common prediction count / total
        counts = Counter(predictions)
        most_common_count = counts.most_common(1)[0][1]
        consistency = most_common_count / len(predictions)

        return consistency

    def rephrase_uncertainty(
        self,
        predictions: List[str]
    ) -> float:
        """
        Compute uncertainty based on rephrase disagreement

        Args:
            predictions: Predictions on rephrased questions

        Returns:
            Uncertainty score
        """
        # Use consistency as measure
        consistency = self.prediction_consistency(predictions)

        # Convert to uncertainty (lower consistency = higher uncertainty)
        uncertainty = -np.log(max(consistency, self.epsilon))

        return uncertainty


if __name__ == "__main__":
    # Test FESTA uncertainty computation
    print("Testing FESTA Uncertainty Computation...")

    # Initialize estimator
    festa = FESTAUncertainty()

    # Test case 1: Consistent model (low uncertainty)
    print("\n" + "="*60)
    print("Test Case 1: Consistent Model (Low Uncertainty)")
    print("="*60)

    original = "A"
    fes_preds = ["A"] * 50 + ["B"] * 5 + ["C"] * 3 + ["D"] * 2  # Mostly A
    fcs_preds = ["B"] * 30 + ["C"] * 20 + ["D"] * 10  # No A (sensitive to complement)

    result1 = festa.compute_festa(fes_preds, fcs_preds, original)
    print(f"Original prediction: {original}")
    print(f"FES predictions: {Counter(fes_preds)}")
    print(f"FCS predictions: {Counter(fcs_preds)}")
    print(f"\nUncertainty scores:")
    print(f"  U_FES: {result1['U_FES']:.4f} (consistency)")
    print(f"  U_FCS: {result1['U_FCS']:.4f} (sensitivity)")
    print(f"  U_FESTA: {result1['U_FESTA']:.4f}")
    print(f"Confidence: {festa.compute_confidence_score(result1['U_FESTA']):.4f}")

    # Test case 2: Inconsistent model (high uncertainty)
    print("\n" + "="*60)
    print("Test Case 2: Inconsistent Model (High Uncertainty)")
    print("="*60)

    original = "A"
    fes_preds = ["A"] * 15 + ["B"] * 15 + ["C"] * 15 + ["D"] * 15  # Inconsistent
    fcs_preds = ["A"] * 30 + ["B"] * 20 + ["C"] * 10  # Still predicts A (not sensitive)

    result2 = festa.compute_festa(fes_preds, fcs_preds, original)
    print(f"Original prediction: {original}")
    print(f"FES predictions: {Counter(fes_preds)}")
    print(f"FCS predictions: {Counter(fcs_preds)}")
    print(f"\nUncertainty scores:")
    print(f"  U_FES: {result2['U_FES']:.4f} (consistency)")
    print(f"  U_FCS: {result2['U_FCS']:.4f} (sensitivity)")
    print(f"  U_FESTA: {result2['U_FESTA']:.4f}")
    print(f"Confidence: {festa.compute_confidence_score(result2['U_FESTA']):.4f}")

    # Test case 3: Mode collapse (low entropy hallucination)
    print("\n" + "="*60)
    print("Test Case 3: Mode Collapse (Low Entropy Hallucination)")
    print("="*60)

    original = "A"
    fes_preds = ["A"] * 55 + ["B"] * 5  # Very consistent
    fcs_preds = ["A"] * 55 + ["B"] * 5  # Not sensitive (mode collapse!)

    result3 = festa.compute_festa(fes_preds, fcs_preds, original)
    print(f"Original prediction: {original}")
    print(f"FES predictions: {Counter(fes_preds)}")
    print(f"FCS predictions: {Counter(fcs_preds)}")
    print(f"\nUncertainty scores:")
    print(f"  U_FES: {result3['U_FES']:.4f} (consistency)")
    print(f"  U_FCS: {result3['U_FCS']:.4f} (sensitivity - HIGH due to mode collapse!)")
    print(f"  U_FESTA: {result3['U_FESTA']:.4f}")
    print(f"Confidence: {festa.compute_confidence_score(result3['U_FESTA']):.4f}")
    print("\n⚠️  High U_FCS detects mode collapse that entropy-based methods would miss!")

    # Test baseline methods
    print("\n" + "="*60)
    print("Testing Baseline Methods")
    print("="*60)

    baseline = BaselineUncertaintyEstimator()

    # Output entropy for case 1
    oe1 = baseline.output_entropy(fes_preds)
    print(f"\nCase 1 Output Entropy: {oe1:.4f}")

    # Output entropy for case 2 (should be higher)
    oe2 = baseline.output_entropy(["A"]*15 + ["B"]*15 + ["C"]*15 + ["D"]*15)
    print(f"Case 2 Output Entropy: {oe2:.4f}")

    # Consistency scores
    cons1 = baseline.prediction_consistency(fes_preds)
    cons2 = baseline.prediction_consistency(["A"]*15 + ["B"]*15 + ["C"]*15 + ["D"]*15)
    print(f"\nCase 1 Consistency: {cons1:.4f}")
    print(f"Case 2 Consistency: {cons2:.4f}")

    print("\n" + "="*60)
    print("FESTA Uncertainty Computation Test Completed!")
    print("="*60)
