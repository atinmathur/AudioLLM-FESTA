"""
Baseline uncertainty estimation methods for comparison with FESTA

Methods implemented:
1. Output Entropy (OE) - Stochastic sampling entropy
2. Verbalized Confidence (VC) - LLM self-assessment
3. Input Augmentation (IA) - Audio/text augmentation entropy
4. Rephrase Uncertainty (RU) - Answer consistency across rephrases
5. Black-box Uncertainty (BU) - Top-K prompting with sampling
"""

import numpy as np
from typing import List, Dict, Optional
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineUncertainty:
    """Collection of baseline uncertainty estimation methods"""

    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize baseline estimator

        Args:
            epsilon: Small value to avoid log(0)
        """
        self.epsilon = epsilon

    def output_entropy(
        self,
        predictions: List[str],
        options: List[str] = ['A', 'B', 'C', 'D']
    ) -> float:
        """
        Compute Output Entropy (OE) from stochastic sampling

        H(q(y|x)) = -Σ q(y|x) log q(y|x)

        Args:
            predictions: List of predictions from multiple stochastic samples
            options: All possible answer options

        Returns:
            Entropy value (higher = more uncertain)
        """
        if not predictions:
            return 0.0

        # Compute empirical distribution
        counts = Counter(predictions)
        total = len(predictions)

        # Calculate entropy
        entropy = 0.0
        for option in options:
            prob = counts.get(option, 0) / total
            if prob > self.epsilon:
                entropy -= prob * np.log(prob)

        return entropy

    def verbalized_confidence(
        self,
        confidence_score: float,
        normalize: bool = True
    ) -> float:
        """
        Compute uncertainty from verbalized confidence

        Args:
            confidence_score: Confidence score from model (0-100 or 0-1)
            normalize: Whether to normalize to 0-1 range

        Returns:
            Uncertainty value (higher = more uncertain)
        """
        if normalize and confidence_score > 1.0:
            confidence_score = confidence_score / 100.0

        # Convert confidence to uncertainty
        uncertainty = -np.log(max(confidence_score, self.epsilon))

        return uncertainty

    def input_augmentation_entropy(
        self,
        predictions: List[str],
        options: List[str] = ['A', 'B', 'C', 'D']
    ) -> float:
        """
        Compute uncertainty from input augmentations
        Similar to output entropy but using augmented inputs

        Args:
            predictions: Predictions on augmented inputs
            options: All possible answer options

        Returns:
            Entropy value (higher = more uncertain)
        """
        return self.output_entropy(predictions, options)

    def rephrase_uncertainty(
        self,
        predictions: List[str]
    ) -> float:
        """
        Compute uncertainty based on prediction consistency across rephrases

        Args:
            predictions: Predictions on rephrased questions

        Returns:
            Uncertainty value (higher = more uncertain)
        """
        if not predictions:
            return 0.0

        # Compute consistency (most common answer frequency)
        counts = Counter(predictions)
        most_common_count = counts.most_common(1)[0][1]
        consistency = most_common_count / len(predictions)

        # Convert to uncertainty
        uncertainty = -np.log(max(consistency, self.epsilon))

        return uncertainty

    def blackbox_uncertainty(
        self,
        predictions: List[str],
        options: List[str] = ['A', 'B', 'C', 'D']
    ) -> float:
        """
        Black-box uncertainty from top-K prompting and sampling

        Args:
            predictions: Predictions from different prompts and samples
            options: All possible answer options

        Returns:
            Uncertainty value (higher = more uncertain)
        """
        # Use entropy over predictions
        return self.output_entropy(predictions, options)

    def compute_all_baselines(
        self,
        stochastic_predictions: Optional[List[str]] = None,
        verbalized_conf: Optional[float] = None,
        augmented_predictions: Optional[List[str]] = None,
        rephrased_predictions: Optional[List[str]] = None,
        blackbox_predictions: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute all baseline uncertainty measures

        Args:
            stochastic_predictions: For output entropy
            verbalized_conf: For verbalized confidence
            augmented_predictions: For input augmentation
            rephrased_predictions: For rephrase uncertainty
            blackbox_predictions: For black-box uncertainty

        Returns:
            Dictionary of uncertainty scores for each method
        """
        results = {}

        if stochastic_predictions is not None:
            results['OE'] = self.output_entropy(stochastic_predictions)

        if verbalized_conf is not None:
            results['VC'] = self.verbalized_confidence(verbalized_conf)

        if augmented_predictions is not None:
            results['IA'] = self.input_augmentation_entropy(augmented_predictions)

        if rephrased_predictions is not None:
            results['RU'] = self.rephrase_uncertainty(rephrased_predictions)

        if blackbox_predictions is not None:
            results['BU'] = self.blackbox_uncertainty(blackbox_predictions)

        return results


class AugmentationGenerator:
    """Generate augmented versions of inputs for baseline methods"""

    def __init__(self):
        pass

    def generate_audio_augmentations(
        self,
        audio_path: str,
        n_augmentations: int = 5
    ) -> List[str]:
        """
        Generate audio augmentations for IA baseline

        Args:
            audio_path: Path to original audio
            n_augmentations: Number of augmentations

        Returns:
            List of augmented audio paths
        """
        # Import here to avoid circular dependency
        from .utils import AudioProcessor
        import tempfile
        import os

        processor = AudioProcessor()
        audio, sr = processor.load_audio(audio_path)

        augmented_paths = [audio_path]  # Include original

        transformations = [
            lambda a: processor.add_noise(a, snr_db=30),
            lambda a: processor.adjust_volume(a, 0.15),
            lambda a: processor.adjust_volume(a, -0.15),
            lambda a: processor.add_silence(a, sr, 0.2, 'middle'),
        ]

        for i, transform in enumerate(transformations[:n_augmentations-1]):
            try:
                augmented = transform(audio)
                temp_path = os.path.join(
                    tempfile.gettempdir(),
                    f"aug_{i}_{os.path.basename(audio_path)}"
                )
                processor.save_audio(augmented, temp_path, sr)
                augmented_paths.append(temp_path)
            except Exception as e:
                logger.warning(f"Failed to apply augmentation {i}: {e}")
                augmented_paths.append(audio_path)

        return augmented_paths

    def generate_text_rephrases(
        self,
        question: str,
        task: str,
        n_rephrases: int = 4
    ) -> List[str]:
        """
        Generate text rephrases for RU baseline

        Args:
            question: Original question
            task: Task type
            n_rephrases: Number of rephrases

        Returns:
            List of rephrased questions
        """
        # Use simple rule-based rephrasing
        # In production, could use LLM for better paraphrasing

        rephrases = [question]  # Include original

        if task == "count":
            if "how many" in question.lower():
                rephrases.append(question.replace("How many", "What is the number of"))
                rephrases.append(question.replace("how many", "Count the"))
            elif "number of" in question.lower():
                rephrases.append(question.replace("number of", "count of"))

        elif task == "order":
            if "first" in question.lower():
                rephrases.append(question.replace("first", "initial"))
            elif "second" in question.lower():
                rephrases.append(question.replace("second", "next"))

        elif task == "duration":
            if "longest" in question.lower():
                rephrases.append(question.replace("longest", "maximum"))
            elif "shortest" in question.lower():
                rephrases.append(question.replace("shortest", "minimum"))

        # Pad if needed
        while len(rephrases) < n_rephrases:
            rephrases.append(question)

        return rephrases[:n_rephrases]


if __name__ == "__main__":
    # Test baseline methods
    print("Testing Baseline Uncertainty Methods...")

    baseline = BaselineUncertainty()

    # Test Output Entropy
    print("\n" + "="*60)
    print("Test 1: Output Entropy (OE)")
    print("="*60)

    # High entropy case (uncertain)
    preds_uncertain = ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10
    oe_high = baseline.output_entropy(preds_uncertain)
    print(f"Uncertain predictions: {Counter(preds_uncertain)}")
    print(f"Output Entropy: {oe_high:.4f} (high uncertainty)")

    # Low entropy case (certain)
    preds_certain = ['A'] * 35 + ['B'] * 3 + ['C'] * 1 + ['D'] * 1
    oe_low = baseline.output_entropy(preds_certain)
    print(f"\nCertain predictions: {Counter(preds_certain)}")
    print(f"Output Entropy: {oe_low:.4f} (low uncertainty)")

    # Test Verbalized Confidence
    print("\n" + "="*60)
    print("Test 2: Verbalized Confidence (VC)")
    print("="*60)

    vc_high_conf = baseline.verbalized_confidence(90.0, normalize=True)
    vc_low_conf = baseline.verbalized_confidence(30.0, normalize=True)
    print(f"High confidence (90%): Uncertainty = {vc_high_conf:.4f}")
    print(f"Low confidence (30%): Uncertainty = {vc_low_conf:.4f}")

    # Test Rephrase Uncertainty
    print("\n" + "="*60)
    print("Test 3: Rephrase Uncertainty (RU)")
    print("="*60)

    rephrases_consistent = ['A', 'A', 'A', 'A', 'B']
    rephrases_inconsistent = ['A', 'B', 'C', 'A', 'D']

    ru_consistent = baseline.rephrase_uncertainty(rephrases_consistent)
    ru_inconsistent = baseline.rephrase_uncertainty(rephrases_inconsistent)

    print(f"Consistent rephrases: {rephrases_consistent}")
    print(f"Rephrase Uncertainty: {ru_consistent:.4f}")
    print(f"\nInconsistent rephrases: {rephrases_inconsistent}")
    print(f"Rephrase Uncertainty: {ru_inconsistent:.4f}")

    # Test compute_all_baselines
    print("\n" + "="*60)
    print("Test 4: Compute All Baselines")
    print("="*60)

    all_uncertainties = baseline.compute_all_baselines(
        stochastic_predictions=preds_certain,
        verbalized_conf=75.0,
        augmented_predictions=preds_uncertain,
        rephrased_predictions=rephrases_consistent,
        blackbox_predictions=preds_certain
    )

    print("All baseline uncertainties:")
    for method, uncertainty in sorted(all_uncertainties.items()):
        print(f"  {method}: {uncertainty:.4f}")

    # Test augmentation generator
    print("\n" + "="*60)
    print("Test 5: Augmentation Generator")
    print("="*60)

    aug_gen = AugmentationGenerator()

    # Test text rephrases
    question = "How many distinct sound sources are in the audio?"
    rephrases = aug_gen.generate_text_rephrases(question, "count", n_rephrases=4)

    print(f"Original question: {question}")
    print(f"\nGenerated rephrases:")
    for i, rephrase in enumerate(rephrases, 1):
        print(f"  {i}. {rephrase}")

    print("\nBaseline methods testing completed successfully!")
