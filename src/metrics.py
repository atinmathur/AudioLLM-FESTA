"""
Evaluation metrics for FESTA
- AUROC for misprediction detection
- Selective prediction metrics
- Coverage vs accuracy curves
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, roc_curve
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_auroc(
    uncertainties: List[float],
    predictions: List[str],
    ground_truths: List[str]
) -> float:
    """
    Compute AUROC for detecting mispredictions

    AUROC = AUC(1/U, 𝟙{ŷ=y_target})

    Args:
        uncertainties: List of uncertainty scores
        predictions: List of model predictions
        ground_truths: List of ground truth answers

    Returns:
        AUROC score (0-1, higher is better)
    """
    if len(uncertainties) != len(predictions) or len(predictions) != len(ground_truths):
        raise ValueError("Length mismatch between uncertainties, predictions, and ground truths")

    # Create binary labels: 1 if correct, 0 if incorrect
    correctness = [
        1 if pred == gt else 0
        for pred, gt in zip(predictions, ground_truths)
    ]

    # Convert uncertainties to confidence scores (1/U)
    confidences = [1.0 / (1.0 + u) for u in uncertainties]

    # Compute AUROC
    # We want high confidence for correct predictions
    # and low confidence for incorrect predictions
    try:
        auroc = roc_auc_score(correctness, confidences)
        return auroc
    except ValueError as e:
        logger.warning(f"Error computing AUROC: {e}")
        return 0.5  # Random baseline


def compute_accuracy(
    predictions: List[str],
    ground_truths: List[str]
) -> float:
    """
    Compute prediction accuracy

    Args:
        predictions: List of predictions
        ground_truths: List of ground truth answers

    Returns:
        Accuracy (0-1)
    """
    if len(predictions) == 0 or len(ground_truths) == 0:
        raise ValueError(
            "Cannot compute accuracy: predictions or ground_truths is empty. "
            "No samples were successfully processed."
        )

    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths"
        )

    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    return correct / len(predictions)


def evaluate_selective_prediction(
    uncertainties: List[float],
    predictions: List[str],
    ground_truths: List[str],
    coverage_levels: Optional[List[float]] = None
) -> Dict[str, List[float]]:
    """
    Evaluate selective prediction performance

    Args:
        uncertainties: Uncertainty scores
        predictions: Model predictions
        ground_truths: Ground truth answers
        coverage_levels: Coverage levels to evaluate (default: 0.1 to 1.0 in steps of 0.1)

    Returns:
        Dictionary with coverage, accuracy, and selective_risk lists
    """
    if coverage_levels is None:
        coverage_levels = np.arange(0.1, 1.01, 0.1).tolist()

    n_samples = len(predictions)

    # Sort samples by uncertainty (ascending = most confident first)
    sorted_indices = np.argsort(uncertainties)

    results = {
        'coverage': [],
        'accuracy': [],
        'selective_risk': []
    }

    for coverage in coverage_levels:
        # Select top-coverage samples (most confident)
        n_selected = max(1, int(coverage * n_samples))
        selected_indices = sorted_indices[:n_selected]

        # Compute accuracy on selected samples
        selected_preds = [predictions[i] for i in selected_indices]
        selected_gts = [ground_truths[i] for i in selected_indices]

        accuracy = compute_accuracy(selected_preds, selected_gts)

        # Selective risk = 1 - accuracy
        selective_risk = 1.0 - accuracy

        results['coverage'].append(coverage)
        results['accuracy'].append(accuracy)
        results['selective_risk'].append(selective_risk)

    return results


def compute_task_wise_metrics(
    uncertainties: List[float],
    predictions: List[str],
    ground_truths: List[str],
    tasks: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for each task

    Args:
        uncertainties: Uncertainty scores
        predictions: Model predictions
        ground_truths: Ground truth answers
        tasks: Task labels for each sample

    Returns:
        Dictionary mapping task names to metrics
    """
    unique_tasks = sorted(set(tasks))
    results = {}

    for task in unique_tasks:
        # Get indices for this task
        task_indices = [i for i, t in enumerate(tasks) if t == task]

        # Extract task-specific data
        task_uncertainties = [uncertainties[i] for i in task_indices]
        task_predictions = [predictions[i] for i in task_indices]
        task_ground_truths = [ground_truths[i] for i in task_indices]

        # Compute metrics
        task_auroc = compute_auroc(
            task_uncertainties,
            task_predictions,
            task_ground_truths
        )

        task_accuracy = compute_accuracy(
            task_predictions,
            task_ground_truths
        )

        results[task] = {
            'auroc': task_auroc,
            'accuracy': task_accuracy,
            'n_samples': len(task_indices)
        }

    return results


def compare_methods(
    results_dict: Dict[str, Dict[str, float]]
) -> None:
    """
    Print comparison table of different methods

    Args:
        results_dict: Dictionary mapping method names to their metrics
    """
    print("\n" + "="*80)
    print("Method Comparison")
    print("="*80)
    print(f"{'Method':<30} {'AUROC':>10} {'Accuracy':>10} {'Improvement':>15}")
    print("-"*80)

    # Sort by AUROC
    sorted_methods = sorted(
        results_dict.items(),
        key=lambda x: x[1].get('auroc', 0),
        reverse=True
    )

    best_auroc = sorted_methods[0][1]['auroc'] if sorted_methods else 0

    for method, metrics in sorted_methods:
        auroc = metrics.get('auroc', 0)
        accuracy = metrics.get('accuracy', 0)

        # Compute relative improvement over second-best
        if method == sorted_methods[0][0]:
            improvement = "-"
        else:
            second_best = sorted_methods[1][1]['auroc'] if len(sorted_methods) > 1 else 0
            rel_improvement = ((auroc - second_best) / second_best * 100) if second_best > 0 else 0
            improvement = f"{rel_improvement:+.1f}%"

        print(f"{method:<30} {auroc:>10.4f} {accuracy:>10.4f} {improvement:>15}")

    print("="*80)


def plot_coverage_accuracy_curve(
    selective_results: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot coverage vs accuracy curve

    Args:
        selective_results: Results from evaluate_selective_prediction
        save_path: Path to save plot (optional)
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(
            selective_results['coverage'],
            selective_results['accuracy'],
            marker='o',
            linewidth=2,
            markersize=6
        )

        plt.xlabel('Coverage', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Selective Prediction: Coverage vs Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.show()

    except ImportError:
        logger.warning("matplotlib not available, skipping plot")


def plot_roc_curve(
    uncertainties: List[float],
    predictions: List[str],
    ground_truths: List[str],
    save_path: Optional[str] = None
):
    """
    Plot ROC curve

    Args:
        uncertainties: Uncertainty scores
        predictions: Model predictions
        ground_truths: Ground truth answers
        save_path: Path to save plot (optional)
    """
    try:
        import matplotlib.pyplot as plt

        # Create binary labels
        correctness = [1 if pred == gt else 0 for pred, gt in zip(predictions, ground_truths)]
        confidences = [1.0 / (1.0 + u) for u in uncertainties]

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(correctness, confidences)
        auroc = roc_auc_score(correctness, confidences)

        # Plot
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, linewidth=2, label=f'AUROC = {auroc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.show()

    except ImportError:
        logger.warning("matplotlib not available, skipping plot")


if __name__ == "__main__":
    # Test metrics
    print("Testing FESTA Metrics...")

    # Generate synthetic test data
    np.random.seed(42)

    n_samples = 100
    predictions = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    ground_truths = np.random.choice(['A', 'B', 'C', 'D'], n_samples)

    # Generate uncertainties (correlated with correctness)
    uncertainties = []
    for pred, gt in zip(predictions, ground_truths):
        if pred == gt:
            # Lower uncertainty for correct predictions
            uncertainties.append(np.random.uniform(0, 2))
        else:
            # Higher uncertainty for incorrect predictions
            uncertainties.append(np.random.uniform(2, 5))

    # Test AUROC
    print("\n" + "="*60)
    print("Testing AUROC Computation")
    print("="*60)

    auroc = compute_auroc(uncertainties, predictions, ground_truths)
    print(f"AUROC: {auroc:.4f}")

    accuracy = compute_accuracy(predictions, ground_truths)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Test selective prediction
    print("\n" + "="*60)
    print("Testing Selective Prediction")
    print("="*60)

    selective_results = evaluate_selective_prediction(
        uncertainties, predictions, ground_truths
    )

    print(f"{'Coverage':<15} {'Accuracy':<15} {'Selective Risk':<15}")
    print("-"*45)
    for cov, acc, risk in zip(
        selective_results['coverage'],
        selective_results['accuracy'],
        selective_results['selective_risk']
    ):
        print(f"{cov:<15.2f} {acc:<15.4f} {risk:<15.4f}")

    # Test task-wise metrics
    print("\n" + "="*60)
    print("Testing Task-wise Metrics")
    print("="*60)

    tasks = np.random.choice(['count', 'order', 'duration'], n_samples)

    task_metrics = compute_task_wise_metrics(
        uncertainties, predictions, ground_truths, tasks.tolist()
    )

    for task, metrics in task_metrics.items():
        print(f"\n{task.upper()}:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Samples: {metrics['n_samples']}")

    # Test method comparison
    print("\n" + "="*60)
    print("Testing Method Comparison")
    print("="*60)

    # Create dummy results for different methods
    methods_results = {
        'FESTA': {'auroc': 0.89, 'accuracy': 0.75},
        'Output Entropy': {'auroc': 0.71, 'accuracy': 0.75},
        'Verbalized Confidence': {'auroc': 0.65, 'accuracy': 0.75},
        'Rephrase Uncertainty': {'auroc': 0.68, 'accuracy': 0.75},
        'Black-box Uncertainty': {'auroc': 0.59, 'accuracy': 0.75}
    }

    compare_methods(methods_results)

    print("\nMetrics testing completed successfully!")
