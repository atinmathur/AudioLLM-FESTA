"""
Main experiment script for FESTA implementation
Runs end-to-end FESTA pipeline on TREA dataset with Qwen2-Audio
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_trea_dataset, TREADataLoader
from src.model_wrapper import Qwen2AudioWrapper
from src.fes_generator import FESGenerator
from src.fcs_generator import FCSGenerator
from src.uncertainty import FESTAUncertainty
from src.metrics import (
    compute_auroc,
    compute_accuracy,
    evaluate_selective_prediction,
    compute_task_wise_metrics,
    compare_methods
)
from src.baselines import BaselineUncertainty, AugmentationGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FESTAExperiment:
    """Main FESTA experiment runner"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize experiment

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Create output directory
        self.output_dir = Path(self.config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        logger.info("Initializing components...")
        self._initialize_components()

        # Results storage
        self.results = {
            'predictions': [],
            'ground_truths': [],
            'tasks': [],
            'uncertainties': {},
            'metadata': []
        }

    def _initialize_components(self):
        """Initialize all experiment components"""

        # Load dataset
        logger.info("Loading TREA dataset...")
        self.dataset = load_trea_dataset(
            data_dir=self.config['dataset']['data_dir'],
            tasks=self.config['dataset']['tasks'],
            samples_per_task=self.config['dataset']['samples_per_task'],
            random_seed=self.config['dataset']['random_seed']
        )

        # Initialize model
        logger.info("Initializing Qwen2-Audio model...")
        self.model = Qwen2AudioWrapper(
            model_name=self.config['model']['name'],
            device=self.config['model']['device'],
            dtype=self.config['model']['dtype'],
            max_length=self.config['model']['max_length']
        )

        # Initialize FES generator
        self.fes_generator = FESGenerator(
            n_audio_samples=self.config['festa']['n_fes_audio'],
            n_text_samples=self.config['festa']['n_fes_text'],
            sr=16000
        )

        # Initialize FCS generator
        self.fcs_generator = FCSGenerator(
            n_audio_samples=self.config['festa']['n_fcs_audio'],
            n_text_samples=self.config['festa']['n_fcs_text'],
            sr=16000,
            synthetic_silence_dir=os.path.join(
                self.config['dataset']['data_dir'],
                'synthetic_silences'
            )
        )

        # Initialize uncertainty estimators
        self.festa_uncertainty = FESTAUncertainty()
        self.baseline_uncertainty = BaselineUncertainty()

        # Initialize augmentation generator for baselines
        self.aug_generator = AugmentationGenerator()

    def process_sample(self, sample: Dict, sample_idx: int) -> Dict:
        """
        Process a single sample through FESTA pipeline

        Args:
            sample: Sample dictionary from dataset
            sample_idx: Sample index for logging

        Returns:
            Results dictionary
        """
        logger.info(f"\nProcessing sample {sample_idx + 1}/{len(self.dataset)}")
        logger.info(f"Task: {sample['task']}, ID: {sample['id']}")

        results = {
            'sample_id': sample['id'],
            'task': sample['task'],
            'question': sample['question'],
            'ground_truth': sample['correct_answer']
        }

        # Get original prediction
        logger.debug("Getting original prediction...")
        original_answer, original_probs = self.model.predict(
            sample['audio_path'],
            sample['question'],
            sample['options'],
            return_probs=True
        )
        results['prediction'] = original_answer

        # Generate FES samples
        logger.debug("Generating FES samples...")
        fes_samples = self.fes_generator.generate(
            sample['audio_path'],
            sample['question'],
            sample['task'],
            sample['options']
        )

        # Get predictions on FES samples
        logger.debug(f"Getting predictions on {len(fes_samples)} FES samples...")
        fes_predictions = []
        for fes_sample in tqdm(fes_samples, desc="FES predictions", leave=False):
            pred, _ = self.model.predict(
                fes_sample['audio_path'],
                fes_sample['question'],
                fes_sample['options']
            )
            fes_predictions.append(pred)

        # Generate FCS samples
        logger.debug("Generating FCS samples...")
        fcs_samples = self.fcs_generator.generate(
            sample['audio_path'],
            sample['question'],
            sample['task'],
            sample['options'],
            original_answer
        )

        # Get predictions on FCS samples
        logger.debug(f"Getting predictions on {len(fcs_samples)} FCS samples...")
        fcs_predictions = []
        for fcs_sample in tqdm(fcs_samples, desc="FCS predictions", leave=False):
            pred, _ = self.model.predict(
                fcs_sample['audio_path'],
                fcs_sample['question'],
                fcs_sample['options']
            )
            fcs_predictions.append(pred)

        # Compute FESTA uncertainty
        logger.debug("Computing FESTA uncertainty...")
        festa_scores = self.festa_uncertainty.compute_festa(
            fes_predictions,
            fcs_predictions,
            original_answer
        )
        results['festa'] = festa_scores

        # Compute baseline uncertainties if enabled
        if self.config['baselines']['output_entropy']['enabled']:
            logger.debug("Computing output entropy baseline...")
            # Generate stochastic samples
            stochastic_probs = self.model.predict_with_sampling(
                sample['audio_path'],
                sample['question'],
                sample['options'],
                num_samples=20,
                temperature=1.0
            )
            stochastic_preds = [max(stochastic_probs, key=stochastic_probs.get)] * 20  # Simplified
            oe = self.baseline_uncertainty.output_entropy(stochastic_preds)
            results['OE'] = oe

        if self.config['baselines']['rephrase_uncertainty']['enabled']:
            logger.debug("Computing rephrase uncertainty baseline...")
            rephrases = self.aug_generator.generate_text_rephrases(
                sample['question'],
                sample['task'],
                n_rephrases=5
            )
            rephrase_preds = []
            for rephrase in rephrases:
                pred, _ = self.model.predict(
                    sample['audio_path'],
                    rephrase,
                    sample['options']
                )
                rephrase_preds.append(pred)
            ru = self.baseline_uncertainty.rephrase_uncertainty(rephrase_preds)
            results['RU'] = ru

        logger.info(f"✓ Sample {sample_idx + 1} completed")
        logger.info(f"  Prediction: {original_answer}, Ground Truth: {sample['correct_answer']}")
        logger.info(f"  FESTA Uncertainty: {festa_scores['U_FESTA']:.4f}")

        return results

    def run_experiment(self):
        """Run the full FESTA experiment"""

        logger.info("\n" + "="*80)
        logger.info("Starting FESTA Experiment")
        logger.info("="*80)
        logger.info(f"Dataset: {len(self.dataset)} samples")
        logger.info(f"Tasks: {self.config['dataset']['tasks']}")
        logger.info(f"Model: {self.config['model']['name']}")

        # Process all samples
        all_results = []
        for idx, sample in enumerate(self.dataset.data):
            try:
                result = self.process_sample(sample, idx)
                all_results.append(result)

                # Save intermediate results
                if self.config['experiment']['save_intermediate']:
                    self._save_intermediate_results(all_results)

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue

        # Compile results
        logger.info("\nCompiling results...")
        self._compile_results(all_results)

        # Compute metrics
        logger.info("\nComputing metrics...")
        self._compute_metrics()

        # Save final results
        logger.info("\nSaving results...")
        self._save_final_results()

        logger.info("\n" + "="*80)
        logger.info("FESTA Experiment Completed Successfully!")
        logger.info("="*80)

    def _compile_results(self, all_results: List[Dict]):
        """Compile results from all samples"""

        for result in all_results:
            self.results['predictions'].append(result['prediction'])
            self.results['ground_truths'].append(result['ground_truth'])
            self.results['tasks'].append(result['task'])
            self.results['metadata'].append({
                'sample_id': result['sample_id'],
                'task': result['task'],
                'question': result['question']
            })

            # Store uncertainties
            for key in ['festa', 'OE', 'RU']:
                if key in result:
                    if key not in self.results['uncertainties']:
                        self.results['uncertainties'][key] = []

                    if key == 'festa':
                        self.results['uncertainties'][key].append(result[key]['U_FESTA'])
                    else:
                        self.results['uncertainties'][key].append(result[key])

    def _compute_metrics(self):
        """Compute evaluation metrics"""

        self.metrics = {}

        predictions = self.results['predictions']
        ground_truths = self.results['ground_truths']
        tasks = self.results['tasks']

        # Overall accuracy
        overall_accuracy = compute_accuracy(predictions, ground_truths)
        self.metrics['overall_accuracy'] = overall_accuracy

        logger.info(f"\nOverall Accuracy: {overall_accuracy:.4f}")

        # Compute AUROC for each method
        method_results = {}

        for method_name, uncertainties in self.results['uncertainties'].items():
            auroc = compute_auroc(uncertainties, predictions, ground_truths)

            method_results[method_name.upper()] = {
                'auroc': auroc,
                'accuracy': overall_accuracy
            }

            logger.info(f"{method_name.upper()} AUROC: {auroc:.4f}")

        # Compare methods
        compare_methods(method_results)

        # Task-wise metrics
        logger.info("\nTask-wise Performance:")
        logger.info("-" * 60)

        for method_name, uncertainties in self.results['uncertainties'].items():
            logger.info(f"\n{method_name.upper()}:")
            task_metrics = compute_task_wise_metrics(
                uncertainties, predictions, ground_truths, tasks
            )

            for task, metrics in task_metrics.items():
                logger.info(f"  {task}: AUROC={metrics['auroc']:.4f}, "
                          f"Acc={metrics['accuracy']:.4f}, "
                          f"N={metrics['n_samples']}")

        self.metrics['method_results'] = method_results

    def _save_intermediate_results(self, results: List[Dict]):
        """Save intermediate results"""
        output_path = self.output_dir / "intermediate_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    def _save_final_results(self):
        """Save final results to files"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save predictions
        predictions_file = self.output_dir / f"predictions_{timestamp}.json"
        with open(predictions_file, 'w') as f:
            json.dump({
                'predictions': self.results['predictions'],
                'ground_truths': self.results['ground_truths'],
                'tasks': self.results['tasks'],
                'metadata': self.results['metadata']
            }, f, indent=2)

        # Save uncertainties
        uncertainties_file = self.output_dir / f"uncertainties_{timestamp}.json"
        with open(uncertainties_file, 'w') as f:
            json.dump(self.results['uncertainties'], f, indent=2)

        # Save metrics
        metrics_file = self.output_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"\nResults saved to {self.output_dir}")
        logger.info(f"  Predictions: {predictions_file.name}")
        logger.info(f"  Uncertainties: {uncertainties_file.name}")
        logger.info(f"  Metrics: {metrics_file.name}")


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(description="Run FESTA experiment")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Run experiment
    experiment = FESTAExperiment(config_path=args.config)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
