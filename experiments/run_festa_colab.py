"""
FESTA Experiment Runner for Google Colab
Modified with checkpoint/resume capability for handling session timeouts
"""

import os
import sys
import json
import yaml
import logging
import argparse
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_trea_dataset
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

# Try to import torch for GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ColabFESTAExperiment:
    """FESTA experiment runner with Colab-specific optimizations"""

    def __init__(self, config_path: str = "config_colab.yaml"):
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

        # Checkpoint file
        self.checkpoint_file = Path(self.config['experiment']['checkpoint_file'])

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

        # Load checkpoint if exists
        self.completed_samples = set()
        self._load_checkpoint()

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
        logger.info("  This may take a few minutes for first-time download...")
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

    def _load_checkpoint(self):
        """Load checkpoint if exists"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)

                self.completed_samples = set(checkpoint.get('completed_sample_ids', []))

                if self.completed_samples:
                    logger.info(f"📂 Checkpoint loaded: {len(self.completed_samples)} samples already completed")
                    logger.info(f"   Resuming from where we left off...")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                logger.info("Starting fresh...")

    def _save_checkpoint(self, sample_id: str):
        """Save checkpoint after processing a sample"""
        self.completed_samples.add(sample_id)

        checkpoint = {
            'completed_sample_ids': list(self.completed_samples),
            'total_samples': len(self.dataset),
            'progress_percent': len(self.completed_samples) / len(self.dataset) * 100,
            'last_updated': datetime.now().isoformat()
        }

        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _clear_gpu_cache(self):
        """Clear GPU cache to prevent memory issues"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _monitor_memory(self):
        """Monitor and log GPU memory usage"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            logger.debug(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def process_sample(self, sample: Dict, sample_idx: int) -> Optional[Dict]:
        """
        Process a single sample through FESTA pipeline

        Args:
            sample: Sample dictionary from dataset
            sample_idx: Sample index for logging

        Returns:
            Results dictionary or None if already processed
        """
        sample_id = str(sample.get('id', sample_idx))

        # Skip if already completed (checkpoint resume)
        if sample_id in self.completed_samples:
            logger.info(f"⏭️  Skipping sample {sample_idx + 1} (already completed)")
            return None

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing sample {sample_idx + 1}/{len(self.dataset)}")
        logger.info(f"Task: {sample['task']}, ID: {sample_id}")
        logger.info(f"{'='*60}")

        results = {
            'sample_id': sample_id,
            'task': sample['task'],
            'question': sample['question'],
            'ground_truth': sample['correct_answer']
        }

        try:
            # Get original prediction
            logger.info("1️⃣  Getting original prediction...")
            original_answer, original_probs = self.model.predict(
                sample['audio_path'],
                sample['question'],
                sample['options'],
                return_probs=True
            )
            results['prediction'] = original_answer
            logger.info(f"   Prediction: {original_answer}, Ground Truth: {sample['correct_answer']}")

            # Generate FES samples
            logger.info(f"2️⃣  Generating FES samples...")
            fes_samples = self.fes_generator.generate(
                sample['audio_path'],
                sample['question'],
                sample['task'],
                sample['options']
            )
            logger.info(f"   Generated {len(fes_samples)} FES samples")

            # Get predictions on FES samples
            logger.info(f"3️⃣  Getting FES predictions...")
            fes_predictions = []
            for fes_sample in tqdm(fes_samples, desc="   FES", leave=False):
                pred, _ = self.model.predict(
                    fes_sample['audio_path'],
                    fes_sample['question'],
                    fes_sample['options']
                )
                fes_predictions.append(pred)

            # Generate FCS samples
            logger.info(f"4️⃣  Generating FCS samples...")
            fcs_samples = self.fcs_generator.generate(
                sample['audio_path'],
                sample['question'],
                sample['task'],
                sample['options'],
                original_answer
            )
            logger.info(f"   Generated {len(fcs_samples)} FCS samples")

            # Get predictions on FCS samples
            logger.info(f"5️⃣  Getting FCS predictions...")
            fcs_predictions = []
            for fcs_sample in tqdm(fcs_samples, desc="   FCS", leave=False):
                pred, _ = self.model.predict(
                    fcs_sample['audio_path'],
                    fcs_sample['question'],
                    fcs_sample['options']
                )
                fcs_predictions.append(pred)

            # Compute FESTA uncertainty
            logger.info(f"6️⃣  Computing FESTA uncertainty...")
            festa_scores = self.festa_uncertainty.compute_festa(
                fes_predictions,
                fcs_predictions,
                original_answer
            )
            results['festa'] = festa_scores
            logger.info(f"   U_FES: {festa_scores['U_FES']:.4f}, "
                       f"U_FCS: {festa_scores['U_FCS']:.4f}, "
                       f"U_FESTA: {festa_scores['U_FESTA']:.4f}")

            # Compute baseline uncertainties if enabled
            if self.config['baselines']['output_entropy']['enabled']:
                logger.info(f"7️⃣  Computing Output Entropy baseline...")
                stochastic_probs = self.model.predict_with_sampling(
                    sample['audio_path'],
                    sample['question'],
                    sample['options'],
                    num_samples=self.config['baselines']['output_entropy']['num_samples'],
                    temperature=self.config['baselines']['output_entropy']['temperature']
                )
                stochastic_preds = [max(stochastic_probs, key=stochastic_probs.get)] * \
                                  self.config['baselines']['output_entropy']['num_samples']
                oe = self.baseline_uncertainty.output_entropy(stochastic_preds)
                results['OE'] = oe
                logger.info(f"   Output Entropy: {oe:.4f}")

            if self.config['baselines']['rephrase_uncertainty']['enabled']:
                logger.info(f"8️⃣  Computing Rephrase Uncertainty baseline...")
                rephrases = self.aug_generator.generate_text_rephrases(
                    sample['question'],
                    sample['task'],
                    n_rephrases=self.config['baselines']['rephrase_uncertainty']['num_rephrases']
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
                logger.info(f"   Rephrase Uncertainty: {ru:.4f}")

            logger.info(f"✅ Sample {sample_idx + 1} completed successfully")

            # Save checkpoint
            self._save_checkpoint(sample_id)

            # Clear GPU cache
            if self.config.get('hardware', {}).get('clear_cache', True):
                self._clear_gpu_cache()

            # Monitor memory
            if self.config.get('colab', {}).get('memory_monitor', True):
                self._monitor_memory()

            return results

        except Exception as e:
            logger.error(f"❌ Error processing sample {sample_idx}: {e}")
            logger.exception(e)
            return None

    def run_experiment(self):
        """Run the full FESTA experiment with checkpoint support"""

        logger.info("\n" + "="*80)
        logger.info("🚀 Starting FESTA Experiment (Colab Version)")
        logger.info("="*80)
        logger.info(f"📊 Dataset: {len(self.dataset)} samples")
        logger.info(f"📋 Tasks: {self.config['dataset']['tasks']}")
        logger.info(f"🤖 Model: {self.config['model']['name']}")
        logger.info(f"💾 Checkpoint: {self.checkpoint_file}")

        if self.completed_samples:
            remaining = len(self.dataset) - len(self.completed_samples)
            logger.info(f"📈 Progress: {len(self.completed_samples)}/{len(self.dataset)} completed, {remaining} remaining")

        logger.info("="*80)

        # Test mode: process only 1 sample
        if self.config.get('colab', {}).get('test_mode', False):
            logger.info("🧪 TEST MODE: Processing only 1 sample")
            samples_to_process = [self.dataset.data[0]]
        else:
            samples_to_process = self.dataset.data

        # Process all samples
        all_results = []
        for idx, sample in enumerate(samples_to_process):
            result = self.process_sample(sample, idx)

            if result is not None:
                all_results.append(result)

                # Save intermediate results after each sample
                if self.config['experiment']['save_intermediate']:
                    self._save_intermediate_results(all_results)
                    logger.info(f"💾 Intermediate results saved")

        # Compile results
        logger.info("\n📊 Compiling results...")
        self._compile_results(all_results)

        # Compute metrics
        logger.info("\n📈 Computing metrics...")
        self._compute_metrics()

        # Save final results
        logger.info("\n💾 Saving final results...")
        self._save_final_results()

        # Cleanup
        logger.info("\n🧹 Cleaning up...")
        self.fes_generator.cleanup_temp_files()
        self.fcs_generator.cleanup_temp_files()

        logger.info("\n" + "="*80)
        logger.info("🎉 FESTA Experiment Completed Successfully!")
        logger.info("="*80)
        logger.info(f"📁 Results saved to: {self.output_dir}")

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

        # Check if we have any results
        if len(predictions) == 0:
            logger.error("\n" + "="*60)
            logger.error("❌ ERROR: No samples were successfully processed!")
            logger.error("="*60)
            logger.error("All samples failed during processing.")
            logger.error("Please check the error messages above for details.")
            logger.error("\nCommon issues:")
            logger.error("  • Audio files not found (check paths in CSV files)")
            logger.error("  • Model loading failed")
            logger.error("  • Memory issues")
            logger.error("="*60)
            raise RuntimeError("No samples were successfully processed. Cannot compute metrics.")

        # Overall accuracy
        overall_accuracy = compute_accuracy(predictions, ground_truths)
        self.metrics['overall_accuracy'] = overall_accuracy

        logger.info(f"\n{'='*60}")
        logger.info(f"📊 RESULTS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Overall Accuracy: {overall_accuracy:.2%}")

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
        logger.info(f"\n{'='*60}")
        logger.info("📋 TASK-WISE PERFORMANCE")
        logger.info(f"{'='*60}")

        for method_name, uncertainties in self.results['uncertainties'].items():
            logger.info(f"\n{method_name.upper()}:")
            task_metrics = compute_task_wise_metrics(
                uncertainties, predictions, ground_truths, tasks
            )

            for task, metrics in task_metrics.items():
                logger.info(f"  {task}: AUROC={metrics['auroc']:.4f}, "
                          f"Acc={metrics['accuracy']:.2%}, "
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

        logger.info(f"\n{'='*60}")
        logger.info(f"📁 Results saved:")
        logger.info(f"  • Predictions: {predictions_file.name}")
        logger.info(f"  • Uncertainties: {uncertainties_file.name}")
        logger.info(f"  • Metrics: {metrics_file.name}")
        logger.info(f"{'='*60}")


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(description="Run FESTA experiment on Colab")
    parser.add_argument(
        '--config',
        type=str,
        default='config_colab.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only 1 sample'
    )
    args = parser.parse_args()

    # Override test mode if specified
    if args.test:
        logger.info("🧪 Test mode enabled via command line")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config.setdefault('colab', {})['test_mode'] = True
        with open(args.config, 'w') as f:
            yaml.safe_dump(config, f)

    # Run experiment
    experiment = ColabFESTAExperiment(config_path=args.config)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
