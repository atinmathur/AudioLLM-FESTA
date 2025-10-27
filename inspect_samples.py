#!/usr/bin/env python3
"""
FES/FCS Sample Inspector
Generates and displays FES and FCS samples for all three tasks (COUNT, ORDER, DURATION)
"""

import sys
import os
import pandas as pd
import shutil
from pathlib import Path
from typing import Dict, List

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.fes_generator import FESGenerator
from src.fcs_generator import FCSGenerator


class SampleInspector:
    """Inspects FES and FCS samples generated for different tasks"""

    def __init__(self, dataset_dir: str = "mini-TREA_dataset",
                 save_perturbations: bool = True,
                 perturbations_dir: str = "perturbations"):
        self.dataset_dir = dataset_dir
        self.tasks = ["count", "order", "duration"]
        self.save_perturbations = save_perturbations
        self.perturbations_dir = perturbations_dir

        # Metadata tracking
        self.perturbations_metadata = []

        # Create perturbations directory structure if saving
        if self.save_perturbations:
            self._create_perturbations_dirs()

        # Initialize generators
        print("Initializing FES and FCS Generators...")
        self.fes_gen = FESGenerator(
            n_audio_samples=15,
            n_text_samples=4,
            sr=16000
        )

        self.fcs_gen = FCSGenerator(
            n_audio_samples=15,
            n_text_samples=4,
            sr=16000,
            synthetic_silence_dir=os.path.join(dataset_dir, "synthetic_silences")
        )
        print()

    def _create_perturbations_dirs(self):
        """Create directory structure for saving perturbations"""
        base_dir = Path(self.perturbations_dir)

        for task in self.tasks:
            for ptype in ['fes', 'fcs']:
                dir_path = base_dir / task / ptype
                dir_path.mkdir(parents=True, exist_ok=True)

        print(f"Created perturbations directory: {self.perturbations_dir}/")
        print(f"  Structure: <task>/fes/ and <task>/fcs/ for each task")

    def _save_perturbation(self, temp_audio_path: str, sample: Dict,
                          perturbation_type: str, perturbation_id: str,
                          transformation_name: str, transformed_question: str) -> str:
        """
        Save a perturbation file and track its metadata

        Args:
            temp_audio_path: Path to temporary audio file
            sample: Original sample dictionary
            perturbation_type: 'fes' or 'fcs'
            perturbation_id: Unique ID for this perturbation
            transformation_name: Name of the transformation applied
            transformed_question: The transformed question text

        Returns:
            Path to saved perturbation file
        """
        if not self.save_perturbations:
            return temp_audio_path

        # Create destination path
        filename = f"{sample['id']}_{transformation_name}_{perturbation_id}.wav"
        dest_dir = Path(self.perturbations_dir) / sample['task'] / perturbation_type
        dest_path = dest_dir / filename

        # Copy file
        try:
            shutil.copy2(temp_audio_path, dest_path)
        except Exception as e:
            print(f"Warning: Failed to copy {temp_audio_path}: {e}")
            return temp_audio_path

        # Track metadata
        metadata = {
            'original_sample_id': sample['id'],
            'task': sample['task'],
            'perturbation_type': perturbation_type,
            'perturbation_id': perturbation_id,
            'transformation_name': transformation_name,
            'audio_path': str(dest_path),
            'original_audio_path': sample['audio_path'],
            'original_question': sample['question'],
            'transformed_question': transformed_question
        }
        self.perturbations_metadata.append(metadata)

        return str(dest_path)

    def load_sample_from_task(self, task: str) -> Dict:
        """Load one sample from the specified task"""
        csv_path = os.path.join(self.dataset_dir, task, f"{task}_with_metadata.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)

        # Take the first sample
        sample = df.iloc[0]

        # Prepare options dictionary
        options = {
            'A': str(sample['optionA']),
            'B': str(sample['optionB']),
            'C': str(sample['optionC']),
            'D': str(sample['optionD'])
        }

        return {
            'task': task,
            'id': sample['id'],
            'audio_path': sample['audio_path'],
            'question': sample['question'],
            'options': options,
            'correct': sample['correct']
        }

    def print_section_header(self, title: str):
        """Print a formatted section header"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def print_sample_info(self, sample: Dict):
        """Print information about a sample"""
        print(f"\nTask: {sample['task'].upper()}")
        print(f"Sample ID: {sample['id']}")
        print(f"Audio Path: {sample['audio_path']}")
        print(f"Question: {sample['question']}")
        print(f"Options: {sample['options']}")
        print(f"Correct Answer: {sample['correct']}")

    def inspect_fes_samples(self, sample: Dict):
        """Generate and inspect FES samples"""
        self.print_section_header(f"FES Samples for {sample['task'].upper()} Task")

        # Generate FES samples
        print("\nGenerating FES (Functional Equivalent Samples)...")
        fes_samples = self.fes_gen.generate(
            audio_path=sample['audio_path'],
            question=sample['question'],
            task=sample['task'],
            options=sample['options']
        )

        print(f"\nTotal FES samples generated: {len(fes_samples)}")

        # Save FES samples if enabled
        if self.save_perturbations:
            print(f"Saving FES perturbations to {self.perturbations_dir}/{sample['task']}/fes/...")
            for sample_data in fes_samples:
                # Extract transformation name
                basename = os.path.basename(sample_data['audio_path'])
                if f"{sample['task']}_" in basename:
                    transformation_name = basename.split(f"{sample['task']}_")[1].replace(f"_{sample['id']}.wav", "")
                else:
                    transformation_name = "original"

                # Save perturbation
                self._save_perturbation(
                    temp_audio_path=sample_data['audio_path'],
                    sample=sample,
                    perturbation_type='fes',
                    perturbation_id=sample_data['fes_id'],
                    transformation_name=transformation_name,
                    transformed_question=sample_data['question']
                )

        # Get unique audio transformations
        unique_audio_transforms = list(set([
            os.path.basename(s['audio_path']).split(f"{sample['task']}_")[1].replace(f"_{sample['id']}.wav", "")
            if f"{sample['task']}_" in os.path.basename(s['audio_path'])
            else "original"
            for s in fes_samples
        ]))

        # Get unique text paraphrases
        unique_questions = list(set([s['question'] for s in fes_samples]))

        print(f"\nAudio transformations ({len(unique_audio_transforms)}):")
        for i, transform in enumerate(sorted(unique_audio_transforms), 1):
            print(f"  {i}. {transform}")

        print(f"\nText paraphrases ({len(unique_questions)}):")
        for i, question in enumerate(unique_questions, 1):
            print(f"  {i}. {question}")

        # Show sample combinations
        print(f"\nSample FES combinations (showing first 5 of {len(fes_samples)}):")
        for i, sample_data in enumerate(fes_samples[:5], 1):
            audio_name = os.path.basename(sample_data['audio_path'])
            print(f"\n  {i}. FES ID: {sample_data['fes_id']}")
            print(f"     Audio: {audio_name}")
            print(f"     Question: {sample_data['question'][:70]}...")

        return fes_samples

    def inspect_fcs_samples(self, sample: Dict):
        """Generate and inspect FCS samples"""
        self.print_section_header(f"FCS Samples for {sample['task'].upper()} Task")

        # Generate FCS samples
        print("\nGenerating FCS (Functional Complementary Samples)...")
        print(f"Task-specific transformations for {sample['task'].upper()}:")

        if sample['task'] == 'count':
            print("  - Adding synthetic sound events at different positions (start/middle/end)")
        elif sample['task'] == 'order':
            print("  - Swapping temporal order of sound events")
        elif sample['task'] == 'duration':
            print("  - Replacing longest/shortest sound events")

        fcs_samples = self.fcs_gen.generate(
            audio_path=sample['audio_path'],
            question=sample['question'],
            task=sample['task'],
            options=sample['options'],
            original_answer=sample['correct']
        )

        print(f"\nTotal FCS samples generated: {len(fcs_samples)}")

        # Save FCS samples if enabled
        if self.save_perturbations:
            print(f"Saving FCS perturbations to {self.perturbations_dir}/{sample['task']}/fcs/...")
            for sample_data in fcs_samples:
                # Extract transformation name
                basename = os.path.basename(sample_data['audio_path'])
                if 'fcs_' in basename and basename.endswith('.wav'):
                    transformation_name = basename.replace('.wav', '')
                else:
                    transformation_name = "original"

                # Save perturbation
                self._save_perturbation(
                    temp_audio_path=sample_data['audio_path'],
                    sample=sample,
                    perturbation_type='fcs',
                    perturbation_id=sample_data['fcs_id'],
                    transformation_name=transformation_name,
                    transformed_question=sample_data['question']
                )

        # Get unique audio transformations
        unique_audio_transforms = []
        for s in fcs_samples:
            basename = os.path.basename(s['audio_path'])
            if basename.endswith('.wav'):
                if 'fcs_' in basename:
                    transform = basename.replace('.wav', '')
                    unique_audio_transforms.append(transform)
                else:
                    unique_audio_transforms.append('original')

        unique_audio_transforms = list(set(unique_audio_transforms))

        # Get unique text complements
        unique_questions = list(set([s['question'] for s in fcs_samples]))

        print(f"\nAudio transformations ({len(unique_audio_transforms)}):")
        for i, transform in enumerate(sorted(unique_audio_transforms), 1):
            print(f"  {i}. {transform}")

        print(f"\nText complements ({len(unique_questions)}):")
        original_question = sample['question']
        for i, question in enumerate(unique_questions, 1):
            marker = " [ORIGINAL]" if question == original_question else ""
            print(f"  {i}. {question}{marker}")

        # Show sample combinations
        print(f"\nSample FCS combinations (showing first 5 of {len(fcs_samples)}):")
        for i, sample_data in enumerate(fcs_samples[:5], 1):
            audio_name = os.path.basename(sample_data['audio_path'])
            print(f"\n  {i}. FCS ID: {sample_data['fcs_id']}")
            print(f"     Audio: {audio_name}")
            print(f"     Question: {sample_data['question'][:70]}...")

        return fcs_samples

    def run_inspection(self):
        """Run inspection for all tasks"""
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "FES/FCS Sample Inspector" + " " * 34 + "║")
        print("╚" + "═" * 78 + "╝")

        all_results = {}

        for task in self.tasks:
            try:
                print(f"\n\n{'█' * 80}")
                print(f"  Processing {task.upper()} Task")
                print(f"{'█' * 80}")

                # Load sample
                sample = self.load_sample_from_task(task)
                self.print_sample_info(sample)

                # Inspect FES
                fes_samples = self.inspect_fes_samples(sample)

                # Inspect FCS
                fcs_samples = self.inspect_fcs_samples(sample)

                all_results[task] = {
                    'sample': sample,
                    'fes_count': len(fes_samples),
                    'fcs_count': len(fcs_samples)
                }

            except Exception as e:
                print(f"\nError processing {task} task: {e}")
                import traceback
                traceback.print_exc()

        # Print summary
        self.print_section_header("Summary")
        print("\nSamples generated per task:")
        print(f"\n{'Task':<12} {'FES Samples':<15} {'FCS Samples':<15} {'Total':<10}")
        print("-" * 55)

        total_fes = 0
        total_fcs = 0

        for task in self.tasks:
            if task in all_results:
                fes_count = all_results[task]['fes_count']
                fcs_count = all_results[task]['fcs_count']
                total = fes_count + fcs_count

                print(f"{task.upper():<12} {fes_count:<15} {fcs_count:<15} {total:<10}")

                total_fes += fes_count
                total_fcs += fcs_count

        print("-" * 55)
        print(f"{'TOTAL':<12} {total_fes:<15} {total_fcs:<15} {total_fes + total_fcs:<10}")

        print("\n" + "=" * 80)
        print("Inspection complete!")
        print("=" * 80 + "\n")

        # Save metadata CSV
        if self.save_perturbations and self.perturbations_metadata:
            self._save_metadata_csv()

        # Cleanup
        print("Cleaning up temporary files...")
        self.fes_gen.cleanup_temp_files()
        self.fcs_gen.cleanup_temp_files()
        print("Done!\n")

    def _save_metadata_csv(self):
        """Save perturbations metadata to CSV file"""
        csv_path = Path(self.perturbations_dir) / "perturbations_metadata.csv"

        # Convert to DataFrame
        df = pd.DataFrame(self.perturbations_metadata)

        # Reorder columns for readability
        column_order = [
            'original_sample_id',
            'task',
            'perturbation_type',
            'perturbation_id',
            'transformation_name',
            'audio_path',
            'original_audio_path',
            'original_question',
            'transformed_question'
        ]
        df = df[column_order]

        # Save to CSV
        df.to_csv(csv_path, index=False)

        print(f"\n{'='*80}")
        print(f"💾 Saved perturbations metadata:")
        print(f"   File: {csv_path}")
        print(f"   Total perturbations: {len(df)}")
        print(f"   FES perturbations: {len(df[df['perturbation_type'] == 'fes'])}")
        print(f"   FCS perturbations: {len(df[df['perturbation_type'] == 'fcs'])}")
        print(f"{'='*80}")


def main():
    """Main entry point"""
    inspector = SampleInspector()
    inspector.run_inspection()


if __name__ == "__main__":
    main()
