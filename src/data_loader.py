"""
Data loader for TREA (Temporal Reasoning Evaluation of Audio) dataset
Handles loading audio files, questions, and annotations for temporal reasoning tasks
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TREADataset:
    """
    TREA Dataset class for loading and managing temporal reasoning audio tasks

    Tasks:
    - Count: How many distinct sound sources are in the audio?
    - Order: Which sound event occurs first/second/after another?
    - Duration: Which sound has the longest/shortest duration?
    """

    def __init__(
        self,
        data_dir: str,
        tasks: List[str] = ["count", "order", "duration"],
        samples_per_task: Optional[int] = None,
        random_seed: int = 42,
        use_metadata: bool = False
    ):
        """
        Args:
            data_dir: Path to TREA_dataset directory
            tasks: List of tasks to load (count, order, duration)
            samples_per_task: Number of samples to load per task (None = all)
            random_seed: Random seed for sampling
            use_metadata: Whether to load metadata files (with event information)
        """
        self.data_dir = Path(data_dir)
        self.tasks = tasks
        self.samples_per_task = samples_per_task
        self.random_seed = random_seed
        self.use_metadata = use_metadata

        # Validate data directory
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        # Load all data
        self.data = self._load_data()

        logger.info(f"Loaded TREA dataset with {len(self.data)} samples")
        for task in tasks:
            task_samples = [d for d in self.data if d['task'] == task]
            logger.info(f"  {task}: {len(task_samples)} samples")

    def _load_data(self) -> List[Dict]:
        """Load data from CSV files for all tasks"""
        all_data = []

        for task in self.tasks:
            task_dir = self.data_dir / task

            # Choose CSV file (with or without metadata)
            if self.use_metadata:
                csv_file = task_dir / f"{task}_with_metadata.csv"
            else:
                csv_file = task_dir / f"{task}.csv"

            if not csv_file.exists():
                logger.warning(f"CSV file not found: {csv_file}")
                continue

            # Load CSV
            df = pd.read_csv(csv_file)

            # Sample if requested
            if self.samples_per_task is not None and self.samples_per_task < len(df):
                df = df.sample(
                    n=self.samples_per_task,
                    random_state=self.random_seed
                ).reset_index(drop=True)

            # Convert to list of dictionaries
            for idx, row in df.iterrows():
                sample = {
                    'task': task,
                    'id': row['id'],
                    'question': row['question'],
                    'audio_path': str(self.data_dir / row['audio_path']),
                    'options': {
                        'A': row['optionA'],
                        'B': row['optionB'],
                        'C': row['optionC'],
                        'D': row['optionD']
                    },
                    'correct_answer': row['correct'],
                    'original_index': idx
                }

                # Add metadata if available
                if self.use_metadata:
                    metadata_cols = [col for col in row.index if col not in
                                    ['id', 'question', 'audio_path', 'optionA',
                                     'optionB', 'optionC', 'optionD', 'correct']]
                    sample['metadata'] = {col: row[col] for col in metadata_cols}

                all_data.append(sample)

        return all_data

    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample by index"""
        return self.data[idx]

    def get_by_task(self, task: str) -> List[Dict]:
        """Get all samples for a specific task"""
        return [d for d in self.data if d['task'] == task]

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self.data),
            'tasks': {}
        }

        for task in self.tasks:
            task_data = self.get_by_task(task)
            stats['tasks'][task] = {
                'count': len(task_data),
                'unique_questions': len(set(d['question'] for d in task_data))
            }

        return stats

    def format_mcq_prompt(self, sample: Dict, include_options: bool = True) -> str:
        """
        Format a sample as a multiple-choice question prompt

        Args:
            sample: Dictionary containing question and options
            include_options: Whether to include answer options

        Returns:
            Formatted prompt string
        """
        prompt = f"{sample['question']}\n"

        if include_options:
            prompt += "\n"
            for key in ['A', 'B', 'C', 'D']:
                prompt += f"({key}) {sample['options'][key]}\n"
            prompt += "\nAnswer with only the letter (A, B, C, or D):"

        return prompt


class TREADataLoader:
    """
    DataLoader wrapper for batching and iterating over TREA dataset
    """

    def __init__(
        self,
        dataset: TREADataset,
        batch_size: int = 1,
        shuffle: bool = False,
        random_seed: int = 42
    ):
        """
        Args:
            dataset: TREADataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            random_seed: Random seed for shuffling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed

        self.indices = list(range(len(dataset)))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        """Return number of batches"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Iterate over batches"""
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield batch


def load_trea_dataset(
    data_dir: str = "TREA_dataset",
    tasks: List[str] = ["count", "order", "duration"],
    samples_per_task: int = 30,
    random_seed: int = 42
) -> TREADataset:
    """
    Convenience function to load TREA dataset

    Args:
        data_dir: Path to TREA_dataset directory
        tasks: List of tasks to load
        samples_per_task: Number of samples per task
        random_seed: Random seed for sampling

    Returns:
        TREADataset instance
    """
    dataset = TREADataset(
        data_dir=data_dir,
        tasks=tasks,
        samples_per_task=samples_per_task,
        random_seed=random_seed
    )

    logger.info("\nDataset Statistics:")
    stats = dataset.get_statistics()
    logger.info(f"Total samples: {stats['total_samples']}")
    for task, task_stats in stats['tasks'].items():
        logger.info(f"  {task}: {task_stats['count']} samples, "
                   f"{task_stats['unique_questions']} unique questions")

    return dataset


if __name__ == "__main__":
    # Test the data loader
    print("Testing TREA Data Loader...")

    # Load dataset with 30 samples per task
    dataset = load_trea_dataset(
        data_dir="TREA_dataset",
        samples_per_task=30,
        random_seed=42
    )

    # Print sample from each task
    for task in ["count", "order", "duration"]:
        task_samples = dataset.get_by_task(task)
        if task_samples:
            sample = task_samples[0]
            print(f"\n{'='*60}")
            print(f"Task: {task.upper()}")
            print(f"{'='*60}")
            print(f"ID: {sample['id']}")
            print(f"Audio: {sample['audio_path']}")
            print(f"\n{dataset.format_mcq_prompt(sample)}")
            print(f"Correct Answer: {sample['correct_answer']}")

    # Test dataloader
    print(f"\n{'='*60}")
    print("Testing DataLoader...")
    print(f"{'='*60}")

    dataloader = TREADataLoader(dataset, batch_size=5, shuffle=False)
    print(f"Number of batches: {len(dataloader)}")

    # Get first batch
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}: {len(batch)} samples")
        for sample in batch:
            print(f"  - {sample['task']}: {sample['id']}")
        if batch_idx == 0:  # Only show first batch
            break

    print("\nData loader test completed successfully!")
