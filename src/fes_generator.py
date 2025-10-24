"""
FES (Functional Equivalent Sampling) Generator
Generates task-preserving transformations that should not change model predictions

Transformations:
- Generic: silence addition, volume adjustment, noise
- Task-specific: preserves task objective while varying input
- Text: question paraphrasing
"""

import numpy as np
import os
import tempfile
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

from .utils import AudioProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FESGenerator:
    """Generator for Functional Equivalent Samples (FES)"""

    def __init__(
        self,
        n_audio_samples: int = 15,
        n_text_samples: int = 4,
        sr: int = 16000,
        temp_dir: Optional[str] = None
    ):
        """
        Initialize FES Generator

        Args:
            n_audio_samples: Number of audio transformations to generate
            n_text_samples: Number of text paraphrases to generate
            sr: Sample rate for audio
            temp_dir: Directory for temporary files
        """
        self.n_audio_samples = n_audio_samples
        self.n_text_samples = n_text_samples
        self.sr = sr
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.audio_processor = AudioProcessor()

        # Create temp directory if it doesn't exist
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"FES Generator initialized:")
        logger.info(f"  Audio samples: {n_audio_samples}")
        logger.info(f"  Text samples: {n_text_samples}")
        logger.info(f"  Total FES samples: {n_audio_samples * n_text_samples}")

    def generate(
        self,
        audio_path: str,
        question: str,
        task: str,
        options: Dict[str, str]
    ) -> List[Dict]:
        """
        Generate FES samples for a given input

        Args:
            audio_path: Path to original audio
            question: Original question text
            task: Task type ('count', 'order', 'duration')
            options: Answer options

        Returns:
            List of FES samples with audio paths and paraphrased questions
        """
        # Generate audio transformations
        audio_samples = self._generate_audio_fes(audio_path, task)

        # Generate text paraphrases
        text_samples = self._generate_text_fes(question, task)

        # Combine all variations
        fes_samples = []
        for i, audio_sample in enumerate(audio_samples):
            for j, text_sample in enumerate(text_samples):
                fes_samples.append({
                    'audio_path': audio_sample,
                    'question': text_sample,
                    'options': options,
                    'fes_id': f"fes_{i}_{j}"
                })

        logger.debug(f"Generated {len(fes_samples)} FES samples")
        return fes_samples

    def _generate_audio_fes(self, audio_path: str, task: str) -> List[str]:
        """
        Generate equivalent audio transformations

        Args:
            audio_path: Path to original audio
            task: Task type

        Returns:
            List of paths to transformed audio files
        """
        # Load original audio
        audio, sr = self.audio_processor.load_audio(audio_path, sr=self.sr)
        audio_samples = []

        # Generic transformations (applicable to all tasks)
        transformations = [
            # Original audio (identity transformation)
            ('original', lambda a: a),

            # Silence addition (between events)
            ('silence_0.1s', lambda a: self.audio_processor.add_silence(a, sr, 0.1, 'middle')),
            ('silence_0.2s', lambda a: self.audio_processor.add_silence(a, sr, 0.2, 'middle')),
            ('silence_0.3s', lambda a: self.audio_processor.add_silence(a, sr, 0.3, 'middle')),
            ('silence_0.5s', lambda a: self.audio_processor.add_silence(a, sr, 0.5, 'middle')),

            # Volume adjustments
            ('volume_+10%', lambda a: self.audio_processor.adjust_volume(a, 0.1)),
            ('volume_+20%', lambda a: self.audio_processor.adjust_volume(a, 0.2)),
            ('volume_-10%', lambda a: self.audio_processor.adjust_volume(a, -0.1)),
            ('volume_-20%', lambda a: self.audio_processor.adjust_volume(a, -0.2)),

            # Noise addition
            ('noise_snr30', lambda a: self.audio_processor.add_noise(a, snr_db=30)),
            ('noise_snr35', lambda a: self.audio_processor.add_noise(a, snr_db=35)),
            ('noise_snr40', lambda a: self.audio_processor.add_noise(a, snr_db=40)),

            # Normalization
            ('normalized', lambda a: self.audio_processor.normalize_audio(a)),

            # Combined transformations
            ('silence+volume', lambda a: self.audio_processor.adjust_volume(
                self.audio_processor.add_silence(a, sr, 0.2, 'middle'), 0.1)),
            ('silence+noise', lambda a: self.audio_processor.add_noise(
                self.audio_processor.add_silence(a, sr, 0.2, 'middle'), snr_db=35)),
        ]

        # Select transformations (limit to n_audio_samples)
        selected_transforms = transformations[:self.n_audio_samples]

        for name, transform_func in selected_transforms:
            try:
                # Apply transformation
                transformed_audio = transform_func(audio)

                # Save to temporary file
                temp_filename = f"fes_{task}_{name}_{os.path.basename(audio_path)}"
                temp_path = os.path.join(self.temp_dir, temp_filename)
                self.audio_processor.save_audio(transformed_audio, temp_path, sr)

                audio_samples.append(temp_path)
            except Exception as e:
                logger.warning(f"Failed to apply transformation '{name}': {e}")
                # Use original audio as fallback
                audio_samples.append(audio_path)

        return audio_samples

    def _generate_text_fes(self, question: str, task: str) -> List[str]:
        """
        Generate equivalent text paraphrases

        Args:
            question: Original question
            task: Task type

        Returns:
            List of paraphrased questions
        """
        # For now, use manual paraphrases
        # In production, could use LLM for paraphrasing

        paraphrases = [question]  # Include original

        # Task-specific paraphrases
        if task == "count":
            if "how many" in question.lower():
                paraphrases.extend([
                    question.replace("How many", "What is the number of"),
                    question.replace("How many", "Count the"),
                    question.replace("how many", "Identify the count of")
                ])
            elif "number of" in question.lower():
                paraphrases.extend([
                    question.replace("number of", "count of"),
                    question.replace("What is the number of", "How many"),
                    question.replace("number of", "total of")
                ])

        elif task == "order":
            if "first" in question.lower():
                paraphrases.extend([
                    question.replace("first", "initial"),
                    question.replace("occurs first", "happens first"),
                    question.replace("occurs first", "comes first")
                ])
            elif "second" in question.lower():
                paraphrases.extend([
                    question.replace("second", "next"),
                    question.replace("occurs second", "happens second"),
                    question.replace("occurs second", "comes second")
                ])
            elif "after" in question.lower():
                paraphrases.extend([
                    question.replace("occurs after", "happens after"),
                    question.replace("occurs after", "comes after"),
                    question.replace("after", "following")
                ])

        elif task == "duration":
            if "longest" in question.lower():
                paraphrases.extend([
                    question.replace("longest", "maximum"),
                    question.replace("longest time", "most time"),
                    question.replace("longest", "greatest duration")
                ])
            elif "shortest" in question.lower():
                paraphrases.extend([
                    question.replace("shortest", "minimum"),
                    question.replace("shortest time", "least time"),
                    question.replace("shortest", "smallest duration")
                ])

        # Remove duplicates and limit to n_text_samples
        paraphrases = list(dict.fromkeys(paraphrases))[:self.n_text_samples]

        # If we don't have enough paraphrases, pad with the original
        while len(paraphrases) < self.n_text_samples:
            paraphrases.append(question)

        return paraphrases

    def cleanup_temp_files(self):
        """Remove temporary audio files created during FES generation"""
        try:
            temp_pattern = os.path.join(self.temp_dir, "fes_*.wav")
            import glob
            temp_files = glob.glob(temp_pattern)
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
            logger.info(f"Cleaned up {len(temp_files)} temporary FES files")
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")


class AdvancedFESGenerator(FESGenerator):
    """
    Advanced FES generator with LLM-based paraphrasing
    Requires API access to paraphrasing service
    """

    def __init__(
        self,
        n_audio_samples: int = 15,
        n_text_samples: int = 4,
        sr: int = 16000,
        temp_dir: Optional[str] = None,
        paraphrase_model: Optional[str] = None
    ):
        super().__init__(n_audio_samples, n_text_samples, sr, temp_dir)
        self.paraphrase_model = paraphrase_model

    def _generate_text_fes_with_llm(self, question: str, task: str) -> List[str]:
        """
        Generate paraphrases using LLM

        Args:
            question: Original question
            task: Task type

        Returns:
            List of paraphrased questions
        """
        # Placeholder for LLM-based paraphrasing
        # Could integrate with OpenAI API or local model

        prompt = f"""
        Generate {self.n_text_samples - 1} paraphrases of the following question.
        Maintain the same meaning and task objective.

        Original question: {question}

        Paraphrases:
        """

        # This would call an LLM API in production
        # For now, fall back to manual paraphrases
        return self._generate_text_fes(question, task)


if __name__ == "__main__":
    # Test FES generator
    print("Testing FES Generator...")

    from typing import Optional

    # Initialize generator
    fes_gen = FESGenerator(
        n_audio_samples=5,
        n_text_samples=3
    )

    # Test with a sample
    sample_audio = "TREA_dataset/count/audios/0.wav"
    question = "How many distinct sound sources are in the audio?"
    task = "count"
    options = {'A': '0', 'B': '2', 'C': '3', 'D': '1'}

    if Path(sample_audio).exists():
        print(f"\nGenerating FES samples for:")
        print(f"  Audio: {sample_audio}")
        print(f"  Question: {question}")
        print(f"  Task: {task}")

        # Generate FES samples
        fes_samples = fes_gen.generate(audio_path=sample_audio, question=question, task=task, options=options)

        print(f"\nGenerated {len(fes_samples)} FES samples:")
        for i, sample in enumerate(fes_samples[:5]):  # Show first 5
            print(f"\nSample {i+1}:")
            print(f"  ID: {sample['fes_id']}")
            print(f"  Audio: {os.path.basename(sample['audio_path'])}")
            print(f"  Question: {sample['question'][:60]}...")

        # Cleanup
        fes_gen.cleanup_temp_files()
        print("\nFES generator test completed successfully!")
    else:
        print(f"Sample audio not found: {sample_audio}")
