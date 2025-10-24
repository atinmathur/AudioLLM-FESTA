"""
FCS (Functional Complementary Sampling) Generator
Generates task-equivalent but functionally divergent transformations
These transformations should change model predictions in ideal models

Transformations:
- Count: Add/remove sound events
- Order: Swap event positions
- Duration: Replace longest/shortest events
- Text: Reverse temporal/spatial relationships
"""

import numpy as np
import os
import tempfile
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
import random

from .utils import AudioProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FCSGenerator:
    """Generator for Functional Complementary Samples (FCS)"""

    def __init__(
        self,
        n_audio_samples: int = 15,
        n_text_samples: int = 4,
        sr: int = 16000,
        temp_dir: Optional[str] = None,
        synthetic_silence_dir: Optional[str] = None
    ):
        """
        Initialize FCS Generator

        Args:
            n_audio_samples: Number of audio transformations to generate
            n_text_samples: Number of text complements to generate
            sr: Sample rate for audio
            temp_dir: Directory for temporary files
            synthetic_silence_dir: Directory containing synthetic silence files
        """
        self.n_audio_samples = n_audio_samples
        self.n_text_samples = n_text_samples
        self.sr = sr
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.synthetic_silence_dir = synthetic_silence_dir or "TREA_dataset/synthetic_silences"
        self.audio_processor = AudioProcessor()

        # Create temp directory if it doesn't exist
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        # Load synthetic silence files for event addition
        self.synthetic_events = self._load_synthetic_events()

        logger.info(f"FCS Generator initialized:")
        logger.info(f"  Audio samples: {n_audio_samples}")
        logger.info(f"  Text samples: {n_text_samples}")
        logger.info(f"  Total FCS samples: {n_audio_samples * n_text_samples}")
        logger.info(f"  Synthetic events available: {len(self.synthetic_events)}")

    def _load_synthetic_events(self) -> List[str]:
        """Load synthetic event audio files"""
        synthetic_dir = Path(self.synthetic_silence_dir)
        if not synthetic_dir.exists():
            logger.warning(f"Synthetic silence directory not found: {self.synthetic_silence_dir}")
            return []

        # Load all .wav files from synthetic silence directory
        synthetic_files = list(synthetic_dir.glob("*.wav"))
        return [str(f) for f in synthetic_files]

    def generate(
        self,
        audio_path: str,
        question: str,
        task: str,
        options: Dict[str, str],
        original_answer: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate FCS samples for a given input

        Args:
            audio_path: Path to original audio
            question: Original question text
            task: Task type ('count', 'order', 'duration')
            options: Answer options
            original_answer: Original predicted answer (for text complementation)

        Returns:
            List of FCS samples with audio paths and complemented questions
        """
        # Generate audio transformations (task-specific)
        audio_samples = self._generate_audio_fcs(audio_path, task)

        # Generate text complements
        text_samples = self._generate_text_fcs(question, task, original_answer)

        # Combine variations
        fcs_samples = []
        for i, audio_sample in enumerate(audio_samples):
            for j, text_sample in enumerate(text_samples):
                fcs_samples.append({
                    'audio_path': audio_sample,
                    'question': text_sample,
                    'options': options,
                    'fcs_id': f"fcs_{i}_{j}"
                })

        logger.debug(f"Generated {len(fcs_samples)} FCS samples")
        return fcs_samples

    def _generate_audio_fcs(self, audio_path: str, task: str) -> List[str]:
        """
        Generate complementary audio transformations (task-specific)

        Args:
            audio_path: Path to original audio
            task: Task type

        Returns:
            List of paths to transformed audio files
        """
        # Load original audio
        audio, sr = self.audio_processor.load_audio(audio_path, sr=self.sr)
        audio_samples = []

        # Detect events in audio
        events = self.audio_processor.extract_events(audio, sr, top_db=20)

        if task == "count":
            # For count task: Add new events
            audio_samples.extend(self._fcs_count(audio, sr, events))

        elif task == "order":
            # For order task: Swap event positions
            audio_samples.extend(self._fcs_order(audio, sr, events))

        elif task == "duration":
            # For duration task: Replace longest/shortest events
            audio_samples.extend(self._fcs_duration(audio, sr, events))

        # Limit to n_audio_samples
        audio_samples = audio_samples[:self.n_audio_samples]

        # If we don't have enough samples, pad with original
        while len(audio_samples) < self.n_audio_samples:
            audio_samples.append(audio_path)

        return audio_samples

    def _fcs_count(self, audio: np.ndarray, sr: int, events: List[Tuple[int, int]]) -> List[str]:
        """
        Generate FCS for count task by adding new events

        Args:
            audio: Original audio
            sr: Sample rate
            events: Detected event intervals

        Returns:
            List of transformed audio paths
        """
        transformed_samples = []

        if not self.synthetic_events:
            logger.warning("No synthetic events available for count FCS")
            return transformed_samples

        # Add events at different positions
        positions = ["start", "end", "middle"]

        for idx, position in enumerate(positions * (self.n_audio_samples // 3 + 1)):
            try:
                # Randomly select a synthetic event
                synthetic_file = random.choice(self.synthetic_events)
                new_event, _ = self.audio_processor.load_audio(synthetic_file, sr=sr)

                # Trim the new event to reasonable length
                max_event_length = min(len(new_event), sr * 2)  # Max 2 seconds
                new_event = new_event[:max_event_length]

                # Add the new event
                if position == "start":
                    transformed = np.concatenate([new_event, audio])
                elif position == "end":
                    transformed = np.concatenate([audio, new_event])
                else:  # middle
                    mid_point = len(audio) // 2
                    transformed = np.concatenate([audio[:mid_point], new_event, audio[mid_point:]])

                # Save transformed audio
                temp_filename = f"fcs_count_add_{position}_{idx}.wav"
                temp_path = os.path.join(self.temp_dir, temp_filename)
                self.audio_processor.save_audio(transformed, temp_path, sr)
                transformed_samples.append(temp_path)

            except Exception as e:
                logger.warning(f"Failed to add event at {position}: {e}")

        return transformed_samples

    def _fcs_order(self, audio: np.ndarray, sr: int, events: List[Tuple[int, int]]) -> List[str]:
        """
        Generate FCS for order task by swapping events

        Args:
            audio: Original audio
            sr: Sample rate
            events: Detected event intervals

        Returns:
            List of transformed audio paths
        """
        transformed_samples = []

        if len(events) < 2:
            logger.warning("Not enough events for order FCS")
            return transformed_samples

        # Generate different swap patterns
        for idx in range(self.n_audio_samples):
            try:
                # Select two events to swap
                if len(events) >= 2:
                    # Swap adjacent events or random pairs
                    if idx % 2 == 0 and len(events) >= 2:
                        # Swap first two events
                        swap_idx = (0, 1)
                    elif len(events) >= 3:
                        # Swap last two events
                        swap_idx = (len(events) - 2, len(events) - 1)
                    else:
                        swap_idx = (0, 1)

                    transformed = self.audio_processor.swap_audio_segments(
                        audio, events, swap_idx
                    )

                    # Save transformed audio
                    temp_filename = f"fcs_order_swap_{idx}.wav"
                    temp_path = os.path.join(self.temp_dir, temp_filename)
                    self.audio_processor.save_audio(transformed, temp_path, sr)
                    transformed_samples.append(temp_path)

            except Exception as e:
                logger.warning(f"Failed to swap events: {e}")

        return transformed_samples

    def _fcs_duration(self, audio: np.ndarray, sr: int, events: List[Tuple[int, int]]) -> List[str]:
        """
        Generate FCS for duration task by replacing longest/shortest events

        Args:
            audio: Original audio
            sr: Sample rate
            events: Detected event intervals

        Returns:
            List of transformed audio paths
        """
        transformed_samples = []

        if not events or not self.synthetic_events:
            logger.warning("Not enough events or synthetic events for duration FCS")
            return transformed_samples

        # Calculate event durations
        event_durations = [(end - start) for start, end in events]

        for idx in range(self.n_audio_samples):
            try:
                # Randomly select a synthetic replacement
                synthetic_file = random.choice(self.synthetic_events)
                replacement_event, _ = self.audio_processor.load_audio(synthetic_file, sr=sr)

                # Choose to replace longest or shortest event
                if idx % 2 == 0 and len(event_durations) > 0:
                    # Replace longest event
                    target_idx = np.argmax(event_durations)
                elif len(event_durations) > 0:
                    # Replace shortest event
                    target_idx = np.argmin(event_durations)
                else:
                    continue

                start, end = events[target_idx]

                # Adjust replacement length to match original duration
                target_length = end - start
                if len(replacement_event) > target_length:
                    replacement_event = replacement_event[:target_length]
                else:
                    # Pad with silence if too short
                    padding = np.zeros(target_length - len(replacement_event))
                    replacement_event = np.concatenate([replacement_event, padding])

                # Replace the event
                transformed = audio.copy()
                transformed[start:end] = replacement_event

                # Save transformed audio
                temp_filename = f"fcs_duration_replace_{idx}.wav"
                temp_path = os.path.join(self.temp_dir, temp_filename)
                self.audio_processor.save_audio(transformed, temp_path, sr)
                transformed_samples.append(temp_path)

            except Exception as e:
                logger.warning(f"Failed to replace event: {e}")

        return transformed_samples

    def _generate_text_fcs(self, question: str, task: str, original_answer: Optional[str] = None) -> List[str]:
        """
        Generate complementary text transformations
        Reverse the temporal/spatial relationship in questions

        Args:
            question: Original question
            task: Task type
            original_answer: Original predicted answer

        Returns:
            List of complemented questions
        """
        complements = []

        if task == "count":
            # For count, we can ask about different types of events
            # But keep the counting nature
            complements.append(question)  # Text complement less relevant for count

        elif task == "order":
            # Reverse temporal relationships
            q_lower = question.lower()
            if "first" in q_lower:
                complements.append(question.replace("first", "last"))
                complements.append(question.replace("first", "second"))
                complements.append(question.replace("occurs first", "occurs last"))
            elif "second" in q_lower:
                complements.append(question.replace("second", "first"))
                complements.append(question.replace("second", "last"))
                complements.append(question.replace("occurs second", "occurs first"))
            elif "after" in q_lower:
                complements.append(question.replace("after", "before"))
                complements.append(question.replace("occurs after", "occurs before"))
            else:
                complements.append(question)

        elif task == "duration":
            # Reverse duration comparisons
            q_lower = question.lower()
            if "longest" in q_lower:
                complements.append(question.replace("longest", "shortest"))
                complements.append(question.replace("longest time", "shortest time"))
            elif "shortest" in q_lower:
                complements.append(question.replace("shortest", "longest"))
                complements.append(question.replace("shortest time", "longest time"))
            else:
                complements.append(question)

        # Remove duplicates
        complements = list(dict.fromkeys(complements))

        # Ensure we have at least n_text_samples
        while len(complements) < self.n_text_samples:
            complements.append(question)

        return complements[:self.n_text_samples]

    def cleanup_temp_files(self):
        """Remove temporary audio files created during FCS generation"""
        try:
            temp_pattern = os.path.join(self.temp_dir, "fcs_*.wav")
            import glob
            temp_files = glob.glob(temp_pattern)
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
            logger.info(f"Cleaned up {len(temp_files)} temporary FCS files")
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")


if __name__ == "__main__":
    # Test FCS generator
    print("Testing FCS Generator...")

    # Initialize generator
    fcs_gen = FCSGenerator(
        n_audio_samples=5,
        n_text_samples=3,
        synthetic_silence_dir="TREA_dataset/synthetic_silences"
    )

    # Test with a sample
    sample_audio = "TREA_dataset/count/audios/0.wav"
    question = "How many distinct sound sources are in the audio?"
    task = "count"
    options = {'A': '0', 'B': '2', 'C': '3', 'D': '1'}

    if Path(sample_audio).exists():
        print(f"\nGenerating FCS samples for:")
        print(f"  Audio: {sample_audio}")
        print(f"  Question: {question}")
        print(f"  Task: {task}")

        # Generate FCS samples
        fcs_samples = fcs_gen.generate(audio_path=sample_audio, question=question, task=task, options=options)

        print(f"\nGenerated {len(fcs_samples)} FCS samples:")
        for i, sample in enumerate(fcs_samples[:5]):  # Show first 5
            print(f"\nSample {i+1}:")
            print(f"  ID: {sample['fcs_id']}")
            print(f"  Audio: {os.path.basename(sample['audio_path'])}")
            print(f"  Question: {sample['question'][:60]}...")

        # Test order task
        print("\n" + "="*60)
        print("Testing ORDER task FCS...")
        order_question = "Which sound event occurs first?"
        fcs_order = fcs_gen._generate_text_fcs(order_question, "order")
        print(f"Original: {order_question}")
        print("Complements:")
        for comp in fcs_order:
            print(f"  - {comp}")

        # Test duration task
        print("\n" + "="*60)
        print("Testing DURATION task FCS...")
        duration_question = "Which sound has the longest duration?"
        fcs_duration = fcs_gen._generate_text_fcs(duration_question, "duration")
        print(f"Original: {duration_question}")
        print("Complements:")
        for comp in fcs_duration:
            print(f"  - {comp}")

        # Cleanup
        fcs_gen.cleanup_temp_files()
        print("\nFCS generator test completed successfully!")
    else:
        print(f"Sample audio not found: {sample_audio}")
