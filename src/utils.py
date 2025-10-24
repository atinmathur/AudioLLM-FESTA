"""
Utility functions for audio processing and manipulation
Used for FES and FCS transformations
"""

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio processing utilities for FESTA transformations"""

    @staticmethod
    def load_audio(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load audio file

        Args:
            audio_path: Path to audio file
            sr: Target sample rate

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio, sample_rate = librosa.load(audio_path, sr=sr)
        return audio, sample_rate

    @staticmethod
    def save_audio(audio: np.ndarray, output_path: str, sr: int = 16000):
        """
        Save audio to file

        Args:
            audio: Audio data array
            output_path: Output file path
            sr: Sample rate
        """
        sf.write(output_path, audio, sr)

    @staticmethod
    def add_silence(
        audio: np.ndarray,
        sr: int,
        duration: float = 0.3,
        position: str = "middle"
    ) -> np.ndarray:
        """
        Add silence to audio

        Args:
            audio: Audio data
            sr: Sample rate
            duration: Duration of silence in seconds
            position: Where to add silence ('start', 'middle', 'end')

        Returns:
            Modified audio
        """
        silence = np.zeros(int(duration * sr))

        if position == "start":
            return np.concatenate([silence, audio])
        elif position == "end":
            return np.concatenate([audio, silence])
        elif position == "middle":
            mid_point = len(audio) // 2
            return np.concatenate([audio[:mid_point], silence, audio[mid_point:]])
        else:
            raise ValueError(f"Unknown position: {position}")

    @staticmethod
    def adjust_volume(
        audio: np.ndarray,
        gain: float = 0.0
    ) -> np.ndarray:
        """
        Adjust audio volume

        Args:
            audio: Audio data
            gain: Gain in amplitude (e.g., 0.1 = +10%, -0.1 = -10%)

        Returns:
            Modified audio
        """
        return audio * (1.0 + gain)

    @staticmethod
    def add_noise(
        audio: np.ndarray,
        snr_db: float = 30.0
    ) -> np.ndarray:
        """
        Add Gaussian noise to audio

        Args:
            audio: Audio data
            snr_db: Signal-to-noise ratio in decibels

        Returns:
            Modified audio with noise
        """
        # Calculate signal power
        signal_power = np.mean(audio ** 2)

        # Calculate noise power based on SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))

        return audio + noise

    @staticmethod
    def time_stretch(
        audio: np.ndarray,
        rate: float = 1.0
    ) -> np.ndarray:
        """
        Time-stretch audio without changing pitch

        Args:
            audio: Audio data
            rate: Stretch rate (>1 = faster, <1 = slower)

        Returns:
            Time-stretched audio
        """
        return librosa.effects.time_stretch(audio, rate=rate)

    @staticmethod
    def pitch_shift(
        audio: np.ndarray,
        sr: int,
        n_steps: float = 0.0
    ) -> np.ndarray:
        """
        Shift audio pitch

        Args:
            audio: Audio data
            sr: Sample rate
            n_steps: Number of semitones to shift

        Returns:
            Pitch-shifted audio
        """
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    @staticmethod
    def concatenate_audios(
        audio_list: List[np.ndarray],
        silence_between: float = 0.0,
        sr: int = 16000
    ) -> np.ndarray:
        """
        Concatenate multiple audio segments

        Args:
            audio_list: List of audio arrays
            silence_between: Duration of silence between segments (seconds)
            sr: Sample rate

        Returns:
            Concatenated audio
        """
        if silence_between > 0:
            silence = np.zeros(int(silence_between * sr))
            result = []
            for i, audio in enumerate(audio_list):
                result.append(audio)
                if i < len(audio_list) - 1:
                    result.append(silence)
            return np.concatenate(result)
        else:
            return np.concatenate(audio_list)

    @staticmethod
    def extract_events(
        audio: np.ndarray,
        sr: int,
        top_db: float = 20
    ) -> List[Tuple[int, int]]:
        """
        Detect sound events in audio using energy-based detection

        Args:
            audio: Audio data
            sr: Sample rate
            top_db: Threshold in dB below peak

        Returns:
            List of (start_sample, end_sample) tuples for each event
        """
        # Get intervals where audio is not silent
        intervals = librosa.effects.split(audio, top_db=top_db)
        return intervals.tolist()

    @staticmethod
    def get_audio_duration(audio_path: str) -> float:
        """
        Get audio duration in seconds

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        audio = AudioSegment.from_wav(audio_path)
        return len(audio) / 1000.0

    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range

        Args:
            audio: Audio data

        Returns:
            Normalized audio
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio

    @staticmethod
    def apply_gaussian_noise_variable(
        audio: np.ndarray,
        noise_factor: float = 0.005
    ) -> np.ndarray:
        """
        Apply Gaussian noise with variable factor

        Args:
            audio: Audio data
            noise_factor: Noise factor (0.005 = 0.5% noise)

        Returns:
            Audio with noise
        """
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise

    @staticmethod
    def swap_audio_segments(
        audio: np.ndarray,
        intervals: List[Tuple[int, int]],
        swap_indices: Tuple[int, int]
    ) -> np.ndarray:
        """
        Swap two audio segments

        Args:
            audio: Audio data
            intervals: List of (start, end) intervals
            swap_indices: Indices of intervals to swap

        Returns:
            Audio with swapped segments
        """
        if len(intervals) < 2:
            return audio

        idx1, idx2 = swap_indices
        if idx1 >= len(intervals) or idx2 >= len(intervals):
            return audio

        # Extract segments
        seg1_start, seg1_end = intervals[idx1]
        seg2_start, seg2_end = intervals[idx2]

        # Create new audio
        result = audio.copy()

        # Swap segments
        seg1 = audio[seg1_start:seg1_end].copy()
        seg2 = audio[seg2_start:seg2_end].copy()

        # Handle different lengths
        if len(seg1) <= (seg2_end - seg2_start):
            result[seg2_start:seg2_start + len(seg1)] = seg1
        if len(seg2) <= (seg1_end - seg1_start):
            result[seg1_start:seg1_start + len(seg2)] = seg2

        return result


def create_temp_audio_file(audio: np.ndarray, sr: int = 16000, prefix: str = "temp") -> str:
    """
    Create a temporary audio file

    Args:
        audio: Audio data
        sr: Sample rate
        prefix: Prefix for filename

    Returns:
        Path to temporary file
    """
    import tempfile
    import os

    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"{prefix}_{np.random.randint(0, 1000000)}.wav")

    sf.write(temp_file, audio, sr)
    return temp_file


if __name__ == "__main__":
    # Test audio processing utilities
    print("Testing Audio Processing Utilities...")

    # Test with a sample from TREA dataset
    sample_audio = "TREA_dataset/count/audios/0.wav"

    if Path(sample_audio).exists():
        # Load audio
        audio, sr = AudioProcessor.load_audio(sample_audio)
        print(f"\nLoaded audio: {sample_audio}")
        print(f"  Duration: {len(audio) / sr:.2f} seconds")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Shape: {audio.shape}")

        # Test silence addition
        audio_with_silence = AudioProcessor.add_silence(audio, sr, duration=0.5, position="middle")
        print(f"\nAdded silence:")
        print(f"  New duration: {len(audio_with_silence) / sr:.2f} seconds")

        # Test volume adjustment
        audio_louder = AudioProcessor.adjust_volume(audio, gain=0.2)
        print(f"\nVolume adjustment:")
        print(f"  Original max amplitude: {np.abs(audio).max():.4f}")
        print(f"  Adjusted max amplitude: {np.abs(audio_louder).max():.4f}")

        # Test noise addition
        audio_noisy = AudioProcessor.add_noise(audio, snr_db=30)
        print(f"\nNoise addition:")
        print(f"  SNR: 30 dB")

        # Test event detection
        events = AudioProcessor.extract_events(audio, sr, top_db=20)
        print(f"\nDetected events: {len(events)}")
        for i, (start, end) in enumerate(events):
            duration = (end - start) / sr
            print(f"  Event {i+1}: {duration:.3f}s")

        print("\nAudio processing utilities test completed successfully!")
    else:
        print(f"Sample audio file not found: {sample_audio}")
        print("Please ensure TREA_dataset is in the correct location.")
