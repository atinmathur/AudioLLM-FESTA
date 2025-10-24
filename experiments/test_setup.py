"""
Test script to verify all components are working correctly
Run this before the full FESTA experiment
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    try:
        import torch
        import transformers
        import librosa
        import soundfile
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        import tqdm
        import yaml
        print("  ✓ All core dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_project_imports():
    """Test project module imports"""
    print("\nTesting project modules...")
    try:
        from src.data_loader import TREADataset, load_trea_dataset
        from src.model_wrapper import Qwen2AudioWrapper
        from src.fes_generator import FESGenerator
        from src.fcs_generator import FCSGenerator
        from src.uncertainty import FESTAUncertainty
        from src.metrics import compute_auroc, compute_accuracy
        from src.baselines import BaselineUncertainty
        from src.utils import AudioProcessor
        print("  ✓ All project modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_data_loading():
    """Test data loading"""
    print("\nTesting data loading...")
    try:
        from src.data_loader import load_trea_dataset

        dataset = load_trea_dataset(
            data_dir='TREA_dataset',
            tasks=['count', 'order', 'duration'],
            samples_per_task=5,
            random_seed=42
        )

        assert len(dataset) == 15, f"Expected 15 samples, got {len(dataset)}"

        # Check first sample structure
        sample = dataset[0]
        assert 'audio_path' in sample
        assert 'question' in sample
        assert 'options' in sample
        assert 'correct_answer' in sample
        assert 'task' in sample

        # Check audio file exists
        audio_path = Path(sample['audio_path'])
        assert audio_path.exists(), f"Audio file not found: {audio_path}"

        print(f"  ✓ Loaded {len(dataset)} samples successfully")
        print(f"    Tasks: {set(s['task'] for s in dataset.data)}")
        return True

    except Exception as e:
        print(f"  ✗ Data loading error: {e}")
        return False


def test_audio_processing():
    """Test audio processing utilities"""
    print("\nTesting audio processing...")
    try:
        from src.utils import AudioProcessor
        from src.data_loader import load_trea_dataset

        dataset = load_trea_dataset(
            data_dir='TREA_dataset',
            samples_per_task=1,
            random_seed=42
        )

        sample = dataset[0]
        processor = AudioProcessor()

        # Load audio
        audio, sr = processor.load_audio(sample['audio_path'], sr=16000)
        assert len(audio) > 0, "Audio is empty"
        assert sr == 16000, f"Expected sr=16000, got {sr}"

        # Test transformations
        audio_silence = processor.add_silence(audio, sr, duration=0.2)
        assert len(audio_silence) > len(audio), "Silence not added"

        audio_volume = processor.adjust_volume(audio, gain=0.1)
        assert len(audio_volume) == len(audio), "Volume adjustment changed length"

        audio_noise = processor.add_noise(audio, snr_db=30)
        assert len(audio_noise) == len(audio), "Noise addition changed length"

        print(f"  ✓ Audio processing works correctly")
        print(f"    Audio shape: {audio.shape}, SR: {sr}")
        return True

    except Exception as e:
        print(f"  ✗ Audio processing error: {e}")
        return False


def test_fes_generator():
    """Test FES generator"""
    print("\nTesting FES generator...")
    try:
        from src.fes_generator import FESGenerator
        from src.data_loader import load_trea_dataset

        dataset = load_trea_dataset(
            data_dir='TREA_dataset',
            samples_per_task=1,
            random_seed=42
        )

        sample = dataset[0]

        fes_gen = FESGenerator(n_audio_samples=3, n_text_samples=2)
        fes_samples = fes_gen.generate(
            sample['audio_path'],
            sample['question'],
            sample['task'],
            sample['options']
        )

        assert len(fes_samples) == 6, f"Expected 6 FES samples (3*2), got {len(fes_samples)}"

        # Check sample structure
        for fes in fes_samples:
            assert 'audio_path' in fes
            assert 'question' in fes
            assert 'options' in fes
            assert Path(fes['audio_path']).exists(), f"FES audio not found: {fes['audio_path']}"

        # Cleanup
        fes_gen.cleanup_temp_files()

        print(f"  ✓ Generated {len(fes_samples)} FES samples successfully")
        return True

    except Exception as e:
        print(f"  ✗ FES generator error: {e}")
        return False


def test_fcs_generator():
    """Test FCS generator"""
    print("\nTesting FCS generator...")
    try:
        from src.fcs_generator import FCSGenerator
        from src.data_loader import load_trea_dataset

        dataset = load_trea_dataset(
            data_dir='TREA_dataset',
            samples_per_task=1,
            random_seed=42
        )

        sample = dataset[0]

        fcs_gen = FCSGenerator(
            n_audio_samples=3,
            n_text_samples=2,
            synthetic_silence_dir='TREA_dataset/synthetic_silences'
        )

        fcs_samples = fcs_gen.generate(
            sample['audio_path'],
            sample['question'],
            sample['task'],
            sample['options'],
            'A'  # Mock original prediction
        )

        assert len(fcs_samples) > 0, "No FCS samples generated"

        # Check sample structure
        for fcs in fcs_samples[:3]:  # Check first 3
            assert 'audio_path' in fcs
            assert 'question' in fcs
            assert 'options' in fcs

        # Cleanup
        fcs_gen.cleanup_temp_files()

        print(f"  ✓ Generated {len(fcs_samples)} FCS samples successfully")
        return True

    except Exception as e:
        print(f"  ✗ FCS generator error: {e}")
        return False


def test_uncertainty_computation():
    """Test FESTA uncertainty computation"""
    print("\nTesting FESTA uncertainty...")
    try:
        from src.uncertainty import FESTAUncertainty

        festa = FESTAUncertainty()

        # Test case: Consistent model (low uncertainty)
        original = "A"
        fes_preds = ["A"] * 20 + ["B"] * 5
        fcs_preds = ["B"] * 15 + ["C"] * 10

        result = festa.compute_festa(fes_preds, fcs_preds, original)

        assert 'U_FES' in result
        assert 'U_FCS' in result
        assert 'U_FESTA' in result

        assert result['U_FES'] >= 0, "U_FES should be non-negative"
        assert result['U_FCS'] >= 0, "U_FCS should be non-negative"
        assert result['U_FESTA'] >= 0, "U_FESTA should be non-negative"

        print(f"  ✓ FESTA uncertainty computation works")
        print(f"    U_FES: {result['U_FES']:.4f}, U_FCS: {result['U_FCS']:.4f}, U_FESTA: {result['U_FESTA']:.4f}")
        return True

    except Exception as e:
        print(f"  ✗ Uncertainty computation error: {e}")
        return False


def test_metrics():
    """Test evaluation metrics"""
    print("\nTesting metrics...")
    try:
        from src.metrics import compute_auroc, compute_accuracy
        import numpy as np

        # Generate synthetic data
        np.random.seed(42)
        n = 50

        predictions = list(np.random.choice(['A', 'B', 'C', 'D'], n))
        ground_truths = list(np.random.choice(['A', 'B', 'C', 'D'], n))

        # Uncertainties correlated with correctness
        uncertainties = []
        for p, g in zip(predictions, ground_truths):
            if p == g:
                uncertainties.append(np.random.uniform(0, 2))
            else:
                uncertainties.append(np.random.uniform(2, 5))

        # Compute accuracy
        accuracy = compute_accuracy(predictions, ground_truths)
        assert 0 <= accuracy <= 1, f"Accuracy out of range: {accuracy}"

        # Compute AUROC
        auroc = compute_auroc(uncertainties, predictions, ground_truths)
        assert 0 <= auroc <= 1, f"AUROC out of range: {auroc}"

        print(f"  ✓ Metrics computation works")
        print(f"    Accuracy: {accuracy:.2%}, AUROC: {auroc:.4f}")
        return True

    except Exception as e:
        print(f"  ✗ Metrics error: {e}")
        return False


def test_config():
    """Test configuration file"""
    print("\nTesting configuration...")
    try:
        import yaml

        config_path = Path('config.yaml')
        assert config_path.exists(), "config.yaml not found"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check required sections
        assert 'model' in config
        assert 'dataset' in config
        assert 'festa' in config
        assert 'baselines' in config
        assert 'metrics' in config
        assert 'experiment' in config

        print("  ✓ Configuration file is valid")
        return True

    except Exception as e:
        print(f"  ✗ Config error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("FESTA Setup Verification")
    print("="*70)

    tests = [
        ("Core Dependencies", test_imports),
        ("Project Modules", test_project_imports),
        ("Data Loading", test_data_loading),
        ("Audio Processing", test_audio_processing),
        ("FES Generator", test_fes_generator),
        ("FCS Generator", test_fcs_generator),
        ("Uncertainty Computation", test_uncertainty_computation),
        ("Metrics", test_metrics),
        ("Configuration", test_config)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:<30} {status}")

    print("="*70)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! You're ready to run FESTA experiments.")
        print("\nNext steps:")
        print("  1. Run: cd experiments && python run_festa.py")
        print("  2. Or open Jupyter notebooks in notebooks/")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before proceeding.")
        print("\nTroubleshooting:")
        print("  1. Check that all dependencies are installed: pip install -r requirements.txt")
        print("  2. Verify TREA_dataset directory exists")
        print("  3. Ensure audio files are present in TREA_dataset/")
        return 1


if __name__ == "__main__":
    exit(main())
