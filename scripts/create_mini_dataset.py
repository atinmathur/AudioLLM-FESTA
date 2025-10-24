"""
Create Mini TREA Dataset for Fast Colab Upload

This script creates a lightweight version of the TREA dataset containing
only the samples needed for quick testing (5 samples per task = 15 total).

Usage:
    python scripts/create_mini_dataset.py

Output:
    mini-TREA_dataset/ folder (~20-30MB instead of 908MB)
"""

import os
import shutil
import pandas as pd
from pathlib import Path
import random

# Configuration
SAMPLES_PER_TASK = 5
SYNTHETIC_SILENCES_COUNT = 10
RANDOM_SEED = 42

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
ORIGINAL_DATASET = PROJECT_ROOT / "TREA_dataset"
MINI_DATASET = PROJECT_ROOT / "mini-TREA_dataset"

random.seed(RANDOM_SEED)


def create_directory_structure():
    """Create mini dataset directory structure"""
    print("📁 Creating directory structure...")

    # Remove existing mini dataset if present
    if MINI_DATASET.exists():
        print(f"   Removing existing {MINI_DATASET.name}/")
        shutil.rmtree(MINI_DATASET)

    # Create base directory
    MINI_DATASET.mkdir(parents=True, exist_ok=True)

    # Create task directories
    for task in ['count', 'order', 'duration']:
        task_dir = MINI_DATASET / task
        task_dir.mkdir(exist_ok=True)
        (task_dir / 'audios').mkdir(exist_ok=True)

    # Create synthetic_silences directory
    (MINI_DATASET / 'synthetic_silences').mkdir(exist_ok=True)

    print("   ✅ Directory structure created")


def select_and_copy_task_samples(task_name):
    """
    Select and copy samples for a specific task

    Args:
        task_name: Task name ('count', 'order', 'duration')
    """
    print(f"\n📋 Processing '{task_name}' task...")

    task_dir = ORIGINAL_DATASET / task_name
    mini_task_dir = MINI_DATASET / task_name

    # Read CSV
    csv_file = task_dir / f"{task_name}.csv"
    csv_with_meta = task_dir / f"{task_name}_with_metadata.csv"

    if not csv_file.exists():
        print(f"   ⚠️  CSV not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)

    # Select samples
    total_samples = len(df)
    if total_samples < SAMPLES_PER_TASK:
        selected_indices = list(range(total_samples))
        print(f"   ⚠️  Only {total_samples} samples available, using all")
    else:
        # Use stratified sampling if possible, else random
        selected_indices = sorted(random.sample(range(total_samples), SAMPLES_PER_TASK))
        print(f"   Selected {SAMPLES_PER_TASK} samples from {total_samples} total")

    # Filter dataframe
    mini_df = df.iloc[selected_indices].reset_index(drop=True)

    # Copy audio files
    copied_count = 0
    for idx, row in mini_df.iterrows():
        # Get audio filename from path
        audio_path = row['audio_path']
        audio_filename = Path(audio_path).name

        # Source and destination
        src_audio = task_dir / 'audios' / audio_filename
        dst_audio = mini_task_dir / 'audios' / audio_filename

        if src_audio.exists():
            shutil.copy2(src_audio, dst_audio)
            copied_count += 1
        else:
            print(f"   ⚠️  Audio file not found: {src_audio}")

    print(f"   📁 Copied {copied_count} audio files")

    # Update audio paths in CSV to point to mini dataset
    mini_df['audio_path'] = mini_df['audio_path'].str.replace(
        'TREA_dataset', 'mini-TREA_dataset', regex=False
    )

    # Save mini CSV
    mini_csv = mini_task_dir / f"{task_name}.csv"
    mini_df.to_csv(mini_csv, index=False)
    print(f"   💾 Saved: {mini_csv.name}")

    # Copy metadata CSV if exists (with selected rows only)
    if csv_with_meta.exists():
        df_meta = pd.read_csv(csv_with_meta)
        mini_df_meta = df_meta.iloc[selected_indices].reset_index(drop=True)

        # Update paths
        if 'audio_path' in mini_df_meta.columns:
            mini_df_meta['audio_path'] = mini_df_meta['audio_path'].str.replace(
                'TREA_dataset', 'mini-TREA_dataset', regex=False
            )

        mini_csv_meta = mini_task_dir / f"{task_name}_with_metadata.csv"
        mini_df_meta.to_csv(mini_csv_meta, index=False)
        print(f"   💾 Saved: {mini_csv_meta.name}")

    print(f"   ✅ '{task_name}' task completed")


def copy_synthetic_silences():
    """Copy a subset of synthetic silence files"""
    print(f"\n🔇 Copying synthetic silence files...")

    src_dir = ORIGINAL_DATASET / 'synthetic_silences'
    dst_dir = MINI_DATASET / 'synthetic_silences'

    if not src_dir.exists():
        print(f"   ⚠️  Synthetic silences directory not found")
        return

    # Get all wav files
    silence_files = list(src_dir.glob('*.wav'))

    if not silence_files:
        print(f"   ⚠️  No silence files found")
        return

    # Select subset
    if len(silence_files) <= SYNTHETIC_SILENCES_COUNT:
        selected_files = silence_files
    else:
        selected_files = random.sample(silence_files, SYNTHETIC_SILENCES_COUNT)

    # Copy files
    for src_file in selected_files:
        dst_file = dst_dir / src_file.name
        shutil.copy2(src_file, dst_file)

    print(f"   📁 Copied {len(selected_files)} silence files")
    print(f"   ✅ Synthetic silences completed")


def show_summary():
    """Display summary of mini dataset"""
    print("\n" + "="*60)
    print("📊 MINI DATASET SUMMARY")
    print("="*60)

    # Count files and sizes
    audio_counts = {}
    total_size = 0

    for task in ['count', 'order', 'duration']:
        task_dir = MINI_DATASET / task / 'audios'
        if task_dir.exists():
            audio_files = list(task_dir.glob('*.wav'))
            audio_counts[task] = len(audio_files)

            # Calculate size
            task_size = sum(f.stat().st_size for f in audio_files)
            total_size += task_size

    # Synthetic silences
    silence_dir = MINI_DATASET / 'synthetic_silences'
    if silence_dir.exists():
        silence_files = list(silence_dir.glob('*.wav'))
        silence_count = len(silence_files)
        silence_size = sum(f.stat().st_size for f in silence_files)
        total_size += silence_size
    else:
        silence_count = 0
        silence_size = 0

    # Display
    print(f"\n{'Task':<15} {'Samples':<10} {'CSV Files'}")
    print("-"*60)
    for task in ['count', 'order', 'duration']:
        csv_file = MINI_DATASET / task / f"{task}.csv"
        csv_status = "✅" if csv_file.exists() else "❌"
        print(f"{task:<15} {audio_counts.get(task, 0):<10} {csv_status}")

    print(f"\n{'Category':<30} {'Count':<10}")
    print("-"*60)
    print(f"{'Total audio samples':<30} {sum(audio_counts.values()):<10}")
    print(f"{'Synthetic silences':<30} {silence_count:<10}")
    print(f"{'Total files':<30} {sum(audio_counts.values()) + silence_count:<10}")

    print(f"\n{'Size':<30} {'Value'}")
    print("-"*60)
    print(f"{'Mini dataset size':<30} {total_size / (1024**2):.2f} MB")

    # Compare with original
    if ORIGINAL_DATASET.exists():
        original_size = sum(
            f.stat().st_size
            for f in ORIGINAL_DATASET.rglob('*.wav')
        )
        print(f"{'Original dataset size':<30} {original_size / (1024**2):.2f} MB")
        reduction = (1 - total_size / original_size) * 100
        print(f"{'Size reduction':<30} {reduction:.1f}%")

    print("\n" + "="*60)
    print(f"✅ Mini dataset created: {MINI_DATASET}/")
    print("="*60)

    print("\n📝 Next steps:")
    print("  1. Run: python scripts/package_for_colab.py")
    print("  2. Upload: AudioLLM-FESTA-colab.zip to Colab")
    print("  3. Experiment will use mini dataset automatically!")


def main():
    """Main execution"""
    print("="*60)
    print("🚀 CREATING MINI TREA DATASET")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Samples per task: {SAMPLES_PER_TASK}")
    print(f"  Total samples: {SAMPLES_PER_TASK * 3}")
    print(f"  Synthetic silences: {SYNTHETIC_SILENCES_COUNT}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"\nSource: {ORIGINAL_DATASET}")
    print(f"Destination: {MINI_DATASET}\n")

    # Check if original dataset exists
    if not ORIGINAL_DATASET.exists():
        print(f"❌ ERROR: Original dataset not found at {ORIGINAL_DATASET}")
        print(f"   Please ensure TREA_dataset/ exists in project root.")
        return

    try:
        # Create directory structure
        create_directory_structure()

        # Process each task
        for task in ['count', 'order', 'duration']:
            select_and_copy_task_samples(task)

        # Copy synthetic silences
        copy_synthetic_silences()

        # Show summary
        show_summary()

        print("\n🎉 Mini dataset created successfully!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
