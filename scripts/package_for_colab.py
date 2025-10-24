"""
Package FESTA for Google Colab

This script creates a lightweight package for Colab containing:
- All code (src/, experiments/, notebooks/)
- Configuration files
- Mini dataset (renamed to TREA_dataset)

Usage:
    python scripts/package_for_colab.py

Prerequisites:
    Run create_mini_dataset.py first to generate mini-TREA_dataset/

Output:
    AudioLLM-FESTA-colab.zip (~25-30MB)
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MINI_DATASET = PROJECT_ROOT / "mini-TREA_dataset"
PACKAGE_DIR = PROJECT_ROOT / "AudioLLM-FESTA"
OUTPUT_ZIP = PROJECT_ROOT / "AudioLLM-FESTA.zip"

# Directories and files to include
INCLUDE_DIRS = [
    'src',
    'experiments',
    'notebooks',
]

INCLUDE_FILES = [
    'config_colab.yaml',
    'requirements.txt',
    'README.md',
    'QUICK_START.md',
    'IMPLEMENTATION_SUMMARY.md',
    'COLAB_INSTRUCTIONS.md',
]

# Patterns to exclude
EXCLUDE_PATTERNS = [
    '__pycache__',
    '*.pyc',
    '*.pyo',
    '.git',
    '.gitignore',
    '.DS_Store',
    '*.egg-info',
    '.pytest_cache',
    '.ipynb_checkpoints',
    'results',
    'festa_results',
    '*.log',
]


def check_mini_dataset():
    """Check if mini dataset exists"""
    print("🔍 Checking for mini dataset...")

    if not MINI_DATASET.exists():
        print(f"\n❌ ERROR: Mini dataset not found at {MINI_DATASET}")
        print(f"\nPlease run first:")
        print(f"  python scripts/create_mini_dataset.py")
        return False

    # Count audio files
    audio_count = len(list(MINI_DATASET.rglob('*.wav')))
    if audio_count == 0:
        print(f"\n❌ ERROR: No audio files found in mini dataset")
        return False

    print(f"   ✅ Found mini dataset with {audio_count} audio files")
    return True


def create_package_directory():
    """Create clean package directory"""
    print("\n📁 Creating package directory...")

    # Remove existing package directory
    if PACKAGE_DIR.exists():
        print(f"   Removing existing {PACKAGE_DIR.name}/")
        shutil.rmtree(PACKAGE_DIR)

    # Create base directory
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Created: {PACKAGE_DIR.name}/")


def should_exclude(path, base_path):
    """Check if path should be excluded"""
    rel_path = path.relative_to(base_path)

    # Check against exclude patterns
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith('*'):
            # File extension pattern
            if str(rel_path).endswith(pattern[1:]):
                return True
        else:
            # Directory or filename pattern
            if pattern in str(rel_path):
                return True

    return False


def copy_directory(src, dst, base_path):
    """
    Copy directory while excluding patterns

    Args:
        src: Source directory
        dst: Destination directory
        base_path: Base path for relative path calculation
    """
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        src_path = src / item.name
        dst_path = dst / item.name

        # Skip if should be excluded
        if should_exclude(src_path, base_path):
            continue

        if src_path.is_dir():
            copy_directory(src_path, dst_path, base_path)
        else:
            shutil.copy2(src_path, dst_path)


def copy_code_and_configs():
    """Copy all code directories and configuration files"""
    print("\n📋 Copying code and configurations...")

    # Copy directories
    for dir_name in INCLUDE_DIRS:
        src_dir = PROJECT_ROOT / dir_name
        dst_dir = PACKAGE_DIR / dir_name

        if src_dir.exists():
            print(f"   Copying {dir_name}/")
            copy_directory(src_dir, dst_dir, PROJECT_ROOT)
            file_count = len(list(dst_dir.rglob('*')))
            print(f"     ✅ {file_count} items copied")
        else:
            print(f"   ⚠️  {dir_name}/ not found, skipping")

    # Copy individual files
    print(f"\n   Copying configuration files:")
    for file_name in INCLUDE_FILES:
        src_file = PROJECT_ROOT / file_name
        dst_file = PACKAGE_DIR / file_name

        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            size_kb = src_file.stat().st_size / 1024
            print(f"     ✅ {file_name} ({size_kb:.1f} KB)")
        else:
            print(f"     ⚠️  {file_name} not found, skipping")

    print(f"   ✅ Code and configurations copied")


def copy_mini_dataset_as_trea():
    """Copy mini dataset and rename to TREA_dataset"""
    print("\n📦 Copying mini dataset as TREA_dataset...")

    src_dataset = MINI_DATASET
    dst_dataset = PACKAGE_DIR / "TREA_dataset"

    if not src_dataset.exists():
        print(f"   ❌ Mini dataset not found")
        return False

    # Copy and rename
    print(f"   Copying mini-TREA_dataset/ → TREA_dataset/")
    shutil.copytree(src_dataset, dst_dataset)

    # Count files
    audio_count = len(list(dst_dataset.rglob('*.wav')))
    csv_count = len(list(dst_dataset.rglob('*.csv')))

    print(f"   ✅ Copied {audio_count} audio files, {csv_count} CSV files")
    return True


def create_readme():
    """Create README for the package"""
    readme_content = """# FESTA for Google Colab

This package contains a lightweight version of FESTA for quick testing on Google Colab.

## What's Included

- **Code**: All source code, experiments, and notebooks
- **Mini Dataset**: 15 samples (5 per task) instead of full 620 samples
- **Configuration**: Pre-configured for Colab with quick test settings

## Size

- **This package**: ~25-30 MB
- **Full dataset**: ~908 MB
- **Size reduction**: ~97%

## Quick Start

1. **Upload to Colab**: Upload `AudioLLM-FESTA-colab.zip`
2. **Open notebook**: Open `notebooks/colab_festa.ipynb`
3. **Run all cells**: Runtime → Run all
4. **Wait ~10-15 minutes**: Complete experiment

## What You'll Get

- FESTA uncertainty scores
- Baseline comparisons
- AUROC metrics (~0.70-0.85 expected)
- Visualization plots

## Limitations

- Only 15 samples (vs 90 in full dataset)
- Results may vary due to small sample size
- Use for testing/verification, not final evaluation

## Scaling Up

To run full experiments later:

**Option 1**: Upload to Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
# Upload full TREA_dataset to Drive
```

**Option 2**: Generate larger mini dataset
```bash
# In create_mini_dataset.py, change:
SAMPLES_PER_TASK = 30
```

## Documentation

- `README.md` - Full project documentation
- `COLAB_INSTRUCTIONS.md` - Detailed Colab setup guide
- `QUICK_START.md` - Quick start guide

## Support

For issues or questions, see COLAB_INSTRUCTIONS.md

---

**Generated**: {timestamp}
**Package size**: Will be shown after creation
"""

    readme_path = PACKAGE_DIR / "COLAB_README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    print(f"\n📝 Created COLAB_README.md")


def create_zip():
    """Create zip file of the package"""
    print("\n📦 Creating zip file...")

    # Remove existing zip
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
        print(f"   Removed existing {OUTPUT_ZIP.name}")

    # Create zip
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(PACKAGE_DIR):
            for file in files:
                file_path = Path(root) / file
                arc_path = file_path.relative_to(PACKAGE_DIR.parent)
                zipf.write(file_path, arc_path)

    zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024**2)
    print(f"   ✅ Created: {OUTPUT_ZIP.name} ({zip_size_mb:.2f} MB)")

    return zip_size_mb


def cleanup():
    """Remove temporary package directory"""
    print("\n🧹 Cleaning up...")

    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
        print(f"   ✅ Removed temporary directory: {PACKAGE_DIR.name}")


def show_summary(zip_size_mb):
    """Display final summary"""
    print("\n" + "="*60)
    print("📊 PACKAGE SUMMARY")
    print("="*60)

    # Analyze package contents
    package_size = sum(f.stat().st_size for f in PACKAGE_DIR.rglob('*') if f.is_file())

    print(f"\n{'Component':<30} {'Size'}")
    print("-"*60)

    # Count sizes by type
    code_size = sum(
        f.stat().st_size for f in PACKAGE_DIR.rglob('*.py')
    )
    dataset_size = sum(
        f.stat().st_size for f in (PACKAGE_DIR / 'TREA_dataset').rglob('*') if f.is_file()
    ) if (PACKAGE_DIR / 'TREA_dataset').exists() else 0

    config_size = sum(
        f.stat().st_size for f in PACKAGE_DIR.glob('*.yaml')
    ) + sum(
        f.stat().st_size for f in PACKAGE_DIR.glob('*.md')
    )

    print(f"{'Python code':<30} {code_size / 1024:.1f} KB")
    print(f"{'Mini dataset':<30} {dataset_size / (1024**2):.2f} MB")
    print(f"{'Configuration & docs':<30} {config_size / 1024:.1f} KB")
    print(f"{'Total (uncompressed)':<30} {package_size / (1024**2):.2f} MB")
    print(f"{'Zip file':<30} {zip_size_mb:.2f} MB")

    # Compare with original
    original_dataset_size = sum(
        f.stat().st_size for f in (PROJECT_ROOT / 'TREA_dataset').rglob('*.wav')
    ) if (PROJECT_ROOT / 'TREA_dataset').exists() else 0

    if original_dataset_size > 0:
        print(f"\n{'Comparison':<30} {'Value'}")
        print("-"*60)
        print(f"{'Original dataset':<30} {original_dataset_size / (1024**2):.2f} MB")
        print(f"{'Mini dataset':<30} {dataset_size / (1024**2):.2f} MB")
        reduction = (1 - dataset_size / original_dataset_size) * 100
        print(f"{'Size reduction':<30} {reduction:.1f}%")

    print("\n" + "="*60)
    print(f"✅ Package created: {OUTPUT_ZIP.name}")
    print("="*60)

    print("\n📝 Next steps:")
    print(f"  1. Upload {OUTPUT_ZIP.name} to Google Colab")
    print(f"  2. Open notebooks/colab_festa.ipynb")
    print(f"  3. Follow the instructions in the notebook")
    print(f"  4. Experiment will complete in ~10-15 minutes!")

    print(f"\n💡 Tip: Upload time ~1-2 minutes (vs 30+ min for full dataset)")


def main():
    """Main execution"""
    print("="*60)
    print("📦 PACKAGING FESTA FOR GOOGLE COLAB")
    print("="*60)

    try:
        # Check prerequisites
        if not check_mini_dataset():
            return 1

        # Create package
        create_package_directory()
        copy_code_and_configs()

        if not copy_mini_dataset_as_trea():
            return 1

        create_readme()

        # Create zip
        zip_size_mb = create_zip()

        # Show summary before cleanup (while package dir still exists)
        show_summary(zip_size_mb)

        # Cleanup
        cleanup()

        print("\n🎉 Package created successfully!")
        print(f"\n📦 Ready to upload: {OUTPUT_ZIP}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

        # Cleanup on error
        if PACKAGE_DIR.exists():
            shutil.rmtree(PACKAGE_DIR)

        return 1

    return 0


if __name__ == "__main__":
    exit(main())
