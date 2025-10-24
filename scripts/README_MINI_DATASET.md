# Mini Dataset Scripts for Google Colab

This directory contains scripts to create a lightweight version of FESTA for fast uploading to Google Colab.

## Problem

The full TREA dataset is **908MB** (620 audio files). Uploading directly to Colab takes 20-30+ minutes.

## Solution

Create a **mini dataset** with only the samples needed for quick testing:
- **Mini dataset**: ~20-30MB (15 audio files)
- **Upload time**: 1-2 minutes ⚡
- **Size reduction**: 97%

---

## Quick Usage

### Step 1: Generate Mini Dataset

```bash
# Navigate to project root
cd AudioLLM-FESTA

# Run script
python scripts/create_mini_dataset.py
```

**Output**: `mini-TREA_dataset/` folder (~20-30MB)

**What it does**:
- Selects 5 samples per task (count, order, duration) = 15 total
- Copies only those 15 audio files
- Copies corresponding CSV metadata
- Copies subset of synthetic silence files
- Preserves exact directory structure

**Time**: ~30 seconds

---

### Step 2: Package for Colab

```bash
# Run packaging script
python scripts/package_for_colab.py
```

**Output**: `AudioLLM-FESTA-colab.zip` (~25-30MB)

**What it does**:
- Copies all code (src/, experiments/, notebooks/)
- Copies configuration files
- Renames mini-TREA_dataset → TREA_dataset
- Excludes full TREA_dataset (saves 900MB!)
- Excludes unnecessary files (.git, __pycache__, etc.)
- Creates ready-to-upload zip file

**Time**: ~10 seconds

---

### Step 3: Upload to Colab

1. Go to https://colab.research.google.com
2. Upload `AudioLLM-FESTA-colab.zip` (1-2 minutes)
3. Run the notebook
4. Complete experiment in ~10-15 minutes

**Total time**: ~15 minutes (vs 45+ minutes with full dataset)

---

## Detailed Script Documentation

### create_mini_dataset.py

**Purpose**: Generate mini version of TREA dataset

**Configuration** (edit at top of script):
```python
SAMPLES_PER_TASK = 5          # Samples per task
SYNTHETIC_SILENCES_COUNT = 10  # Number of silence files
RANDOM_SEED = 42               # For reproducibility
```

**Process**:
1. Creates `mini-TREA_dataset/` directory structure
2. For each task (count, order, duration):
   - Reads CSV file
   - Randomly selects N samples (stratified if possible)
   - Copies corresponding audio files
   - Updates CSV with new paths
   - Saves mini CSV files
3. Copies subset of synthetic silence files
4. Displays summary with size comparison

**Output Structure**:
```
mini-TREA_dataset/
├── count/
│   ├── audios/
│   │   ├── 0.wav
│   │   ├── 15.wav
│   │   ├── ...          (5 files)
│   ├── count.csv
│   └── count_with_metadata.csv
├── order/
│   └── ...                (same structure)
├── duration/
│   └── ...                (same structure)
└── synthetic_silences/
    └── ...                (~10 files)
```

**Example Output**:
```
📊 MINI DATASET SUMMARY
═════════════════════════════════════════════════════════════

Task            Samples    CSV Files
───────────────────────────────────────────────────────────
count           5          ✅
order           5          ✅
duration        5          ✅

Category                       Count
───────────────────────────────────────────────────────────
Total audio samples            15
Synthetic silences             10
Total files                    25

Size                           Value
───────────────────────────────────────────────────────────
Mini dataset size              24.35 MB
Original dataset size          907.82 MB
Size reduction                 97.3%
```

---

### package_for_colab.py

**Purpose**: Package code + mini dataset for Colab upload

**Configuration** (edit at top of script):
```python
INCLUDE_DIRS = ['src', 'experiments', 'notebooks']
INCLUDE_FILES = ['config_colab.yaml', 'requirements.txt', ...]
EXCLUDE_PATTERNS = ['__pycache__', '*.pyc', '.git', ...]
```

**Process**:
1. Checks mini dataset exists (runs create_mini_dataset.py if needed)
2. Creates `AudioLLM-FESTA-colab/` temporary directory
3. Copies code and configuration files (excludes unnecessary files)
4. Copies mini-TREA_dataset as TREA_dataset (important!)
5. Creates COLAB_README.md with instructions
6. Creates zip file with compression
7. Displays summary with size comparison
8. Cleans up temporary directory

**Example Output**:
```
📊 PACKAGE SUMMARY
═════════════════════════════════════════════════════════════

Component                      Size
───────────────────────────────────────────────────────────
Python code                    156.3 KB
Mini dataset                   24.35 MB
Configuration & docs           47.2 KB
Total (uncompressed)           24.55 MB
Zip file                       22.18 MB

Comparison                     Value
───────────────────────────────────────────────────────────
Original dataset               907.82 MB
Mini dataset                   24.35 MB
Size reduction                 97.3%

✅ Package created: AudioLLM-FESTA-colab.zip
```

---

## Customization

### Adjust Sample Count

To create a different size mini dataset:

```python
# In create_mini_dataset.py, line ~13
SAMPLES_PER_TASK = 10  # Instead of 5

# Results in:
# - 30 total samples (10 × 3 tasks)
# - ~50-60MB mini dataset
# - Still much faster than 908MB full dataset
```

### Include Additional Files

To include more files in the package:

```python
# In package_for_colab.py, line ~27
INCLUDE_FILES = [
    'config_colab.yaml',
    'requirements.txt',
    'README.md',
    'your_custom_file.txt',  # Add here
]
```

---

## Troubleshooting

### "Mini dataset not found"

**Problem**: Running `package_for_colab.py` before `create_mini_dataset.py`

**Solution**:
```bash
python scripts/create_mini_dataset.py  # Run this first
python scripts/package_for_colab.py    # Then this
```

---

### "Original dataset not found"

**Problem**: TREA_dataset/ not in project root

**Solution**:
```bash
# Check if dataset exists
ls TREA_dataset/

# If not, ensure it's in the right location:
AudioLLM-FESTA/
├── TREA_dataset/          # Must be here
│   ├── count/
│   ├── order/
│   └── duration/
└── scripts/
```

---

### "Permission denied"

**Problem**: Insufficient permissions

**Solution**:
```bash
# On macOS/Linux:
chmod +x scripts/create_mini_dataset.py
chmod +x scripts/package_for_colab.py

# Or run with python explicitly:
python scripts/create_mini_dataset.py
```

---

### "Zip file too large"

**Problem**: Still including full dataset

**Solution**:
1. Check `mini-TREA_dataset/` was created
2. Verify package_for_colab.py excludes `TREA_dataset/`
3. Delete old zip and regenerate:
   ```bash
   rm AudioLLM-FESTA-colab.zip
   python scripts/package_for_colab.py
   ```

---

## Scaling Up in Colab

After testing with mini dataset, you can scale up:

### Option 1: Upload to Google Drive (Recommended)

```python
# In Colab, mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Upload full TREA_dataset to Drive (one-time, can do in background)
# Then in config:
dataset:
  data_dir: "/content/drive/MyDrive/TREA_dataset"
  samples_per_task: 30  # Full experiment
```

### Option 2: Generate Larger Mini Dataset

```bash
# Locally, edit create_mini_dataset.py:
SAMPLES_PER_TASK = 30

# Regenerate:
python scripts/create_mini_dataset.py
python scripts/package_for_colab.py

# New zip will be ~100-150MB (still faster than 908MB)
```

### Option 3: Direct Upload (Full Dataset)

If you have good internet:
```bash
# Zip only the dataset
cd AudioLLM-FESTA
zip -r TREA_dataset.zip TREA_dataset/

# Upload separately to Colab
# Takes 20-30 minutes but gives you full dataset
```

---

## What's Included in Mini Dataset

### Audio Samples
- **15 total samples**: 5 per task
- **Selection method**: Random with fixed seed (reproducible)
- **Diversity**: Tries to sample different difficulties
- **Format**: Same as original (.wav files)

### CSV Metadata
- **count.csv**, **order.csv**, **duration.csv**
- Contains questions, options, correct answers
- Paths updated to point to mini-TREA_dataset/

### Synthetic Silences
- **10 silence files**: Subset of all available
- **Used for**: FCS generation (adding events)
- **Selection**: Random sampling

### What's NOT Included
- ❌ Full 620 audio files (only 15 included)
- ❌ Development/test sets (mini uses small sample)
- ❌ Extended metadata fields (kept minimal)

---

## Performance Expectations

### With Mini Dataset (5 samples/task)
- ✅ **Tests all components**: Full FESTA pipeline works
- ✅ **Faster iteration**: Quick testing and debugging
- ✅ **Lower memory**: Easier on Colab resources
- ⚠️ **Variable results**: Small sample size = higher variance
- ⚠️ **Not for evaluation**: Use for verification only

### Expected Metrics
- **Accuracy**: 30-60% (varies with sample selection)
- **FESTA AUROC**: 0.60-0.85 (varies significantly)
- **Comparison**: May still show FESTA > baselines
- **Use case**: Verify code works, not final results

### When to Use Full Dataset
- Final experiment results
- Paper-quality evaluation
- Reproducible research
- Comparing with published baselines

---

## File Structure After Running Scripts

```
AudioLLM-FESTA/
├── mini-TREA_dataset/              ← Created by create_mini_dataset.py
│   ├── count/
│   ├── order/
│   ├── duration/
│   └── synthetic_silences/
│
├── AudioLLM-FESTA-colab.zip        ← Created by package_for_colab.py
│
├── scripts/
│   ├── create_mini_dataset.py      ← Script 1
│   ├── package_for_colab.py        ← Script 2
│   └── README_MINI_DATASET.md      ← This file
│
├── TREA_dataset/                   ← Original (not touched)
│   └── ...
│
└── ... (rest of project)
```

---

## Summary Commands

```bash
# Full workflow (run from project root):

# 1. Generate mini dataset
python scripts/create_mini_dataset.py

# 2. Package for Colab
python scripts/package_for_colab.py

# 3. Upload to Colab
# Go to colab.research.google.com
# Upload AudioLLM-FESTA-colab.zip
# Run notebooks/colab_festa.ipynb

# Total time: ~2 minutes local + ~1-2 min upload + ~15 min experiment = ~20 min
# vs Full dataset: ~30 min upload + ~15 min experiment = ~45+ min
```

---

## FAQ

**Q: Can I use mini dataset for final results?**
A: No, only for testing. Use full dataset (30 samples/task) for evaluation.

**Q: How to include more samples?**
A: Edit `SAMPLES_PER_TASK` in create_mini_dataset.py, regenerate.

**Q: Can I distribute mini dataset?**
A: Check TREA dataset license first. Mini dataset contains actual audio data.

**Q: Does mini dataset work with all features?**
A: Yes! FES, FCS, baselines, metrics all work identically.

**Q: What if I need specific samples?**
A: Edit create_mini_dataset.py to select specific indices instead of random.

---

## Additional Resources

- **COLAB_INSTRUCTIONS.md** - Complete Colab setup guide
- **QUICK_START.md** - Quick start guide for local setup
- **README.md** - Full project documentation

---

**Questions?** Check the main documentation or open an issue on GitHub.

**Happy experimenting!** 🚀
