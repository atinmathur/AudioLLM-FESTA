# FESTA Colab Setup Guide - Full Dataset (300 Samples)

This guide will help you run the FESTA experiment on Google Colab using the full TREA dataset (100 samples per task = 300 total samples).

## Overview

**What this setup provides:**
- 📁 Dataset stored in Google Drive (upload once, reuse forever)
- 💾 Automatic checkpointing (resume after session timeouts)
- ☁️ Results saved to Google Drive
- 🔄 Complete reproducibility
- 📊 300 samples processed (100 per task: count, order, duration)

**Estimated time:** 5-8 hours on Colab GPU (T4 or better)

---

## Prerequisites

### 1. Google Account
- You need a Google account with access to Google Colab
- Recommended: Colab Pro for longer session times (optional but helpful)

### 2. TREA Dataset
- The complete TREA dataset folder (should already have it locally)
- Folder structure:
  ```
  TREA_dataset/
  ├── count/
  │   ├── count.csv
  │   ├── count_with_metadata.csv
  │   └── audio/
  ├── order/
  │   ├── order.csv
  │   ├── order_with_metadata.csv
  │   └── audio/
  └── duration/
      ├── duration.csv
      ├── duration_with_metadata.csv
      └── audio/
  ```

---

## Step-by-Step Setup

### Step 1: Upload Dataset to Google Drive

**IMPORTANT:** You only need to do this ONCE. After upload, you can run the experiment as many times as you want.

1. **Open Google Drive** in your browser (drive.google.com)

2. **Upload TREA_dataset folder**:
   - Click "New" → "Folder upload"
   - Select your local `TREA_dataset` folder
   - Upload to the **root level** of "My Drive" (not in a subfolder)
   - Wait for upload to complete (may take 10-30 minutes depending on size)

3. **Verify upload**:
   - After upload, you should see: `My Drive/TREA_dataset/`
   - It should contain the three subfolders: `count/`, `order/`, `duration/`
   - Each subfolder should have CSV files and audio files

**Expected structure in Google Drive:**
```
My Drive/
└── TREA_dataset/          ← Dataset folder in root of My Drive
    ├── count/
    ├── order/
    └── duration/
```

---

### Step 2: Upload Notebook to Google Colab

**Option A: Upload the notebook file**

1. Open [Google Colab](https://colab.research.google.com)
2. Click "File" → "Upload notebook"
3. Select `festa_colab_full.ipynb` from your local machine
4. The notebook will open in Colab

**Option B: Open from Google Drive** (if you uploaded the notebook to Drive)

1. Upload `festa_colab_full.ipynb` to your Google Drive
2. Right-click the file → "Open with" → "Google Colaboratory"

**Option C: Copy-paste from GitHub** (if repository is public)

1. Open [Google Colab](https://colab.research.google.com)
2. Click "GitHub" tab
3. Enter your repository URL
4. Select `festa_colab_full.ipynb`

---

### Step 3: Configure GPU Runtime

**IMPORTANT:** The experiment requires a GPU to run efficiently.

1. In Colab, click **"Runtime"** → **"Change runtime type"**
2. Set **"Hardware accelerator"** to **"GPU"**
3. (Optional) Choose **"High-RAM"** if available
4. Click **"Save"**

---

### Step 4: Run the Experiment

Now you're ready to run! Follow the cells in the notebook:

#### Cell 1: Mount Google Drive
- Run this cell
- Click the authorization link
- Sign in to your Google account
- Allow Colab to access your Drive

#### Cell 2: Verify GPU
- Run this cell to confirm GPU is available
- Should show GPU name and memory

#### Cell 3: Verify Dataset
- Run this cell to check if dataset is properly uploaded
- Should show: "✅ Dataset found in Google Drive!"
- Verifies all 3 tasks (count, order, duration)

#### Cell 4: Clone Repository
- **IMPORTANT:** Update the GitHub URL in this cell to your repository
- Or copy the code from Drive if you uploaded it there
- This downloads the FESTA code to Colab

#### Cell 5: Install Dependencies
- Installs all required Python packages
- Takes 1-2 minutes

#### Cell 6: Check Existing Progress
- Shows if you have previous runs
- Displays checkpoint progress if resuming

#### Cell 7: Run Experiment
- **This is the main cell - runs the full experiment**
- Takes 5-8 hours to complete
- Saves checkpoint after each sample
- Can be interrupted and resumed

#### Cell 8: Monitor Progress
- Run this **while** experiment is running (in a new cell)
- Shows real-time progress
- Updates every 5 seconds

---

## Running the Experiment

### First-Time Run

1. Run cells 1-7 in order
2. Cell 7 will start the experiment
3. Optionally run cell 8 in parallel to monitor progress

**What happens:**
- Processes 300 samples (100 per task)
- Saves checkpoint after each sample to Google Drive
- Saves intermediate results every sample
- Each sample takes ~1-2 minutes
- Total time: 5-8 hours

### Resuming After Timeout

If Colab disconnects or times out:

1. **Reopen the notebook** in Colab
2. **Re-run cells 1-6** (mount Drive, setup environment)
3. **Cell 6 will show your progress** (e.g., "150/300 samples completed")
4. **Re-run cell 7** - automatically resumes from last checkpoint
5. **Continue where you left off** - no data lost!

**The checkpoint system ensures:**
- ✅ No duplicate work
- ✅ All completed samples are skipped
- ✅ Picks up exactly where it stopped

---

## Understanding the Results

### During Experiment

**Checkpoint file:** `/content/drive/MyDrive/festa_checkpoint.json`
- Tracks completed samples
- Shows progress percentage
- Last updated timestamp

**Intermediate results:** `/content/drive/MyDrive/festa_results/intermediate_results.json`
- Updated after each sample
- Contains all results so far

### After Completion

**Final result files in** `/content/drive/MyDrive/festa_results/`:

1. **`predictions_YYYYMMDD_HHMMSS.json`**
   - All model predictions
   - Ground truth labels
   - Task assignments

2. **`uncertainties_YYYYMMDD_HHMMSS.json`**
   - FESTA uncertainty scores (U_FES, U_FCS, U_FESTA)
   - Baseline uncertainty scores (OE, RU)
   - One value per sample

3. **`metrics_YYYYMMDD_HHMMSS.json`**
   - Overall accuracy
   - AUROC scores for each method
   - Task-wise performance
   - Method comparisons

### Viewing Results

**Option 1: In notebook** (Cell 9)
- View summary statistics
- See AUROC scores
- Check overall accuracy

**Option 2: In Google Drive**
- Navigate to `My Drive/festa_results/`
- Download JSON files
- Open in JSON viewer or Python

**Option 3: Download as ZIP** (Cell 10)
- Downloads all results to your computer
- Convenient for offline analysis

---

## Configuration Details

The experiment uses `config_colab_full.yaml` with these settings:

### Dataset
- **Samples per task:** 100 (300 total)
- **Tasks:** count, order, duration
- **Location:** `/content/drive/MyDrive/TREA_dataset`

### FESTA Parameters (Optimized for Colab)
- **FES samples:** 5 audio × 2 text = 10 per sample
- **FCS samples:** 5 audio × 2 text = 10 per sample
- **Total augmentations per sample:** 20 (vs 120 in full config)

**Why reduced?**
- Balances accuracy vs runtime
- Prevents GPU memory issues
- Still captures uncertainty effectively
- Reduces 300-sample runtime from 15+ hours to 5-8 hours

### Baselines Enabled
- ✅ Output Entropy (10 samples)
- ✅ Rephrase Uncertainty (3 rephrases)
- ❌ Verbalized Confidence (disabled)
- ❌ Input Augmentation (disabled)
- ❌ Blackbox Uncertainty (disabled)

---

## Troubleshooting

### "TREA_dataset not found in Google Drive"

**Solution:**
- Verify dataset is uploaded to **root** of My Drive
- Path should be: `My Drive/TREA_dataset/`
- NOT: `My Drive/SomeFolder/TREA_dataset/`
- Re-run Cell 3 to verify

### "Out of memory" or "CUDA out of memory"

**Solutions:**
1. Restart runtime: Runtime → Restart runtime
2. Clear outputs: Edit → Clear all outputs
3. Reduce samples in config (edit Cell 5):
   - Change `samples_per_task: 50` (instead of 100)
4. Request High-RAM runtime (if available)

### "No GPU available"

**Solution:**
- Go to Runtime → Change runtime type
- Set Hardware accelerator to "GPU"
- Click Save
- Re-run cells from beginning

### Session timed out / disconnected

**Solution:**
- This is normal for long runs (>2 hours on free Colab)
- Simply reopen notebook and re-run cells 1-7
- Checkpoint system will resume automatically
- Consider Colab Pro for longer sessions

### Experiment running very slowly

**Expected behavior:**
- 1-2 minutes per sample
- 300 samples = 5-8 hours total
- This is normal!

**Tips:**
- Let it run overnight
- Use Colab Pro for background execution
- Monitor with Cell 8 periodically

### Repository clone failed (Cell 4)

**Solution Option 1:**
- Update GitHub URL to your actual repository
- Make sure repository is public

**Solution Option 2:**
- Upload entire AudioLLM-FESTA folder to Google Drive
- Modify Cell 4 to copy from Drive instead:
  ```python
  !cp -r /content/drive/MyDrive/AudioLLM-FESTA /content/
  %cd /content/AudioLLM-FESTA
  ```

---

## Expected Timeline

### Setup (one-time): ~30-60 minutes
- Upload dataset to Drive: 10-30 min
- Upload notebook: 1 min
- First-time cell execution: 5-10 min

### Experiment Runtime: ~5-8 hours
- 300 samples × 1.5 min average = ~450 min = 7.5 hours
- Faster with better GPU (T4 vs A100)
- Slower if system is under load

### Total first run: ~6-9 hours

### Subsequent runs:
- Resume from checkpoint: Variable (depends on how much was completed)
- Rerun from scratch: Delete checkpoint file first

---

## Best Practices

### ✅ Do:
- Upload dataset to Drive once, reuse forever
- Let experiment run uninterrupted if possible
- Check Cell 6 before running to see existing progress
- Save/bookmark your Colab notebook
- Use Colab Pro for long experiments (optional)

### ❌ Don't:
- Don't delete checkpoint file while experiment is running
- Don't run multiple experiments simultaneously (conflicts)
- Don't close browser tab if on free Colab (may disconnect)
- Don't modify config during experiment

---

## File Locations Summary

| File | Location | Purpose |
|------|----------|---------|
| Dataset | `/content/drive/MyDrive/TREA_dataset/` | Audio files and CSVs |
| Checkpoint | `/content/drive/MyDrive/festa_checkpoint.json` | Resume capability |
| Results | `/content/drive/MyDrive/festa_results/` | All output files |
| Config | `/content/AudioLLM-FESTA/config_colab_full.yaml` | Experiment settings |
| Code | `/content/AudioLLM-FESTA/` | FESTA implementation |

---

## Questions?

### How do I know it's working?
- Cell 7 shows logs for each sample
- Cell 8 shows live progress bar
- Checkpoint file updates in Drive

### Can I run test first?
Yes! Edit config in Cell 5:
```python
'colab': {
    'test_mode': True  # Process only 1 sample
}
```

### How much does this cost?
- **Free Colab:** $0 (but may disconnect after 2-4 hours)
- **Colab Pro:** ~$10/month (longer sessions, better GPUs)
- Recommended: Try free first, upgrade if needed

### Can I analyze results later?
Yes! Results are saved in Google Drive. You can:
- Download anytime
- Analyze in separate notebook
- Share with collaborators

---

## Summary

1. ✅ Upload TREA_dataset to Google Drive (once)
2. ✅ Open festa_colab_full.ipynb in Colab
3. ✅ Set runtime to GPU
4. ✅ Run cells 1-7 in order
5. ✅ Wait ~5-8 hours (or resume after timeout)
6. ✅ View results in Cell 9 or download in Cell 10

**You're all set! Happy experimenting! 🎉**
