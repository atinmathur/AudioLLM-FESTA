# Running FESTA on Google Colab

This guide provides step-by-step instructions for running the FESTA framework on Google Colab.

## Why Use Colab?

- **Free GPU access** (T4 GPU with 15GB VRAM)
- **No local setup** required
- **Cloud storage** for experiments
- **Easy sharing** of results

---

## ⚡ IMPORTANT: Use Mini Dataset for Fast Upload!

**Problem**: The full TREA dataset is **908MB** (620 audio files) and takes 20-30+ minutes to upload to Colab.

**Solution**: Use the **mini dataset** - only 15 audio files (~25MB total package) for **97% size reduction**!

### Quick Setup (2 minutes):

```bash
# On your local machine, in AudioLLM-FESTA directory:

# Step 1: Generate mini dataset
python scripts/create_mini_dataset.py

# Step 2: Package for Colab
python scripts/package_for_colab.py

# Output: AudioLLM-FESTA-colab.zip (~25MB)
```

### Upload to Colab:
- Upload `AudioLLM-FESTA-colab.zip` (1-2 min upload ⚡)
- vs full dataset (30+ min upload ⏰)

### What's Included:
- ✅ All code and configurations
- ✅ 15 audio samples (5 per task)
- ✅ Full FESTA pipeline works identically
- ✅ Perfect for quick testing
- ⚠️ Small sample size (use for verification, not final evaluation)

**See detailed instructions**: `scripts/README_MINI_DATASET.md`

**For first-time users**: Use mini dataset! You can scale up later with Google Drive.

---

## Prerequisites

### What You Need:
1. Google account (for Colab access)
2. Your local AudioLLM-FESTA folder
3. TREA_dataset folder
4. **Optional**: Google Drive for persistent storage

### Time Requirements:
- **Setup**: ~5-10 minutes (first time)
- **Quick test** (5 samples/task): ~10-15 minutes
- **Full experiment** (30 samples/task): ~1-2 hours

---

## Step-by-Step Instructions

### Step 1: Prepare Files Locally

**⚡ Recommended: Use Mini Dataset Scripts (FASTEST)**

```bash
# Navigate to AudioLLM-FESTA directory
cd /path/to/AudioLLM-FESTA/

# Generate mini dataset (~30 seconds)
python scripts/create_mini_dataset.py

# Package for Colab (~10 seconds)
python scripts/package_for_colab.py

# Output: AudioLLM-FESTA-colab.zip (~25MB)
```

**Why this is better:**
- ✅ Automatically excludes full dataset (saves 900MB!)
- ✅ Includes only necessary files
- ✅ Creates optimized structure
- ✅ Fast upload (1-2 minutes vs 30+ minutes)

---

**Alternative: Manual Zip (NOT Recommended)**

⚠️ **Problem**: If you just zip `AudioLLM-FESTA/`, it includes the 908MB dataset inside!

```bash
# DON'T DO THIS (includes huge dataset):
# zip -r AudioLLM-FESTA.zip AudioLLM-FESTA/  ❌

# If you must zip manually, exclude dataset:
cd /path/to/parent/directory/
zip -r AudioLLM-FESTA.zip AudioLLM-FESTA/ -x "AudioLLM-FESTA/TREA_dataset/*"
```

**Expected sizes:**
- With mini dataset script: ~25MB (AudioLLM-FESTA-colab.zip) ✅
- Manual code-only: ~50KB (without any dataset) ⚠️
- Full dataset included: ~900MB+ (very slow!) ❌

---

### Step 2: Open Google Colab

1. Go to **https://colab.research.google.com**
2. Sign in with your Google account
3. Click **File → Upload notebook**
4. Upload `notebooks/colab_festa.ipynb` from your local machine

**Alternative**: If you pushed to GitHub:
```
File → Open notebook → GitHub → Enter your repo URL
```

---

### Step 3: Enable GPU Runtime

**IMPORTANT**: Colab uses CPU by default. You MUST enable GPU:

1. Click **Runtime** in the menu bar
2. Select **Change runtime type**
3. Hardware accelerator: **GPU**
4. GPU type: **T4** (free tier) or **V100/A100** (Pro)
5. Click **Save**

**Verify GPU is enabled:**
```python
# Run this in a cell:
!nvidia-smi
```

You should see GPU information displayed.

---

### Step 4: Install Dependencies

Run the setup cells in the notebook sequentially:

#### Cell 1: Check GPU
```python
!nvidia-smi
```
✅ **Expected**: GPU information displayed
❌ **If no GPU**: Go back to Step 3

#### Cell 2: Install system packages
```python
!apt-get update -qq
!apt-get install -y -qq ffmpeg libsndfile1
```
⏱️ **Time**: ~30 seconds

#### Cell 3: Install Python packages
```python
!pip install -q torch transformers librosa soundfile ...
```
⏱️ **Time**: ~2-3 minutes

#### Cell 4: Verify installations
```python
import torch, transformers, librosa, ...
```
✅ **Expected**: All packages show ✅
❌ **If failures**: Re-run installation cell

---

### Step 5: Upload Your Files

#### Cell 5: Upload files
```python
from google.colab import files
uploaded = files.upload()
```

**What to upload:**
1. **AudioLLM-FESTA.zip** - Your project code
2. **TREA_dataset.zip** - Your dataset (if separate)

**OR** upload individual folders (slower):
- Just drag and drop into Colab file browser on the left

⏱️ **Time**: Depends on internet speed (2-5 minutes typical)

#### Cell 6: Extract files
```python
# Automatically extracts zip files
```
✅ **Expected**: Files extracted to `/content/`

#### Cell 7: Verify structure
```python
# Checks all required directories exist
```
✅ **Expected**: All paths show ✅
❌ **If missing**: Check your folder structure matches:
```
/content/AudioLLM-FESTA/
├── src/
├── experiments/
├── notebooks/
├── TREA_dataset/
│   ├── count/
│   ├── order/
│   ├── duration/
│   └── synthetic_silences/
├── config_colab.yaml
└── ...
```

---

### Step 6: Configure and Test

#### Cell 8-9: Load configuration
```python
os.chdir('/content/AudioLLM-FESTA')
# Displays current configuration
```

**Default configuration (Quick Test):**
- Samples per task: **5** (15 total)
- FES samples: **5 × 2 = 10** (reduced from 60)
- FCS samples: **5 × 2 = 10** (reduced from 60)
- Estimated time: **~10-15 minutes**

**To modify** (optional):
```python
# For even faster test (1 sample only):
config['colab']['test_mode'] = True

# For full experiment (90 samples):
config['dataset']['samples_per_task'] = 30
config['festa']['n_fes_audio'] = 15
config['festa']['n_fes_text'] = 4
```

#### Cell 10: Test imports
```python
from src.data_loader import load_trea_dataset
# ... tests all imports
```
✅ **Expected**: All modules show ✅

---

### Step 7: Load Model

#### Cell 11: Check disk space
```python
!df -h /content
```
⚠️ **Required**: ~14GB free space for model

#### Cell 12: Load Qwen2-Audio model
```python
model = Qwen2AudioWrapper(...)
```

**⏱️ First time**: ~5-10 minutes (downloads ~14GB model)
**⏱️ Subsequent runs**: ~1-2 minutes (cached)

**Progress indicators:**
- Download progress bar
- Model loading messages
- "✅ Model loaded successfully!"

#### Cell 13: Test prediction
```python
# Tests model on one sample
```
✅ **Expected**:
- Prediction shown
- Probabilities displayed
- "✅ Model test passed!"

---

### Step 8: Run FESTA Experiment

#### Cell 14: Run main experiment
```python
!python experiments/run_festa_colab.py --config config_colab.yaml
```

**What happens:**
1. Loads dataset (5 samples × 3 tasks = 15 samples)
2. For each sample:
   - Gets original prediction
   - Generates 10 FES samples
   - Generates 10 FCS samples
   - Computes FESTA uncertainty
   - Computes baseline uncertainties
   - Saves checkpoint
3. Computes metrics (AUROC, accuracy)
4. Saves results

**Progress tracking:**
```
Processing sample 1/15
1️⃣ Getting original prediction...
2️⃣ Generating FES samples...
3️⃣ Getting FES predictions...
   FES: 100%|██████████| 10/10
4️⃣ Generating FCS samples...
5️⃣ Getting FCS predictions...
   FCS: 100%|██████████| 10/10
6️⃣ Computing FESTA uncertainty...
   U_FES: 0.1234, U_FCS: 0.5678, U_FESTA: 0.6912
✅ Sample 1 completed successfully
💾 Intermediate results saved
```

**⏱️ Time**:
- Quick test (5/task): ~10-15 minutes
- Full experiment (30/task): ~1-2 hours

---

### Step 9: Handle Session Timeouts

**If Colab disconnects** (timeout after ~90 minutes of inactivity):

1. **Don't panic!** Progress is saved via checkpoints
2. Reconnect to Colab
3. Re-run **only** Step 4 (install packages) and Step 7 (load model)
4. Re-run Cell 14 (experiment)
   - It will automatically resume from the last completed sample
   - You'll see: `📂 Checkpoint loaded: X samples already completed`

**Checkpoint location**: `/content/festa_checkpoint.json`

**To check progress:**
```python
import json
with open('/content/festa_checkpoint.json', 'r') as f:
    checkpoint = json.load(f)
print(f"Completed: {checkpoint['progress_percent']:.1f}%")
```

---

### Step 10: View Results

#### Cell 15-17: Display results
```python
# Lists result files
# Shows metrics summary
# Creates visualization plots
```

**Expected outputs:**
```
📊 FESTA Results Summary

Overall Accuracy: 42-52%

Method Comparison:
──────────────────────────────────────────────────
FESTA                AUROC: 0.8300-0.9100  ⭐ BEST
OE                   AUROC: 0.6300-0.7100
RU                   AUROC: 0.6200-0.6800
```

**Visualization:**
- Bar chart comparing AUROC scores
- Saved as `auroc_comparison.png`

---

### Step 11: Download Results

#### Cell 18: Package results
```python
# Creates zip file with all results
```

**Contents:**
- `predictions_YYYYMMDD_HHMMSS.json`
- `uncertainties_YYYYMMDD_HHMMSS.json`
- `metrics_YYYYMMDD_HHMMSS.json`
- `auroc_comparison.png`
- `intermediate_results.json`

#### Cell 19: Download
```python
files.download(zip_path)
```

✅ **Check your browser's Downloads folder** for `festa_results_YYYYMMDD_HHMMSS.zip`

---

## Configuration Options

### Quick Test (Default)
```yaml
dataset:
  samples_per_task: 5  # 15 total samples
festa:
  n_fes_audio: 5
  n_fes_text: 2
  n_fcs_audio: 5
  n_fcs_text: 2
```
⏱️ **Time**: ~10-15 minutes

### Medium Experiment
```yaml
dataset:
  samples_per_task: 10  # 30 total samples
festa:
  n_fes_audio: 10
  n_fes_text: 3
  n_fcs_audio: 10
  n_fcs_text: 3
```
⏱️ **Time**: ~20-30 minutes

### Full Experiment (Paper Settings)
```yaml
dataset:
  samples_per_task: 30  # 90 total samples
festa:
  n_fes_audio: 15
  n_fes_text: 4
  n_fcs_audio: 15
  n_fcs_text: 4
```
⏱️ **Time**: ~1-2 hours
💡 **Tip**: Consider Colab Pro for longer sessions

---

## Troubleshooting

### Problem: "No GPU detected"
**Solution**:
1. Runtime → Change runtime type → GPU
2. Runtime → Restart runtime
3. Re-run setup cells

### Problem: "Out of memory"
**Solutions**:
1. Reduce `samples_per_task` to 3
2. Reduce `n_fes_audio` and `n_fcs_audio` to 3
3. Set `hardware.clear_cache: true` in config
4. Restart runtime and try again

### Problem: "Model download fails"
**Solutions**:
1. Check internet connection
2. Verify disk space: `!df -h`
3. Try downloading manually:
   ```python
   from transformers import AutoModel
   AutoModel.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
   ```

### Problem: "Audio files not found"
**Solutions**:
1. Verify TREA_dataset structure:
   ```python
   !tree -L 2 TREA_dataset
   ```
2. Check paths in config_colab.yaml
3. Re-upload dataset

### Problem: "Session timeout before completion"
**Solution**:
1. **Automatic resume**: Just re-run the experiment cell
2. **Manual resume**:
   ```python
   !python experiments/run_festa_colab.py --config config_colab.yaml
   # Loads checkpoint automatically
   ```

### Problem: "Import errors"
**Solutions**:
1. Re-run installation cell
2. Check file upload completed
3. Verify file structure
4. Restart runtime

---

## Best Practices

### 1. Start Small
- First run: Use test mode (1 sample)
  ```python
  config['colab']['test_mode'] = True
  ```
- Second run: Quick test (5 samples/task)
- Then: Scale up gradually

### 2. Monitor Progress
- Watch for checkpoint saves: `💾 Intermediate results saved`
- Check GPU memory: Displayed after each sample
- Monitor completion percentage

### 3. Save to Google Drive (Optional)
For persistent storage across sessions:

```python
from google.colab import drive
drive.mount('/content/drive')

# Update output paths in config
config['experiment']['output_dir'] = '/content/drive/MyDrive/festa_results'
config['experiment']['checkpoint_file'] = '/content/drive/MyDrive/festa_checkpoint.json'
```

### 4. Optimize for Speed
- Disable unnecessary baselines in config
- Use lower FES/FCS sample counts for testing
- Process fewer samples per task initially

### 5. Handle Long Experiments
For experiments >1 hour:
- Consider **Colab Pro** ($10/month)
  - Longer runtimes (up to 24 hours)
  - Faster GPUs (V100, A100)
  - More memory
- Or split experiment:
  ```python
  # Run count task only
  config['dataset']['tasks'] = ['count']
  config['dataset']['samples_per_task'] = 30
  ```

---

## Expected Results

Based on FESTA paper (Table 2):

### Quick Test (5 samples/task)
- Results may vary due to small sample size
- Use for verification, not evaluation

### Full Experiment (30 samples/task)
| Task     | Accuracy | FESTA AUROC | Best Baseline | Improvement |
|----------|----------|-------------|---------------|-------------|
| Order    | ~52%     | **0.91**    | 0.70          | **+30.0%**  |
| Duration | ~43%     | **0.75**    | 0.59          | **+27.1%**  |
| Count    | ~30%     | **0.83**    | 0.58          | **+43.1%**  |
| **Overall** | **42%** | **0.83** | **0.62** | **+33.9%** |

---

## Next Steps

After successful Colab run:

1. **Analyze Results**
   - Open downloaded results in local Jupyter
   - Use notebooks/03_festa_evaluation.ipynb
   - Generate additional visualizations

2. **Scale Up**
   - Increase samples_per_task to 30
   - Use full FES/FCS settings (15×4)
   - Enable all baseline methods

3. **Explore Novelty**
   - Modify FES/FCS generators
   - Try different transformations
   - Experiment with other models

4. **Local Setup**
   - If you have GPU, set up locally
   - Use for faster iteration
   - Better for development

---

## Colab Limitations

### Free Tier
- **Runtime**: ~90 minutes idle timeout, 12 hours max
- **GPU**: T4 (15GB VRAM)
- **RAM**: 12-13 GB
- **Disk**: ~78 GB
- **Disconnect**: Sessions may disconnect randomly

### Colab Pro ($10/month)
- **Runtime**: Up to 24 hours
- **GPU**: V100 or A100 options
- **RAM**: Up to 32 GB
- **Priority**: Faster resource allocation

### Recommendations
- **Quick tests**: Free tier is sufficient
- **Full experiments**: Consider Pro for convenience
- **Repeated use**: Consider local setup with GPU

---

## Support

If you encounter issues:

1. **Check this guide** first
2. **Review error messages** carefully
3. **Verify all steps** completed correctly
4. **Try troubleshooting** section above
5. **Restart runtime** and try again

**For code issues:**
- Check main README.md
- Review QUICK_START.md
- Examine error logs in Colab

---

## Summary Checklist

Before starting:
- [ ] Google account ready
- [ ] Files zipped and ready to upload
- [ ] Understand time requirements
- [ ] Know your experiment size (quick test or full)

During setup:
- [ ] GPU enabled (nvidia-smi works)
- [ ] All packages installed (all show ✅)
- [ ] Files uploaded and extracted
- [ ] Directory structure verified
- [ ] Model loaded successfully
- [ ] Test prediction passed

During experiment:
- [ ] Monitor progress messages
- [ ] Watch for checkpoints
- [ ] Note any errors
- [ ] Keep Colab tab active (prevent disconnect)

After completion:
- [ ] Results displayed correctly
- [ ] Metrics make sense
- [ ] Plots generated
- [ ] Results downloaded
- [ ] Files backed up

---

**Ready to run FESTA on Colab? Start with notebooks/colab_festa.ipynb!** 🚀
