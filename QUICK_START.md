# FESTA Quick Start Guide

This guide will help you get started with the FESTA implementation quickly.

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU with 14GB+ VRAM (for full model) or CPU
- ~20GB disk space for model and data

## 🚀 Installation (5 minutes)

### Step 1: Set up environment

```bash
# Navigate to project directory
cd AudioLLM-FESTA

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# This will install:
# - PyTorch (for model inference)
# - Transformers (for Qwen2-Audio)
# - Audio libraries (librosa, soundfile)
# - Data processing (pandas, numpy)
# - Visualization (matplotlib, seaborn)
```

### Step 3: Verify installation

```bash
# Test data loader
python src/data_loader.py

# Test audio processing
python src/utils.py

# Test uncertainty computation
python src/uncertainty.py
```

Expected output: "test completed successfully!" for each script.

---

## 🧪 Quick Test (Without Model Download)

Test all components without downloading the 14GB model:

```bash
# Test individual components
cd src

# 1. Test data loading
python data_loader.py
# ✓ Should load 30 samples per task

# 2. Test FES generator
python fes_generator.py
# ✓ Should generate equivalent samples

# 3. Test FCS generator
python fcs_generator.py
# ✓ Should generate complementary samples

# 4. Test uncertainty computation
python uncertainty.py
# ✓ Should compute FESTA scores

# 5. Test metrics
python metrics.py
# ✓ Should compute AUROC
```

---

## 📊 Run with Sample Data (10-15 minutes)

### Option 1: Jupyter Notebooks (Recommended for Learning)

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_model_testing.ipynb  (requires model)
# 3. notebooks/03_festa_evaluation.ipynb  (requires model)
```

### Option 2: Command Line (For Full Experiments)

```bash
# Edit config.yaml to set sample size
# Change samples_per_task: 10  (for quick test)

# Run FESTA experiment
cd experiments
python run_festa.py --config ../config.yaml
```

---

## 💻 Running Options

### A. GPU with CUDA (Fastest - 14GB VRAM needed)

```yaml
# config.yaml
model:
  device: "cuda"
  dtype: "float16"
```

**Run time**: ~2-3 minutes per sample with FES/FCS

### B. CPU Only (Slowest - No GPU required)

```yaml
# config.yaml
model:
  device: "cpu"
  dtype: "float32"
```

**Run time**: ~10-15 minutes per sample with FES/FCS

### C. Cloud GPU (Google Colab / Kaggle)

1. Upload project to Colab/Kaggle
2. Install dependencies
3. Run experiments with GPU runtime

---

## 📝 Step-by-Step: Your First FESTA Experiment

### 1. Start Small (5 samples per task)

```bash
# Edit config.yaml
nano config.yaml  # or your preferred editor
```

```yaml
dataset:
  samples_per_task: 5  # Start with just 5 samples

festa:
  n_fes_audio: 5  # Fewer samples for speed
  n_fes_text: 2
  n_fcs_audio: 5
  n_fcs_text: 2
```

### 2. Run the experiment

```bash
cd experiments
python run_festa.py
```

### 3. Check results

```bash
# Results saved in results/ directory
ls -lh ../results/

# View metrics
cat ../results/metrics_*.json
```

### 4. Scale Up

Once you verify it works:

```yaml
dataset:
  samples_per_task: 30  # Full subset

festa:
  n_fes_audio: 15  # As in paper
  n_fes_text: 4
  n_fcs_audio: 15
  n_fcs_text: 4
```

---

## 🎯 Expected Results

After running on 30 samples per task (90 total):

```
Overall Accuracy: 42-52%

Method             AUROC
FESTA              0.83-0.89  ← Best
Output Entropy     0.63-0.71
Rephrase Unc.      0.62-0.68

Task Performance:
- Order:    AUROC ~0.91, Acc ~52%
- Duration: AUROC ~0.75, Acc ~43%
- Count:    AUROC ~0.83, Acc ~30%
```

---

## 🐛 Troubleshooting

### Issue 1: Model download fails

```bash
# Pre-download model using Hugging Face CLI
pip install huggingface_hub
huggingface-cli login
huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct
```

### Issue 2: GPU out of memory

```yaml
# Switch to CPU mode
model:
  device: "cpu"
  dtype: "float32"
```

Or reduce sample sizes:

```yaml
festa:
  n_fes_audio: 5  # Reduce from 15
  n_fes_text: 2   # Reduce from 4
```

### Issue 3: Audio processing errors

```bash
# Install ffmpeg (required by pydub)
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt-get install ffmpeg

# Windows:
# Download from https://ffmpeg.org/download.html
```

### Issue 4: librosa warnings

```bash
# Install additional audio dependencies
pip install audioread soundfile

# On macOS, may need:
brew install libsndfile
```

---

## 📚 Next Steps

### For Understanding:

1. **Read the paper**: Open `FESTA.pdf` to understand the methodology
2. **Explore data**: Run `01_data_exploration.ipynb`
3. **Test model**: Run `02_model_testing.ipynb`

### For Research:

1. **Baseline evaluation**: Run full experiments with 30 samples
2. **Analyze results**: Use `03_festa_evaluation.ipynb`
3. **Novel extensions**: Modify FES/FCS generators in `src/`

### For Novelty:

1. **Custom transformations**: Edit `fes_generator.py` and `fcs_generator.py`
2. **New tasks**: Extend to other audio reasoning tasks
3. **Different models**: Try other audio LLMs
4. **Hybrid methods**: Combine FESTA with other uncertainty measures

---

## 💡 Pro Tips

1. **Start small**: Always test with 5 samples before running on 30
2. **Use notebooks**: Better for understanding and debugging
3. **Monitor GPU**: Use `nvidia-smi` to check memory usage
4. **Save often**: Enable `save_intermediate: true` in config
5. **Compare baselines**: Run all baseline methods for comparison

---

## 📞 Getting Help

If you encounter issues:

1. Check the main `README.md` for detailed documentation
2. Review error messages carefully
3. Verify all dependencies are installed correctly
4. Test individual components before full pipeline
5. Check GPU memory if using CUDA

---

## ✅ Quick Checklist

Before running full experiments:

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list` shows torch, transformers, etc.)
- [ ] TREA dataset present in `TREA_dataset/`
- [ ] Individual component tests pass
- [ ] Config file set to small sample size (5 per task)
- [ ] GPU/CPU settings correct in config
- [ ] Disk space available (~20GB for model + results)

---

**Ready to start? Follow the steps above and you'll have FESTA running in 15 minutes!**

For detailed documentation, see `README.md`.
