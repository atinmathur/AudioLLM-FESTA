# FESTA: Functionally Equivalent Sampling for Trust Assessment
## Implementation for Qwen Audio LLM on TREA Dataset

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## 📖 Overview

This repository contains a complete implementation of **FESTA** (Functionally Equivalent Sampling for Trust Assessment) for uncertainty quantification in audio large language models, specifically targeting **Qwen2-Audio-7B-Instruct** on the **TREA** (Temporal Reasoning Evaluation of Audio) dataset.

FESTA is a black-box, unsupervised uncertainty estimation method that uses two types of input sampling:
- **FES (Functional Equivalent Sampling)**: Generates task-preserving transformations to measure model consistency
- **FCS (Functional Complementary Sampling)**: Generates task-equivalent but functionally divergent transformations to measure model sensitivity

### Key Features

✅ **End-to-end FESTA implementation** with modular, reusable components
✅ **Qwen2-Audio-7B-Instruct** model integration
✅ **TREA dataset** support for temporal reasoning (Count, Order, Duration tasks)
✅ **Comprehensive baselines** (Output Entropy, Verbalized Confidence, Rephrase Uncertainty, etc.)
✅ **Evaluation metrics** (AUROC, Selective Prediction, Task-wise analysis)
✅ **GPU/CPU compatibility** with both local and cloud environments
✅ **Configurable experiments** via YAML configuration
✅ **Jupyter notebooks** for exploration and visualization

---

## 🏗️ Project Structure

```
AudioLLM-FESTA/
├── FESTA.pdf                    # Original research paper
├── TREA_dataset/                # TREA dataset
│   ├── count/                   # Count task data
│   ├── order/                   # Order task data
│   ├── duration/                # Duration task data
│   └── synthetic_silences/      # Synthetic audio events
├── src/                         # Source code
│   ├── __init__.py
│   ├── data_loader.py           # TREA dataset loader
│   ├── model_wrapper.py         # Qwen2-Audio wrapper
│   ├── fes_generator.py         # FES sample generator
│   ├── fcs_generator.py         # FCS sample generator
│   ├── uncertainty.py           # FESTA uncertainty computation
│   ├── metrics.py               # Evaluation metrics
│   ├── baselines.py             # Baseline methods
│   └── utils.py                 # Audio processing utilities
├── experiments/
│   └── run_festa.py             # Main experiment script
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_testing.ipynb
│   └── 03_festa_evaluation.ipynb
├── results/                     # Output directory
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd AudioLLM-FESTA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config.yaml` to customize experiment settings:

```yaml
# Key settings
dataset:
  samples_per_task: 30  # Number of samples per task (count, order, duration)

model:
  name: "Qwen/Qwen2-Audio-7B-Instruct"
  device: "cuda"  # or "cpu"
  dtype: "float16"

festa:
  n_fes_audio: 15  # Number of audio FES transformations
  n_fes_text: 4    # Number of text paraphrases
  n_fcs_audio: 15  # Number of audio FCS transformations
  n_fcs_text: 4    # Number of text complements
```

### 3. Run FESTA Experiment

```bash
# Run the main experiment
cd experiments
python run_festa.py --config ../config.yaml
```

This will:
1. Load 30 samples per task (90 total samples)
2. Generate FES and FCS samples for each input
3. Get model predictions on all samples
4. Compute FESTA uncertainty scores
5. Evaluate with AUROC and other metrics
6. Save results to `results/` directory

### 4. Expected Output

```
================================================================================
FESTA Experiment Completed Successfully!
================================================================================

Overall Accuracy: 0.52

FESTA AUROC: 0.89
OE AUROC: 0.71
RU AUROC: 0.68

Method Comparison:
FESTA      : AUROC=0.8900 (Best)
OE         : AUROC=0.7100
RU         : AUROC=0.6800

Task-wise Performance:
  count    : AUROC=0.83, Acc=0.30
  order    : AUROC=0.91, Acc=0.51
  duration : AUROC=0.75, Acc=0.45
```

---

## 📊 FESTA Algorithm

### Algorithm 1: FESTA Uncertainty Estimator

```python
Input: X (audio + question), ŷ (original prediction), K (num samples)

# Step 1: FES Sampling
Generate K1 FES samples: {x̃₁, x̃₂, ..., x̃_K1}
Get predictions: q(y|x̃ₖ) for each FES sample
Compute q_FES(y|X) = E[q(y|x̃ₖ)]
Calculate U_FES = -log(q_FES(y=ŷ|X))

# Step 2: FCS Sampling
Generate K2 FCS samples: {x'₁, x'₂, ..., x'_K2}
Get predictions: q(y|x'ₖ) for each FCS sample
Compute q_FCS(y|X) = E[q(y|x'ₖ)]
Calculate U_FCS = -log(Σ_{y≠ŷ} q_FCS(y|X))

# Step 3: FESTA Score
U_FESTA = U_FES + U_FCS

Output: U_FESTA (higher = more uncertain)
```

### FES Transformations

**Audio:**
- Add silence between events (100-500ms)
- Volume adjustment (±10-20%)
- Add Gaussian noise (SNR: 30-40 dB)
- Normalization

**Text:**
- Question paraphrasing
- Semantic preservation
- Task objective unchanged

### FCS Transformations

**Task-Specific Audio:**
- **Count**: Add new sound events
- **Order**: Swap event positions
- **Duration**: Replace longest/shortest events

**Text Complementary:**
- **Order**: Reverse temporal relationships (first → last)
- **Duration**: Reverse comparisons (longest → shortest)
- **Count**: Maintain counting nature

---

## 🎯 Key Results (Expected)

Based on the FESTA paper, expected performance on TREA dataset:

| Task     | Accuracy | FESTA AUROC | Best Baseline | Improvement |
|----------|----------|-------------|---------------|-------------|
| Order    | ~0.52    | **0.91**    | 0.70          | +30.0%      |
| Duration | ~0.43    | **0.75**    | 0.59          | +27.1%      |
| Count    | ~0.30    | **0.83**    | 0.58          | +43.1%      |
| **Average** | **0.42** | **0.83** | **0.62** | **+33.9%** |

---

## 📈 Usage Examples

### Basic Usage

```python
from src.data_loader import load_trea_dataset
from src.model_wrapper import Qwen2AudioWrapper
from src.fes_generator import FESGenerator
from src.fcs_generator import FCSGenerator
from src.uncertainty import FESTAUncertainty

# Load data
dataset = load_trea_dataset(samples_per_task=30)

# Initialize model
model = Qwen2AudioWrapper(device="cuda")

# Initialize generators
fes_gen = FESGenerator(n_audio_samples=15, n_text_samples=4)
fcs_gen = FCSGenerator(n_audio_samples=15, n_text_samples=4)

# Process a sample
sample = dataset[0]

# Get original prediction
prediction, _ = model.predict(
    sample['audio_path'],
    sample['question'],
    sample['options']
)

# Generate FES samples
fes_samples = fes_gen.generate(
    sample['audio_path'],
    sample['question'],
    sample['task'],
    sample['options']
)

# Generate FCS samples
fcs_samples = fcs_gen.generate(
    sample['audio_path'],
    sample['question'],
    sample['task'],
    sample['options']
)

# Get predictions on samples
fes_preds = [model.predict(s['audio_path'], s['question'], s['options'])[0]
             for s in fes_samples]
fcs_preds = [model.predict(s['audio_path'], s['question'], s['options'])[0]
             for s in fcs_samples]

# Compute FESTA uncertainty
festa = FESTAUncertainty()
uncertainty_scores = festa.compute_festa(fes_preds, fcs_preds, prediction)

print(f"U_FES: {uncertainty_scores['U_FES']:.4f}")
print(f"U_FCS: {uncertainty_scores['U_FCS']:.4f}")
print(f"U_FESTA: {uncertainty_scores['U_FESTA']:.4f}")
```

### Testing Individual Components

```bash
# Test data loader
python src/data_loader.py

# Test model wrapper (requires model download)
python src/model_wrapper.py

# Test FES generator
python src/fes_generator.py

# Test FCS generator
python src/fcs_generator.py

# Test uncertainty computation
python src/uncertainty.py

# Test metrics
python src/metrics.py
```

---

## 🔧 Configuration Options

### Model Settings

```yaml
model:
  name: "Qwen/Qwen2-Audio-7B-Instruct"
  device: "cuda"  # or "cpu"
  dtype: "float16"  # or "float32"
  max_length: 512
```

### FESTA Parameters

```yaml
festa:
  n_fes_audio: 15  # Audio transformations (recommended: 10-20)
  n_fes_text: 4    # Text paraphrases (recommended: 3-5)
  n_fcs_audio: 15  # Complementary audio (recommended: 10-20)
  n_fcs_text: 4    # Complementary text (recommended: 3-5)
```

### Baseline Methods

```yaml
baselines:
  output_entropy:
    enabled: true
    num_samples: 20
  verbalized_confidence:
    enabled: true
  rephrase_uncertainty:
    enabled: true
    num_rephrases: 5
```

---

## 📊 Evaluation Metrics

### 1. AUROC (Area Under ROC Curve)

Measures ability to detect mispredictions:
- **1.0**: Perfect uncertainty calibration
- **0.5**: Random baseline
- **Higher is better**

### 2. Selective Prediction

Evaluate accuracy at different coverage levels:
- **Coverage**: % of samples retained
- **Selective Risk**: Error rate on retained samples

### 3. Task-wise Analysis

Per-task performance breakdown:
- Count task
- Order task
- Duration task

---

## 🧪 Jupyter Notebooks

### 1. Data Exploration (`01_data_exploration.ipynb`)
- Explore TREA dataset structure
- Analyze audio files
- Visualize question distributions

### 2. Model Testing (`02_model_testing.ipynb`)
- Test Qwen2-Audio inference
- Examine model predictions
- Analyze baseline accuracy

### 3. FESTA Evaluation (`03_festa_evaluation.ipynb`)
- Run FESTA on samples
- Compare with baselines
- Generate visualizations
- Ablation studies

---

## 🤝 Extending for Novelty

The modular design allows easy extensions:

### Custom Transformations

```python
# Add custom FES transformation
class CustomFESGenerator(FESGenerator):
    def _generate_audio_fes(self, audio_path, task):
        # Your custom transformations
        pass
```

### New Uncertainty Measures

```python
# Implement custom uncertainty
class CustomUncertainty:
    def compute(self, predictions, original):
        # Your uncertainty computation
        pass
```

### Different Models

```python
# Wrap a different audio LLM
class CustomAudioLLM:
    def predict(self, audio_path, question, options):
        # Your model inference
        pass
```

---

## 📝 Citation

If you use this implementation, please cite the FESTA paper:

```bibtex
@inproceedings{bhattacharya2025festa,
  title={FESTA: Functionally Equivalent Sampling for Trust Assessment of Multimodal LLMs},
  author={Bhattacharya, Debarpan and Kulkarni, Apoorva and Ganapathy, Sriram},
  booktitle={Findings of EMNLP},
  year={2025}
}
```

---

## 🐛 Troubleshooting

### Model Download Issues

If model download fails:
```bash
# Pre-download model
huggingface-cli login
huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct
```

### GPU Memory Issues

Reduce batch size or use CPU:
```yaml
model:
  device: "cpu"
  dtype: "float32"
```

### Audio Processing Errors

Ensure audio dependencies are installed:
```bash
pip install librosa soundfile pydub
```

---

## 📧 Support

For questions or issues:
- Check existing issues in the repository
- Review the FESTA paper for methodology details
- Consult the code documentation

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **FESTA Paper**: Bhattacharya et al., 2025
- **Qwen2-Audio**: Alibaba Cloud
- **TREA Dataset**: Bhattacharya et al., 2025
- **HuggingFace Transformers**: For model infrastructure

---

**Built with ❤️ for Audio LLM Uncertainty Quantification**
