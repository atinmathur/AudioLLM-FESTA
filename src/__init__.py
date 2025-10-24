"""
FESTA: Functionally Equivalent Sampling for Trust Assessment
Implementation for Qwen Audio LLM on TREA Dataset
"""

__version__ = "0.1.0"
__author__ = "FESTA Implementation Team"

from .data_loader import TREADataset, TREADataLoader
from .model_wrapper import Qwen2AudioWrapper
from .fes_generator import FESGenerator
from .fcs_generator import FCSGenerator
from .uncertainty import FESTAUncertainty
from .metrics import compute_auroc, evaluate_selective_prediction
from .baselines import BaselineUncertainty

__all__ = [
    "TREADataset",
    "TREADataLoader",
    "Qwen2AudioWrapper",
    "FESGenerator",
    "FCSGenerator",
    "FESTAUncertainty",
    "compute_auroc",
    "evaluate_selective_prediction",
    "BaselineUncertainty",
]
