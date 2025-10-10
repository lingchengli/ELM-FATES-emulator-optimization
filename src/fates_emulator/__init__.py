"""
FATES-Emulator: Machine Learning Framework for Ecosystem Model Calibration

This package provides tools for calibrating FATES (Functionally Assembled 
Terrestrial Ecosystem Simulator) using XGBoost machine learning emulators.

Main modules:
    - sampling: Parameter sampling and space exploration
    - emulator: XGBoost model training and prediction
    - calibration: Parameter optimization
    - diagnostics: Model evaluation and SHAP analysis
    - sensitivity: Sensitivity analysis tools
"""

__version__ = "0.1.0"
__author__ = "FATES-Emulator Contributors"

from . import sampling
from . import emulator
from . import calibration
from . import diagnostics
from . import sensitivity
from . import utils

__all__ = [
    "sampling",
    "emulator",
    "calibration",
    "diagnostics",
    "sensitivity",
    "utils",
]

