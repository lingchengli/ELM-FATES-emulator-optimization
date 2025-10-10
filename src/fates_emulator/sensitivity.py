"""
Sensitivity analysis wrapper

Simple wrapper around SALib for FATES sensitivity analysis.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from SALib.analyze import sobol

logger = logging.getLogger(__name__)


def analyze_sensitivity_sobol(
    param_samples: pd.DataFrame,
    model_outputs: pd.DataFrame,
    param_names: List[str],
    param_bounds: List[List[float]],
    output_variable: str
) -> Dict:
    """
    Perform Sobol sensitivity analysis.
    
    Parameters
    ----------
    param_samples : pd.DataFrame
        Parameter samples
    model_outputs : pd.DataFrame
        Model outputs
    param_names : List[str]
        Parameter names
    param_bounds : List[List[float]]
        Parameter bounds
    output_variable : str
        Output variable to analyze
        
    Returns
    -------
    Dict
        Sensitivity indices (S1, ST, S2)
    """
    logger.info(f"Sobol sensitivity analysis for {output_variable}")
    
    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': param_bounds
    }
    
    Y = model_outputs[output_variable].values
    
    Si = sobol.analyze(problem, Y)
    
    # Convert to DataFrame for easier handling
    results = {
        'parameter': param_names,
        'S1': Si['S1'],
        'S1_conf': Si['S1_conf'],
        'ST': Si['ST'],
        'ST_conf': Si['ST_conf']
    }
    
    df_sensitivity = pd.DataFrame(results)
    df_sensitivity = df_sensitivity.sort_values('ST', ascending=False)
    
    logger.info(f"Top 3 sensitive parameters:")
    for i, row in df_sensitivity.head(3).iterrows():
        logger.info(f"  {row['parameter']}: ST = {row['ST']:.4f}")
    
    return {
        'sensitivity_indices': df_sensitivity,
        'raw_indices': Si
    }


if __name__ == "__main__":
    print("Sensitivity Analysis Module")
    print("For full Sobol analysis, use SALib directly with your samples.")

