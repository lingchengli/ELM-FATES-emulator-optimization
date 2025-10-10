"""
Parameter sampling for FATES sensitivity analysis

This module provides functions for generating parameter samples using various
sampling strategies (Latin Hypercube, Sobol, etc.) with ecological constraints.

All paths are configuration-based - no hardcoded directories.
"""

import numpy as np
import pandas as pd
from SALib.sample import latin, saltelli, fast_sampler
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_lhs_samples(
    param_names: List[str],
    bounds: List[List[float]],
    n_samples: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate Latin Hypercube Samples for parameter space exploration.
    
    Parameters
    ----------
    param_names : List[str]
        Names of parameters to sample
    bounds : List[List[float]]
        Parameter bounds as [[min1, max1], [min2, max2], ...]
    n_samples : int
        Number of samples to generate
    seed : Optional[int]
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with parameter samples
        
    Examples
    --------
    >>> params = generate_lhs_samples(
    ...     param_names=['vcmax', 'sla'],
    ...     bounds=[[40, 105], [0.005, 0.04]],
    ...     n_samples=100
    ... )
    """
    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': bounds
    }
    
    logger.info(f"Generating {n_samples} LHS samples for {len(param_names)} parameters")
    
    param_values = latin.sample(problem, n_samples, seed=seed)
    df_samples = pd.DataFrame(param_values, columns=param_names)
    
    logger.info(f"Generated {len(df_samples)} samples")
    
    return df_samples


def apply_ecological_constraints(
    df_params: pd.DataFrame,
    pft_suffix_early: str = '_e',
    pft_suffix_late: str = '_l',
    constraints: Optional[Dict[str, Tuple[str, float]]] = None
) -> pd.DataFrame:
    """
    Apply ecological constraints to parameter samples.
    
    Ensures parameter relationships follow ecological theory (e.g., early 
    successional species have higher Vcmax than late successional).
    
    Parameters
    ----------
    df_params : pd.DataFrame
        Parameter samples to filter
    pft_suffix_early : str
        Suffix for early successional PFT parameters (default '_e')
    pft_suffix_late : str
        Suffix for late successional PFT parameters (default '_l')
    constraints : Optional[Dict[str, Tuple[str, float]]]
        Custom constraints as {'param_base': ('operator', threshold)}
        operator can be '>', '<', '>=', '<='
        
    Returns
    -------
    pd.DataFrame
        Filtered parameter samples meeting constraints
        
    Examples
    --------
    >>> df = apply_ecological_constraints(
    ...     df_params,
    ...     constraints={'vcmax': ('>', 2.0)}  # early > late + 2
    ... )
    """
    logger.info(f"Applying ecological constraints to {len(df_params)} samples")
    
    initial_count = len(df_params)
    df_filtered = df_params.copy()
    
    # Default constraints for tropical forest PFTs
    if constraints is None:
        constraints = {
            'vcmax': ('>', 2.0),      # Early successional higher Vcmax
            'sla': ('>', -0.001),     # Early successional higher/similar SLA  
            'bmort': ('>', 0.001),    # Early successional higher mortality
            'wdens': ('<', -0.01),    # Early successional lower wood density
            'lflog': ('<', -0.05),    # Early successional shorter leaf life
        }
    
    # Apply constraints
    for param_base, (operator, threshold) in constraints.items():
        param_early = f'{param_base}{pft_suffix_early}'
        param_late = f'{param_base}{pft_suffix_late}'
        
        if param_early not in df_filtered.columns or param_late not in df_filtered.columns:
            logger.warning(f"Parameters {param_early} or {param_late} not found, skipping")
            continue
            
        if operator == '>':
            mask = df_filtered[param_early] > (df_filtered[param_late] + threshold)
        elif operator == '>=':
            mask = df_filtered[param_early] >= (df_filtered[param_late] + threshold)
        elif operator == '<':
            mask = df_filtered[param_early] < (df_filtered[param_late] + threshold)
        elif operator == '<=':
            mask = df_filtered[param_early] <= (df_filtered[param_late] + threshold)
        else:
            logger.warning(f"Unknown operator {operator}, skipping")
            continue
            
        df_filtered = df_filtered[mask]
        logger.info(f"After {param_base} constraint: {len(df_filtered)} samples remain")
    
    logger.info(f"Ecological constraints: {initial_count} → {len(df_filtered)} samples "
                f"({100*len(df_filtered)/initial_count:.1f}% retained)")
    
    return df_filtered


def add_derived_parameters(
    df_params: pd.DataFrame,
    pft_suffix_early: str = '_e',
    pft_suffix_late: str = '_l',
    suffix_diff: str = '_d'
) -> pd.DataFrame:
    """
    Add derived parameters (differences between PFTs) to sample dataframe.
    
    Parameters
    ----------
    df_params : pd.DataFrame
        Parameter samples
    pft_suffix_early : str
        Suffix for early successional PFT
    pft_suffix_late : str
        Suffix for late successional PFT
    suffix_diff : str
        Suffix for difference parameters
        
    Returns
    -------
    pd.DataFrame
        Parameter samples with derived difference parameters
        
    Examples
    --------
    >>> df = add_derived_parameters(df_params)
    >>> # Creates vcmax_d = vcmax_e - vcmax_l, etc.
    """
    df_out = df_params.copy()
    
    # Find parameter base names (parameters that have both _e and _l versions)
    param_bases = set()
    for col in df_params.columns:
        if col.endswith(pft_suffix_early):
            base = col[:-len(pft_suffix_early)]
            late_param = base + pft_suffix_late
            if late_param in df_params.columns:
                param_bases.add(base)
    
    # Add difference parameters
    for base in param_bases:
        early_param = base + pft_suffix_early
        late_param = base + pft_suffix_late
        diff_param = base + suffix_diff
        
        df_out[diff_param] = df_out[early_param] - df_out[late_param]
        logger.debug(f"Added derived parameter: {diff_param}")
    
    logger.info(f"Added {len(param_bases)} derived difference parameters")
    
    return df_out


def generate_fates_parameter_samples(
    n_samples: int = 1500,
    param_config: Optional[Dict] = None,
    seed: Optional[int] = 11,
    apply_constraints: bool = True,
    add_differences: bool = True,
    oversample_factor: float = 40.0
) -> pd.DataFrame:
    """
    Complete pipeline to generate FATES parameter samples with constraints.
    
    This is the main function for Step 1 of the workflow.
    
    Parameters
    ----------
    n_samples : int
        Target number of samples after constraints
    param_config : Optional[Dict]
        Configuration with 'names' and 'bounds'
    seed : Optional[int]
        Random seed
    apply_constraints : bool
        Whether to apply ecological constraints
    add_differences : bool
        Whether to add derived difference parameters
    oversample_factor : float
        Oversample by this factor to account for constraint filtering
        
    Returns
    -------
    pd.DataFrame
        Parameter samples ready for FATES simulations
        
    Examples
    --------
    >>> samples = generate_fates_parameter_samples(n_samples=1500)
    >>> samples.to_csv('parameter_samples.csv')
    """
    # Default FATES parameter configuration
    if param_config is None:
        param_config = get_default_fates_params()
    
    # Oversample to account for constraint filtering
    n_initial = int(n_samples * oversample_factor)
    
    # Generate samples
    df_samples = generate_lhs_samples(
        param_names=param_config['names'],
        bounds=param_config['bounds'],
        n_samples=n_initial,
        seed=seed
    )
    
    # Apply ecological constraints
    if apply_constraints:
        df_samples = apply_ecological_constraints(df_samples)
        
        # If we don't have enough samples, resample
        if len(df_samples) < n_samples:
            logger.warning(f"Only {len(df_samples)} samples after constraints, "
                          f"need {n_samples}. Regenerating with more oversampling.")
            return generate_fates_parameter_samples(
                n_samples=n_samples,
                param_config=param_config,
                seed=seed + 1 if seed else None,
                apply_constraints=True,
                add_differences=add_differences,
                oversample_factor=oversample_factor * 2
            )
        
        # Subsample to target
        df_samples = df_samples.sample(n=n_samples, random_state=seed)
    
    # Add derived parameters
    if add_differences:
        df_samples = add_derived_parameters(df_samples)
    
    logger.info(f"Final parameter samples: {len(df_samples)} × {len(df_samples.columns)} parameters")
    
    return df_samples


def get_default_fates_params() -> Dict:
    """
    Get default FATES parameter configuration for tropical forests.
    
    Returns
    -------
    Dict
        Configuration with parameter names and bounds
    """
    param_names = [
        'fates_leaf_vcmax25top_e', 'fates_leaf_vcmax25top_l',
        'fates_leaf_slatop_e', 'fates_leaf_slatop_l',
        'fates_mort_bmort_e', 'fates_mort_bmort_l',
        'fates_wood_density_e', 'fates_wood_density_l',
        'fates_leaf_long_e', 'fates_leaf_long_l',
        'fates_alloc_storage_cushion',
    ]
    
    param_bounds = [
        [40, 105], [40, 105],           # vcmax (μmol/m²/s)
        [0.005, 0.04], [0.005, 0.04],   # SLA (m²/gC)
        [0.005, 0.05], [0.005, 0.05],   # mortality (/year)
        [0.2, 1.0], [0.2, 1.0],          # wood density (g/cm³)
        [0.2, 3.0], [0.2, 3.0],          # leaf longevity (years)
        [0.8, 1.5],                      # storage cushion
    ]
    
    return {
        'names': param_names,
        'bounds': param_bounds,
        'descriptions': {
            'fates_leaf_vcmax25top': 'Maximum carboxylation rate at 25C',
            'fates_leaf_slatop': 'Specific leaf area at canopy top',
            'fates_mort_bmort': 'Background mortality rate',
            'fates_wood_density': 'Wood density',
            'fates_leaf_long': 'Leaf longevity',
            'fates_alloc_storage_cushion': 'Storage allocation cushion',
        }
    }


if __name__ == "__main__":
    # Example usage
    logger.setLevel(logging.INFO)
    
    print("Generating FATES parameter samples...")
    samples = generate_fates_parameter_samples(n_samples=100)
    
    print(f"\nGenerated {len(samples)} samples")
    print(f"Parameters: {list(samples.columns)}")
    print(f"\nFirst 5 samples:")
    print(samples.head())
    
    print(f"\nParameter ranges:")
    print(samples.describe().loc[['min', 'max']])

