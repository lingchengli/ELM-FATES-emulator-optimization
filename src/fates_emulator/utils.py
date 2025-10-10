"""
Utility functions for FATES-Emulator

Common helper functions used across the framework.
All functions are path-agnostic and use configuration parameters.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : Union[str, Path]
        Path to configuration file
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    config_path : Union[str, Path]
        Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved configuration to {config_path}")


def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[Union[str, Path]] = None
) -> None:
    """
    Setup logging configuration.
    
    Parameters
    ----------
    log_level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    log_file : Optional[Union[str, Path]]
        Path to log file (if None, log to console only)
    """
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Parameters
    ----------
    directory : Union[str, Path]
        Directory path
        
    Returns
    -------
    Path
        Directory path as Path object
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_param_names_by_pft(
    param_base_names: list,
    pft_suffixes: list = ['_e', '_l']
) -> list:
    """
    Generate parameter names for multiple PFTs.
    
    Parameters
    ----------
    param_base_names : list
        Base parameter names (e.g., ['vcmax', 'sla'])
    pft_suffixes : list
        PFT suffixes (e.g., ['_e', '_l'] for early/late)
        
    Returns
    -------
    list
        Full parameter names with PFT suffixes
        
    Examples
    --------
    >>> get_param_names_by_pft(['vcmax', 'sla'], ['_e', '_l'])
    ['vcmax_e', 'vcmax_l', 'sla_e', 'sla_l']
    """
    param_names = []
    for base in param_base_names:
        for suffix in pft_suffixes:
            param_names.append(f"{base}{suffix}")
    return param_names


def normalize_dataframe(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    method: str = 'minmax'
) -> tuple:
    """
    Normalize DataFrame columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : Optional[list]
        Columns to normalize (if None, normalize all numeric columns)
    method : str
        Normalization method ('minmax' or 'zscore')
        
    Returns
    -------
    tuple
        (normalized_df, normalization_params)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_norm = df.copy()
    norm_params = {}
    
    for col in columns:
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            norm_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
        
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            df_norm[col] = (df[col] - mean_val) / std_val
            norm_params[col] = {'method': 'zscore', 'mean': mean_val, 'std': std_val}
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return df_norm, norm_params


def denormalize_dataframe(
    df_norm: pd.DataFrame,
    norm_params: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Denormalize DataFrame using normalization parameters.
    
    Parameters
    ----------
    df_norm : pd.DataFrame
        Normalized dataframe
    norm_params : Dict[str, Dict]
        Normalization parameters from normalize_dataframe
        
    Returns
    -------
    pd.DataFrame
        Denormalized dataframe
    """
    df = df_norm.copy()
    
    for col, params in norm_params.items():
        if col not in df.columns:
            continue
            
        if params['method'] == 'minmax':
            df[col] = df[col] * (params['max'] - params['min']) + params['min']
        
        elif params['method'] == 'zscore':
            df[col] = df[col] * params['std'] + params['mean']
    
    return df


def compute_biomass_ratio(
    biomass_early: Union[float, np.ndarray],
    biomass_late: Union[float, np.ndarray],
    epsilon: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    Compute PFT biomass ratio (early / (early + late)).
    
    Parameters
    ----------
    biomass_early : Union[float, np.ndarray]
        Early successional PFT biomass
    biomass_late : Union[float, np.ndarray]
        Late successional PFT biomass
    epsilon : float
        Small value to avoid division by zero
        
    Returns
    -------
    Union[float, np.ndarray]
        Biomass ratio (0-1)
    """
    total_biomass = biomass_early + biomass_late + epsilon
    ratio = biomass_early / total_biomass
    return ratio


def check_coexistence(
    biomass_ratio: Union[float, np.ndarray],
    min_ratio: float = 0.1,
    max_ratio: float = 0.9
) -> Union[bool, np.ndarray]:
    """
    Check if PFT coexistence criteria is met.
    
    Parameters
    ----------
    biomass_ratio : Union[float, np.ndarray]
        PFT biomass ratio
    min_ratio : float
        Minimum ratio for coexistence (default 0.1 = 10% of total)
    max_ratio : float
        Maximum ratio for coexistence (default 0.9 = 90% of total)
        
    Returns
    -------
    Union[bool, np.ndarray]
        True if coexistence criteria met
    """
    return (biomass_ratio >= min_ratio) & (biomass_ratio <= max_ratio)


def load_dataframe_safe(
    filepath: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """
    Safely load DataFrame with error handling.
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Path to CSV file
    **kwargs
        Additional arguments passed to pd.read_csv
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe
        
    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If file is empty or invalid
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.stat().st_size == 0:
        raise ValueError(f"File is empty: {filepath}")
    
    try:
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Loaded dataframe from {filepath}: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Failed to load dataframe from {filepath}: {e}")
        raise


def save_dataframe_safe(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    **kwargs
) -> None:
    """
    Safely save DataFrame with error handling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save
    filepath : Union[str, Path]
        Path to save CSV file
    **kwargs
        Additional arguments passed to df.to_csv
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_csv(filepath, **kwargs)
        logger.info(f"Saved dataframe to {filepath}: {df.shape}")
    
    except Exception as e:
        logger.error(f"Failed to save dataframe to {filepath}: {e}")
        raise


def print_summary_statistics(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print summary statistics for DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    name : str
        Name for display
    """
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nNumeric columns statistics:")
    print(df.describe())


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("FATES-Emulator Utilities Example")
    print("=" * 60)
    
    # Test biomass ratio calculation
    biomass_e = np.array([5.0, 8.0, 2.0, 9.0])
    biomass_l = np.array([5.0, 2.0, 8.0, 1.0])
    
    ratios = compute_biomass_ratio(biomass_e, biomass_l)
    coexist = check_coexistence(ratios, min_ratio=0.1, max_ratio=0.9)
    
    print(f"\nBiomass Ratio Calculation:")
    print(f"Early biomass: {biomass_e}")
    print(f"Late biomass:  {biomass_l}")
    print(f"Ratios:        {ratios}")
    print(f"Coexistence:   {coexist}")
    
    # Test parameter names generation
    param_names = get_param_names_by_pft(['vcmax', 'sla', 'bmort'], ['_e', '_l'])
    print(f"\nGenerated parameter names:")
    print(param_names)

