"""
Data preparation for machine learning

Functions to prepare FATES simulation data for emulator training.
"""

import logging
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def prepare_training_data(
    df_params: pd.DataFrame,
    df_outputs: pd.DataFrame,
    target_variable: str,
    param_columns: Optional[List[str]] = None,
    remove_failed: bool = True,
    test_size: float = 0.1,
    random_state: int = 23
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare training and test data for emulator.
    
    Parameters
    ----------
    df_params : pd.DataFrame
        Parameter samples
    df_outputs : pd.DataFrame
        FATES simulation outputs
    target_variable : str
        Target variable name (e.g., 'GPP', 'ET', 'AGB')
    param_columns : Optional[List[str]]
        Specific parameter columns to use (if None, use all)
    remove_failed : bool
        Remove failed simulations (missing or extreme values)
    test_size : float
        Fraction for test set
    random_state : int
        Random seed
        
    Returns
    -------
    Tuple
        (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Preparing training data for {target_variable}")
    
    # Merge parameters and outputs
    if len(df_params) != len(df_outputs):
        raise ValueError(f"Parameter and output lengths don't match: {len(df_params)} vs {len(df_outputs)}")
    
    df = pd.concat([df_params.reset_index(drop=True), df_outputs.reset_index(drop=True)], axis=1)
    
    # Check target variable exists
    if target_variable not in df.columns:
        raise ValueError(f"Target variable {target_variable} not found in outputs")
    
    # Remove failed simulations
    if remove_failed:
        n_before = len(df)
        
        # Remove missing values
        df = df.dropna(subset=[target_variable])
        
        # Remove extreme negative values (failure indicator)
        df = df[df[target_variable] > -1000]
        
        n_after = len(df)
        logger.info(f"Removed {n_before - n_after} failed simulations ({100*(n_before-n_after)/n_before:.1f}%)")
    
    if len(df) < 50:
        raise ValueError(f"Too few valid samples: {len(df)}")
    
    # Select features
    if param_columns is None:
        param_columns = list(df_params.columns)
    
    X = df[param_columns]
    y = df[target_variable]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Features: {len(param_columns)}")
    
    return X_train, X_test, y_train, y_test


def filter_coexistence_data(
    df: pd.DataFrame,
    biomass_ratio_col: str = 'PFTbiomass_r',
    min_ratio: float = 0.1,
    max_ratio: float = 0.9
) -> pd.DataFrame:
    """
    Filter data to only include coexistence cases.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with biomass ratios
    biomass_ratio_col : str
        Column name for biomass ratio
    min_ratio : float
        Minimum ratio for coexistence
    max_ratio : float
        Maximum ratio for coexistence
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    if biomass_ratio_col not in df.columns:
        logger.warning(f"Biomass ratio column {biomass_ratio_col} not found")
        return df
    
    n_before = len(df)
    df_filtered = df[
        (df[biomass_ratio_col] >= min_ratio) & 
        (df[biomass_ratio_col] <= max_ratio)
    ].copy()
    n_after = len(df_filtered)
    
    logger.info(f"Coexistence filter: {n_before} → {n_after} samples "
                f"({100*n_after/n_before:.1f}% retained)")
    
    return df_filtered


def downsample_dominant_pft(
    df: pd.DataFrame,
    biomass_ratio_col: str = 'PFTbiomass_r',
    coexist_threshold: float = 0.1,
    n_samples_per_category: int = 500,
    random_state: int = 23
) -> pd.DataFrame:
    """
    Downsample cases where one PFT dominates.
    
    Helps balance training data to better learn coexistence region.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    biomass_ratio_col : str
        Biomass ratio column
    coexist_threshold : float
        Threshold for coexistence (0.1 = 10% minimum)
    n_samples_per_category : int
        Max samples for dominated categories
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Downsampled dataframe
    """
    if biomass_ratio_col not in df.columns:
        return df
    
    # Categorize
    df_coexist = df[
        (df[biomass_ratio_col] >= coexist_threshold) &
        (df[biomass_ratio_col] <= (1 - coexist_threshold))
    ]
    
    df_early_dom = df[df[biomass_ratio_col] > (1 - coexist_threshold)]
    df_late_dom = df[df[biomass_ratio_col] < coexist_threshold]
    
    # Downsample dominated categories
    if len(df_early_dom) > n_samples_per_category:
        df_early_dom = df_early_dom.sample(n=n_samples_per_category, random_state=random_state)
    
    if len(df_late_dom) > n_samples_per_category:
        df_late_dom = df_late_dom.sample(n=n_samples_per_category, random_state=random_state)
    
    # Combine
    df_balanced = pd.concat([df_coexist, df_early_dom, df_late_dom], ignore_index=True)
    
    logger.info(f"Downsampling: {len(df)} → {len(df_balanced)} samples")
    logger.info(f"  Coexist: {len(df_coexist)}")
    logger.info(f"  Early dominant: {len(df_early_dom)}")
    logger.info(f"  Late dominant: {len(df_late_dom)}")
    
    return df_balanced


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Data Preparation Example")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(23)
    n = 1000
    
    df_params = pd.DataFrame({
        'vcmax_e': np.random.uniform(40, 105, n),
        'vcmax_l': np.random.uniform(40, 105, n),
    })
    
    df_outputs = pd.DataFrame({
        'GPP': np.random.uniform(5, 15, n),
        'PFTbiomass_r': np.random.uniform(0, 1, n)
    })
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_training_data(
        df_params, df_outputs, 'GPP'
    )
    
    print(f"\nTraining data prepared:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")

