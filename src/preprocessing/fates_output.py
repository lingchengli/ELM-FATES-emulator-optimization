"""
Extract and process FATES simulation outputs

Functions to read FATES NetCDF output files and extract relevant variables
for emulator training. All paths are configuration-based.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc

logger = logging.getLogger(__name__)


def extract_fates_outputs(
    output_files: List[Union[str, Path]],
    variables: List[str],
    time_aggregation: str = 'mean'
) -> pd.DataFrame:
    """
    Extract variables from FATES output files and aggregate.
    
    Parameters
    ----------
    output_files : List[Union[str, Path]]
        List of FATES output NetCDF files
    variables : List[str]
        Variable names to extract
    time_aggregation : str
        How to aggregate over time ('mean', 'sum', 'last')
        
    Returns
    -------
    pd.DataFrame
        Extracted outputs with one row per simulation
        
    Examples
    --------
    >>> files = ['run_001/output.nc', 'run_002/output.nc']
    >>> vars = ['GPP', 'EFLX_LH_TOT', 'FATES_VEGC_PF']
    >>> df = extract_fates_outputs(files, vars)
    """
    results = []
    
    for i, filepath in enumerate(output_files):
        try:
            data = extract_single_file(filepath, variables, time_aggregation)
            data['simulation_id'] = i
            data['filepath'] = str(filepath)
            results.append(data)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(output_files)} files")
        
        except Exception as e:
            logger.warning(f"Failed to process {filepath}: {e}")
            continue
    
    if not results:
        raise ValueError("No files successfully processed")
    
    df = pd.DataFrame(results)
    logger.info(f"Extracted {len(df)} simulations × {len(variables)} variables")
    
    return df


def extract_single_file(
    filepath: Union[str, Path],
    variables: List[str],
    time_aggregation: str = 'mean'
) -> Dict:
    """
    Extract variables from a single FATES output file.
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Path to FATES output NetCDF file
    variables : List[str]
        Variable names to extract
    time_aggregation : str
        How to aggregate over time
        
    Returns
    -------
    Dict
        Dictionary of extracted values
    """
    with xr.open_dataset(filepath) as ds:
        result = {}
        
        for var in variables:
            if var not in ds:
                logger.warning(f"Variable {var} not found in {filepath}")
                result[var] = np.nan
                continue
            
            # Get variable data
            data = ds[var].values
            
            # Handle PFT-specific variables (has pft dimension)
            if 'pft' in ds[var].dims or 'fates_levpft' in ds[var].dims:
                # Extract by PFT (assume PFT 1 = early, PFT 2 = late)
                if len(data.shape) >= 2:
                    if time_aggregation == 'mean':
                        result[f"{var}_e"] = np.nanmean(data[:, 0])
                        result[f"{var}_l"] = np.nanmean(data[:, 1])
                    elif time_aggregation == 'sum':
                        result[f"{var}_e"] = np.nansum(data[:, 0])
                        result[f"{var}_l"] = np.nansum(data[:, 1])
                    elif time_aggregation == 'last':
                        result[f"{var}_e"] = data[-1, 0]
                        result[f"{var}_l"] = data[-1, 1]
            else:
                # Non-PFT variable
                if time_aggregation == 'mean':
                    result[var] = np.nanmean(data)
                elif time_aggregation == 'sum':
                    result[var] = np.nansum(data)
                elif time_aggregation == 'last':
                    result[var] = data[-1] if len(data) > 0 else np.nan
        
        return result


def compute_derived_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived output variables (e.g., biomass ratios).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with FATES outputs
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional derived variables
    """
    df_out = df.copy()
    
    # Biomass ratio (if PFT biomass available)
    if 'FATES_VEGC_PF_e' in df.columns and 'FATES_VEGC_PF_l' in df.columns:
        total = df['FATES_VEGC_PF_e'] + df['FATES_VEGC_PF_l']
        df_out['PFTbiomass_r'] = df['FATES_VEGC_PF_e'] / (total + 1e-10)
        df_out['PFTbiomass_e'] = df['FATES_VEGC_PF_e']
        df_out['PFTbiomass_l'] = df['FATES_VEGC_PF_l']
    
    # Bowen ratio
    if 'FSH' in df.columns and 'EFLX_LH_TOT' in df.columns:
        df_out['Bowen'] = df['FSH'] / (df['EFLX_LH_TOT'] + 1e-10)
    
    # Rename common variables
    if 'EFLX_LH_TOT' in df.columns:
        df_out['ET'] = df['EFLX_LH_TOT'] / 2.5e6  # Latent heat to ET (rough conversion)
    
    if 'FATES_GPP' in df.columns:
        df_out['GPP'] = df['FATES_GPP']
    
    if 'FATES_VEGC_ABOVEGROUND' in df.columns:
        df_out['AGB'] = df['FATES_VEGC_ABOVEGROUND']
    
    logger.info(f"Added {len(df_out.columns) - len(df.columns)} derived variables")
    
    return df_out


def find_output_files(
    base_dir: Union[str, Path],
    pattern: str = "*.elm.h0.*.nc",
    recursive: bool = True
) -> List[Path]:
    """
    Find FATES output files in directory.
    
    Parameters
    ----------
    base_dir : Union[str, Path]
        Base directory to search
    pattern : str
        File pattern (glob)
    recursive : bool
        Search recursively
        
    Returns
    -------
    List[Path]
        List of output file paths
    """
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    
    if recursive:
        files = sorted(base_dir.rglob(pattern))
    else:
        files = sorted(base_dir.glob(pattern))
    
    logger.info(f"Found {len(files)} output files in {base_dir}")
    
    return files


def check_simulation_success(
    filepath: Union[str, Path],
    min_timesteps: int = 12
) -> bool:
    """
    Check if FATES simulation completed successfully.
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Path to output file
    min_timesteps : int
        Minimum number of timesteps expected
        
    Returns
    -------
    bool
        True if simulation appears successful
    """
    try:
        with xr.open_dataset(filepath) as ds:
            # Check if has time dimension with enough steps
            if 'time' in ds.dims:
                return ds.dims['time'] >= min_timesteps
            return True
    except:
        return False


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("FATES Output Extraction Example")
    print("=" * 60)
    
    print("\nThis module extracts FATES outputs from NetCDF files.")
    print("Usage:")
    print("  1. Find output files: find_output_files(base_dir)")
    print("  2. Extract variables: extract_fates_outputs(files, variables)")
    print("  3. Compute derived: compute_derived_outputs(df)")

