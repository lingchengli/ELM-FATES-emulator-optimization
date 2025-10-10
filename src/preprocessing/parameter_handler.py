"""
FATES parameter file handling

Functions to read and modify FATES parameter NetCDF files.
All operations create new files - no in-place modifications.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Union, List, Optional

import netCDF4 as nc
import numpy as np

logger = logging.getLogger(__name__)


def update_fates_parameters(
    base_param_file: Union[str, Path],
    output_param_file: Union[str, Path],
    param_values: Dict[str, float],
    pft_indices: Optional[Dict[str, int]] = None
) -> None:
    """
    Create new FATES parameter file with updated values.
    
    Parameters
    ----------
    base_param_file : Union[str, Path]
        Base parameter file to copy from
    output_param_file : Union[str, Path]
        Output parameter file to create
    param_values : Dict[str, float]
        Parameter values to update {param_name: value}
        Use suffix _e for early PFT (index 0), _l for late PFT (index 1)
    pft_indices : Optional[Dict[str, int]]
        Mapping of PFT suffixes to indices (default: {'_e': 0, '_l': 1})
        
    Examples
    --------
    >>> params = {
    ...     'fates_leaf_vcmax25top_e': 75.0,
    ...     'fates_leaf_vcmax25top_l': 50.0,
    ...     'fates_alloc_storage_cushion': 1.2
    ... }
    >>> update_fates_parameters('base.nc', 'run_001.nc', params)
    """
    if pft_indices is None:
        pft_indices = {'_e': 0, '_l': 1}
    
    base_param_file = Path(base_param_file)
    output_param_file = Path(output_param_file)
    
    if not base_param_file.exists():
        raise FileNotFoundError(f"Base parameter file not found: {base_param_file}")
    
    # Copy base file to output
    output_param_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_param_file, output_param_file)
    
    # Update parameters
    with nc.Dataset(output_param_file, 'r+') as ds:
        updated_count = 0
        
        for param_name, value in param_values.items():
            # Check if parameter has PFT suffix
            pft_index = None
            base_name = param_name
            
            for suffix, index in pft_indices.items():
                if param_name.endswith(suffix):
                    base_name = param_name[:-len(suffix)]
                    pft_index = index
                    break
            
            # Check if parameter exists
            if base_name not in ds.variables:
                logger.warning(f"Parameter {base_name} not found in file, skipping")
                continue
            
            # Update value
            try:
                if pft_index is not None:
                    # PFT-specific parameter
                    ds.variables[base_name][pft_index] = value
                else:
                    # Global parameter
                    ds.variables[base_name][:] = value
                
                updated_count += 1
                logger.debug(f"Updated {param_name} = {value}")
            
            except Exception as e:
                logger.warning(f"Failed to update {param_name}: {e}")
                continue
    
    logger.info(f"Updated {updated_count}/{len(param_values)} parameters in {output_param_file}")


def read_fates_parameters(
    param_file: Union[str, Path],
    param_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Read parameter values from FATES parameter file.
    
    Parameters
    ----------
    param_file : Union[str, Path]
        FATES parameter file
    param_names : Optional[List[str]]
        Specific parameters to read (if None, read all)
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of parameter values
    """
    param_file = Path(param_file)
    
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")
    
    params = {}
    
    with nc.Dataset(param_file, 'r') as ds:
        if param_names is None:
            param_names = list(ds.variables.keys())
        
        for param_name in param_names:
            if param_name in ds.variables:
                params[param_name] = ds.variables[param_name][:].copy()
            else:
                logger.warning(f"Parameter {param_name} not found")
    
    logger.info(f"Read {len(params)} parameters from {param_file}")
    
    return params


def create_parameter_files_batch(
    base_param_file: Union[str, Path],
    output_dir: Union[str, Path],
    param_samples: Union[Dict, 'pd.DataFrame'],
    prefix: str = 'fates_params',
    start_index: int = 1
) -> List[Path]:
    """
    Create multiple parameter files from samples.
    
    Parameters
    ----------
    base_param_file : Union[str, Path]
        Base parameter file
    output_dir : Union[str, Path]
        Output directory for parameter files
    param_samples : Union[Dict, pd.DataFrame]
        Parameter samples (DataFrame or dict of arrays)
    prefix : str
        Filename prefix
    start_index : int
        Starting index for numbering
        
    Returns
    -------
    List[Path]
        List of created parameter file paths
        
    Examples
    --------
    >>> import pandas as pd
    >>> samples = pd.DataFrame({
    ...     'fates_leaf_vcmax25top_e': [70, 80, 90],
    ...     'fates_leaf_vcmax25top_l': [45, 50, 55]
    ... })
    >>> files = create_parameter_files_batch('base.nc', 'params/', samples)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame if dict
    if isinstance(param_samples, dict):
        import pandas as pd
        param_samples = pd.DataFrame(param_samples)
    
    created_files = []
    n_samples = len(param_samples)
    
    logger.info(f"Creating {n_samples} parameter files...")
    
    for i, (idx, row) in enumerate(param_samples.iterrows()):
        file_index = start_index + i
        output_file = output_dir / f"{prefix}_{file_index:04d}.nc"
        
        param_dict = row.to_dict()
        
        update_fates_parameters(
            base_param_file=base_param_file,
            output_param_file=output_file,
            param_values=param_dict
        )
        
        created_files.append(output_file)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Created {i + 1}/{n_samples} files")
    
    logger.info(f"Created {len(created_files)} parameter files in {output_dir}")
    
    return created_files


def get_parameter_info(param_file: Union[str, Path]) -> Dict:
    """
    Get information about parameters in file.
    
    Parameters
    ----------
    param_file : Union[str, Path]
        FATES parameter file
        
    Returns
    -------
    Dict
        Dictionary with parameter metadata
    """
    param_file = Path(param_file)
    
    info = {
        'file': str(param_file),
        'parameters': {},
        'dimensions': {}
    }
    
    with nc.Dataset(param_file, 'r') as ds:
        # Get dimensions
        for dim_name, dim in ds.dimensions.items():
            info['dimensions'][dim_name] = len(dim)
        
        # Get parameters
        for var_name, var in ds.variables.items():
            info['parameters'][var_name] = {
                'shape': var.shape,
                'dtype': str(var.dtype),
                'dimensions': var.dimensions
            }
            
            # Get attributes if available
            if hasattr(var, 'long_name'):
                info['parameters'][var_name]['long_name'] = var.long_name
            if hasattr(var, 'units'):
                info['parameters'][var_name]['units'] = var.units
    
    return info


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("FATES Parameter Handler Example")
    print("=" * 60)
    
    print("\nThis module handles FATES parameter files.")
    print("Key functions:")
    print("  - update_fates_parameters(): Modify parameters")
    print("  - read_fates_parameters(): Read current values")
    print("  - create_parameter_files_batch(): Create multiple files")
    print("  - get_parameter_info(): Inspect parameter file")

