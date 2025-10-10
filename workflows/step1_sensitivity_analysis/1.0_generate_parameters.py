#!/usr/bin/env python
"""
Step 1.0: Generate Parameter Samples

Generate Latin Hypercube samples with ecological constraints for FATES sensitivity analysis.

Usage:
    python 1.0_generate_parameters.py --config ../../examples/manaus_k34/config.yaml
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from fates_emulator import sampling, utils

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate parameter samples for FATES')
    parser.add_argument('--config', type=str,
                       default='../../examples/manaus_k34/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='./params/parameter_samples.csv',
                       help='Output CSV file for parameter samples')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Number of samples (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    config = utils.load_config(args.config)
    
    # Setup logging
    log_level = config.get('logging', {}).get('level', 'INFO')
    utils.setup_logging(log_level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("FATES Parameter Sample Generation")
    logger.info("="*60)
    
    # Get parameters from config
    param_config = config.get('parameters', {})
    sensitivity_config = config.get('sensitivity', {})
    
    # Build parameter names and bounds
    param_names = []
    param_bounds = []
    
    for param in param_config.get('param_list', []):
        name = param['name']
        bounds = param['bounds']
        pft_specific = param.get('pft_specific', False)
        
        if pft_specific:
            # Add for each PFT
            pfts = param_config.get('pfts', {})
            for pft_key, pft_info in pfts.items():
                suffix = pft_info['suffix']
                param_names.append(f"{name}{suffix}")
                param_bounds.append(bounds)
        else:
            # Global parameter
            param_names.append(name)
            param_bounds.append(bounds)
    
    logger.info(f"\nParameters to sample ({len(param_names)}):")
    for name, bounds in zip(param_names, param_bounds):
        logger.info(f"  {name}: {bounds}")
    
    # Generate samples
    n_samples = args.n_samples or sensitivity_config.get('n_samples', 1500)
    seed = args.seed or sensitivity_config.get('seed', 11)
    apply_constraints = sensitivity_config.get('apply_ecological_constraints', True)
    oversample_factor = sensitivity_config.get('oversample_factor', 40.0)
    
    logger.info(f"\nGenerating {n_samples} samples...")
    logger.info(f"Seed: {seed}")
    logger.info(f"Apply ecological constraints: {apply_constraints}")
    
    # Build param config for sampling
    sampling_config = {
        'names': param_names,
        'bounds': param_bounds
    }
    
    # Generate samples
    df_samples = sampling.generate_fates_parameter_samples(
        n_samples=n_samples,
        param_config=sampling_config,
        seed=seed,
        apply_constraints=apply_constraints,
        add_differences=True,
        oversample_factor=oversample_factor
    )
    
    # Save samples
    output_path = Path(args.output)
    utils.save_dataframe_safe(df_samples, output_path, index=False)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Parameter Generation Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Generated: {len(df_samples)} samples")
    logger.info(f"Parameters: {len(df_samples.columns)} columns")
    logger.info(f"Saved to: {output_path}")
    
    # Show first few samples
    logger.info(f"\nFirst 3 samples:")
    print(df_samples.head(3).to_string())
    
    # Show parameter ranges
    logger.info(f"\nParameter ranges:")
    print(df_samples.describe().loc[['min', 'max']].to_string())
    
    logger.info("\n✓ All done!")

if __name__ == "__main__":
    main()

