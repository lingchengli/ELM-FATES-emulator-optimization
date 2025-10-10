#!/usr/bin/env python
"""
Step 3.1: Optimize Parameters Using Emulators

Use trained emulators to find optimal parameters that match observations
and maintain PFT coexistence.

Usage:
    python 3.1_optimize_parameters.py --config ../../examples/manaus_k34/config.yaml
"""

import argparse
import logging
from pathlib import Path
import sys
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from fates_emulator import calibration, emulator, utils
import pandas as pd

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Calibrate FATES parameters using emulators')
    parser.add_argument('--config', type=str,
                       default='../../examples/manaus_k34/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--models-dir', type=str, required=True,
                       help='Directory with trained emulator models')
    parser.add_argument('--output', type=str, default='./calibrated_parameters.csv',
                       help='Output file for calibrated parameters')
    parser.add_argument('--n-starts', type=int, default=None,
                       help='Number of optimization starts (overrides config)')
    parser.add_argument('--n-iterations', type=int, default=None,
                       help='Number of iterations (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    config = utils.load_config(args.config)
    
    # Setup logging
    log_level = config.get('logging', {}).get('level', 'INFO')
    utils.setup_logging(log_level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("FATES Parameter Calibration")
    logger.info("="*60)
    
    # Load emulators
    models_dir = Path(args.models_dir)
    logger.info(f"\nLoading emulators from {models_dir}")
    
    emulators = {}
    model_files = list(models_dir.glob("*_emulator.pkl"))
    
    if not model_files:
        logger.error(f"No emulator files found in {models_dir}")
        sys.exit(1)
    
    for model_file in model_files:
        var_name = model_file.stem.replace('_emulator', '')
        em = emulator.FATESEmulator.load(model_file)
        emulators[var_name] = em
        logger.info(f"  ✓ Loaded {var_name}: R² = {em.performance_metrics.get('test_r2', 'N/A'):.3f}")
    
    # Get calibration config
    calib_config = config.get('calibration', {})
    param_config = config.get('parameters', {})
    
    # Build parameter names and bounds
    param_names = []
    param_bounds = []
    
    for param in param_config.get('param_list', []):
        name = param['name']
        bounds = param['bounds']
        pft_specific = param.get('pft_specific', False)
        
        if pft_specific:
            pfts = param_config.get('pfts', {})
            for pft_key, pft_info in pfts.items():
                suffix = pft_info['suffix']
                param_names.append(f"{name}{suffix}")
                param_bounds.append(tuple(bounds))
        else:
            param_names.append(name)
            param_bounds.append(tuple(bounds))
    
    logger.info(f"\nOptimizing {len(param_names)} parameters")
    
    # Get observations and weights
    observations = calib_config.get('observations', {})
    weights = calib_config.get('weights', {})
    
    logger.info(f"\nObservations to match:")
    for var, val in observations.items():
        logger.info(f"  {var}: {val}")
    
    logger.info(f"\nObjective weights:")
    for obj, weight in weights.items():
        logger.info(f"  {obj}: {weight}")
    
    # Run optimization
    n_starts = args.n_starts or calib_config.get('n_starts', 5)
    n_iterations = args.n_iterations or calib_config.get('n_iterations', 1000)
    random_seed = calib_config.get('random_seed', 23)
    
    logger.info(f"\nRunning multi-start calibration...")
    logger.info(f"  Starts: {n_starts}")
    logger.info(f"  Iterations per start: {n_iterations}")
    
    results = calibration.multi_start_calibration(
        emulators=emulators,
        param_names=param_names,
        param_bounds=param_bounds,
        observations=observations,
        weights=weights,
        n_starts=n_starts,
        n_iterations_per_start=n_iterations,
        random_seed=random_seed
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame with all results
    results_data = []
    for i, result in enumerate(results):
        row = {'run': i + 1, 'objective': result['objective_value']}
        row.update(result['best_parameters'])
        row.update({f'pred_{k}': v for k, v in result['predictions'].items()})
        results_data.append(row)
    
    df_results = pd.DataFrame(results_data)
    utils.save_dataframe_safe(df_results, output_path, index=False)
    
    # Save detailed results as pickle
    pickle_path = output_path.parent / 'calibration_results_full.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Calibration Complete")
    logger.info(f"{'='*60}")
    logger.info(f"\nBest result (Run {results_data[0]['run']}):")
    logger.info(f"  Objective value: {results[0]['objective_value']:.6f}")
    
    logger.info(f"\nOptimized parameters:")
    for param, value in results[0]['best_parameters'].items():
        logger.info(f"  {param}: {value:.4f}")
    
    logger.info(f"\nPredicted outputs:")
    for var, value in results[0]['predictions'].items():
        obs_val = observations.get(var, 'N/A')
        logger.info(f"  {var}: {value:.4f} (obs: {obs_val})")
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  Parameters: {output_path}")
    logger.info(f"  Full results: {pickle_path}")
    
    # Check coexistence
    if 'PFTbiomass_r' in results[0]['predictions']:
        biomass_ratio = results[0]['predictions']['PFTbiomass_r']
        coexist_min = calib_config.get('coexistence', {}).get('min_biomass_ratio', 0.1)
        coexist_max = calib_config.get('coexistence', {}).get('max_biomass_ratio', 0.9)
        
        coexists = coexist_min <= biomass_ratio <= coexist_max
        logger.info(f"\nCoexistence check:")
        logger.info(f"  Biomass ratio: {biomass_ratio:.3f}")
        logger.info(f"  Coexistence range: [{coexist_min}, {coexist_max}]")
        logger.info(f"  Status: {'✓ PASS' if coexists else '✗ FAIL'}")
    
    logger.info("\n✓ All done!")
    logger.info("\nNext step: Validate with full FATES simulation")
    logger.info("  python 3.2_validate_calibration.py --params calibrated_parameters.csv")

if __name__ == "__main__":
    main()

