#!/usr/bin/env python
"""
Step 2.1: Train FLAML AutoML Emulators

This script trains XGBoost emulators using FLAML AutoML for multiple FATES output variables.

Usage:
    python 2.1_train_emulators.py --config path/to/config.yaml
    
Or with default config:
    python 2.1_train_emulators.py
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from fates_emulator import emulator, utils
from preprocessing import data_prep

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train FATES emulators with AutoML')
    parser.add_argument('--config', type=str, 
                       default='../../examples/manaus_k34/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--params', type=str, required=True,
                       help='Path to parameter samples CSV')
    parser.add_argument('--outputs', type=str, required=True,
                       help='Path to FATES outputs CSV')
    parser.add_argument('--output-dir', type=str, default='./emulator_models',
                       help='Directory to save trained models')
    parser.add_argument('--time-budget', type=int, default=600,
                       help='Time budget per model in seconds (default: 600)')
    parser.add_argument('--variables', nargs='+', 
                       default=['GPP', 'ET', 'AGB', 'PFTbiomass_r'],
                       help='Variables to train emulators for')
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = utils.load_config(args.config)
        log_level = config.get('logging', {}).get('level', 'INFO')
    else:
        print(f"Warning: Config file {args.config} not found, using defaults")
        log_level = 'INFO'
    
    # Setup logging
    utils.setup_logging(log_level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("FATES Emulator Training with FLAML AutoML")
    logger.info("="*60)
    
    # Load data
    logger.info(f"\nLoading parameter samples from {args.params}")
    df_params = utils.load_dataframe_safe(args.params)
    
    logger.info(f"Loading FATES outputs from {args.outputs}")
    df_outputs = utils.load_dataframe_safe(args.outputs)
    
    logger.info(f"\nParameters: {df_params.shape}")
    logger.info(f"Outputs: {df_outputs.shape}")
    
    # Get parameter columns (exclude any metadata columns)
    param_cols = [col for col in df_params.columns 
                  if not col.startswith('simulation_') and col not in ['filepath']]
    logger.info(f"Using {len(param_cols)} parameter columns")
    
    # Train emulators
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    emulators_trained = {}
    
    for var in args.variables:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training emulator for: {var}")
        logger.info(f"{'='*60}")
        
        if var not in df_outputs.columns:
            logger.warning(f"Variable {var} not found in outputs, skipping")
            continue
        
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = data_prep.prepare_training_data(
                df_params=df_params,
                df_outputs=df_outputs,
                target_variable=var,
                param_columns=param_cols,
                test_size=0.1,
                random_state=23
            )
            
            # Train emulator
            em = emulator.FATESEmulator(target_variable=var, random_state=23)
            
            metrics = em.train(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                time_budget=args.time_budget,
                n_jobs=-1,
                verbose=2
            )
            
            # Save emulator
            save_path = output_dir / f"{var}_emulator.pkl"
            em.save(save_path)
            
            emulators_trained[var] = {
                'emulator': em,
                'metrics': metrics,
                'path': save_path
            }
            
            logger.info(f"\n✓ {var} emulator trained successfully")
            logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
            logger.info(f"  Test RMSE: {metrics['test_rmse']:.4f}")
            logger.info(f"  Best model: {metrics['best_estimator']}")
            logger.info(f"  Saved to: {save_path}")
        
        except Exception as e:
            logger.error(f"✗ Failed to train emulator for {var}: {e}")
            continue
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Training Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Successfully trained {len(emulators_trained)}/{len(args.variables)} emulators")
    logger.info(f"Models saved to: {output_dir}")
    
    # Save summary
    import pandas as pd
    summary_data = []
    for var, info in emulators_trained.items():
        summary_data.append({
            'variable': var,
            'test_r2': info['metrics']['test_r2'],
            'test_rmse': info['metrics']['test_rmse'],
            'best_estimator': info['metrics']['best_estimator'],
            'training_time_sec': info['metrics']['training_time']
        })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        summary_path = output_dir / 'training_summary.csv'
        df_summary.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved to: {summary_path}")
        
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        print(df_summary.to_string(index=False))
    
    logger.info("\n✓ All done!")


if __name__ == "__main__":
    main()

