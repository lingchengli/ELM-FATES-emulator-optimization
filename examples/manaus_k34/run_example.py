#!/usr/bin/env python
"""
FATES-Emulator: Complete Workflow Example

This script demonstrates the complete workflow using synthetic data.
For real analysis, you would use actual FATES simulation outputs.

Usage:
    python run_example.py
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from fates_emulator import sampling, emulator, diagnostics, calibration, utils
import numpy as np
import pandas as pd

# Setup logging
utils.setup_logging(log_level='INFO')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("FATES-Emulator: Complete Workflow Example")
    logger.info("="*60)
    logger.info("\nThis example demonstrates:")
    logger.info("  1. Parameter sampling")
    logger.info("  2. Emulator training with FLAML AutoML")
    logger.info("  3. SHAP analysis")
    logger.info("  4. Parameter calibration")
    logger.info("\nNote: Using synthetic data for demonstration")
    
    # ==========================================
    # Step 1: Generate Parameter Samples
    # ==========================================
    logger.info("\n" + "="*60)
    logger.info("Step 1: Generate Parameter Samples")
    logger.info("="*60)
    
    logger.info("\nGenerating 500 parameter samples with ecological constraints...")
    df_params = sampling.generate_fates_parameter_samples(
        n_samples=500,
        seed=42,
        apply_constraints=True,
        add_differences=True
    )
    
    logger.info(f"Generated {len(df_params)} samples with {len(df_params.columns)} parameters")
    logger.info(f"Parameters: {list(df_params.columns[:5])}...")
    
    # ==========================================
    # Generate Synthetic FATES Outputs
    # ==========================================
    logger.info("\nGenerating synthetic FATES outputs...")
    logger.info("(In real workflow, these come from FATES simulations)")
    
    np.random.seed(42)
    
    # Synthetic relationships (simplified)
    df_outputs = pd.DataFrame({
        'GPP': (
            5 + 0.05 * df_params['fates_leaf_vcmax25top_e'] +
            0.03 * df_params['fates_leaf_vcmax25top_l'] +
            np.random.normal(0, 1, len(df_params))
        ),
        'ET': (
            2 + 0.02 * df_params['fates_leaf_slatop_e'] * 100 +
            np.random.normal(0, 0.3, len(df_params))
        ),
        'AGB': (
            10 + 0.1 * df_params['fates_wood_density_l'] * 10 +
            np.random.normal(0, 2, len(df_params))
        )
    })
    
    # Biomass ratio
    biomass_e = 5 + 0.02 * df_params['fates_leaf_vcmax25top_e']
    biomass_l = 5 + 0.02 * df_params['fates_leaf_vcmax25top_l']
    df_outputs['PFTbiomass_r'] = biomass_e / (biomass_e + biomass_l)
    
    logger.info(f"Generated outputs: {list(df_outputs.columns)}")
    
    # ==========================================
    # Step 2: Train Emulators with AutoML
    # ==========================================
    logger.info("\n" + "="*60)
    logger.info("Step 2: Train Emulators with FLAML AutoML")
    logger.info("="*60)
    
    # Select parameters to use (excluding derived)
    param_cols = [col for col in df_params.columns if not col.endswith('_d')][:8]
    logger.info(f"\nUsing {len(param_cols)} parameters as features")
    
    # Train emulator for GPP
    logger.info("\nTraining GPP emulator (60 second AutoML budget)...")
    gpp_emulator = emulator.FATESEmulator(target_variable='GPP', random_state=42)
    
    metrics = gpp_emulator.train(
        X_train=df_params[param_cols],
        y_train=df_outputs['GPP'],
        time_budget=60,  # 1 minute for demo
        verbose=1
    )
    
    logger.info(f"\n✓ GPP Emulator trained:")
    logger.info(f"  Best model: {metrics['best_estimator']}")
    logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
    logger.info(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    
    # Train emulator for biomass ratio
    logger.info("\nTraining PFT biomass ratio emulator...")
    ratio_emulator = emulator.FATESEmulator(target_variable='PFTbiomass_r', random_state=42)
    
    metrics_ratio = ratio_emulator.train(
        X_train=df_params[param_cols],
        y_train=df_outputs['PFTbiomass_r'],
        time_budget=60,
        verbose=1
    )
    
    logger.info(f"\n✓ Biomass ratio emulator trained:")
    logger.info(f"  Test R²: {metrics_ratio['test_r2']:.4f}")
    
    # ==========================================
    # Step 3: SHAP Analysis
    # ==========================================
    logger.info("\n" + "="*60)
    logger.info("Step 3: SHAP Interpretability Analysis")
    logger.info("="*60)
    
    logger.info("\nComputing SHAP values for GPP emulator...")
    X_sample = df_params[param_cols].sample(n=100, random_state=42)
    shap_values = diagnostics.compute_shap_values(
        gpp_emulator.model,
        X_sample,
        background_samples=50
    )
    
    importance = diagnostics.get_shap_feature_importance(shap_values)
    
    logger.info("\nTop 5 most important parameters for GPP:")
    for i, row in importance.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f} ({row['direction']})")
    
    # ==========================================
    # Step 4: Parameter Calibration
    # ==========================================
    logger.info("\n" + "="*60)
    logger.info("Step 4: Parameter Calibration")
    logger.info("="*60)
    
    # Define observations (synthetic targets)
    observations = {
        'GPP': 8.5,
        'PFTbiomass_r': 0.5  # Equal biomass
    }
    
    logger.info(f"\nTarget observations:")
    for var, val in observations.items():
        logger.info(f"  {var}: {val}")
    
    # Get parameter bounds
    param_config = sampling.get_default_fates_params()
    param_bounds = [tuple(b) for b in param_config['bounds'][:len(param_cols)]]
    
    logger.info(f"\nRunning optimization (200 iterations)...")
    
    result = calibration.calibrate_parameters(
        emulators={'GPP': gpp_emulator, 'PFTbiomass_r': ratio_emulator},
        param_names=param_cols,
        param_bounds=param_bounds,
        observations=observations,
        weights={'observations': 1.0, 'coexistence': 1.0},
        method='differential_evolution',
        n_iterations=200,
        random_seed=42
    )
    
    logger.info(f"\n✓ Calibration complete:")
    logger.info(f"  Objective value: {result['objective_value']:.6f}")
    logger.info(f"  Success: {result['success']}")
    
    logger.info(f"\nOptimized parameters (top 5):")
    for i, (param, value) in enumerate(list(result['best_parameters'].items())[:5]):
        logger.info(f"  {param}: {value:.4f}")
    
    logger.info(f"\nPredictions with optimized parameters:")
    for var, pred in result['predictions'].items():
        obs = observations.get(var, 'N/A')
        logger.info(f"  {var}: {pred:.4f} (target: {obs})")
    
    # Check coexistence
    biomass_ratio = result['predictions']['PFTbiomass_r']
    coexists = 0.1 <= biomass_ratio <= 0.9
    logger.info(f"\nCoexistence check:")
    logger.info(f"  Biomass ratio: {biomass_ratio:.3f}")
    logger.info(f"  Status: {'✓ PASS' if coexists else '✗ FAIL'} (range: [0.1, 0.9])")
    
    # ==========================================
    # Summary
    # ==========================================
    logger.info("\n" + "="*60)
    logger.info("Example Complete!")
    logger.info("="*60)
    
    logger.info("\nWhat was demonstrated:")
    logger.info("  ✓ Parameter sampling with ecological constraints")
    logger.info("  ✓ FLAML AutoML emulator training (automatic hyperparameter tuning)")
    logger.info("  ✓ SHAP interpretability analysis")
    logger.info("  ✓ Multi-objective calibration with coexistence constraints")
    
    logger.info("\nFor real workflow:")
    logger.info("  1. Use actual FATES simulation outputs")
    logger.info("  2. Train emulators with longer time budgets (10-30 min)")
    logger.info("  3. Run more optimization iterations (1000+)")
    logger.info("  4. Validate with full FATES simulations")
    
    logger.info("\nSee workflows/ directory for complete scripts.")
    logger.info("\n✓ All done!")

if __name__ == "__main__":
    main()

