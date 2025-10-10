"""
Parameter calibration using emulators

Functions for optimizing FATES parameters using trained emulators.
Supports multi-objective optimization with coexistence constraints.
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

logger = logging.getLogger(__name__)


class ObjectiveFunction:
    """
    Objective function for parameter calibration.
    
    Combines multiple objectives (observations, coexistence, constraints).
    """
    
    def __init__(
        self,
        emulators: Dict,
        param_names: List[str],
        param_bounds: List[Tuple[float, float]],
        observations: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize objective function.
        
        Parameters
        ----------
        emulators : Dict
            Dictionary of trained emulators {variable: emulator}
        param_names : List[str]
            Parameter names in order
        param_bounds : List[Tuple[float, float]]
            Parameter bounds [(min, max), ...]
        observations : Optional[Dict[str, float]]
            Observed values {variable: value}
        weights : Optional[Dict[str, float]]
            Weights for objectives {objective_name: weight}
        """
        self.emulators = emulators
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.observations = observations or {}
        self.weights = weights or {'observations': 1.0, 'coexistence': 1.0}
        
        logger.info(f"Objective function initialized with {len(emulators)} emulators")
    
    def __call__(self, params: np.ndarray) -> float:
        """
        Evaluate objective function.
        
        Parameters
        ----------
        params : np.ndarray
            Parameter values
            
        Returns
        -------
        float
            Objective value (lower is better)
        """
        # Create DataFrame for prediction
        param_df = pd.DataFrame([params], columns=self.param_names)
        
        # Initialize objective
        total_objective = 0.0
        
        # 1. Observation matching
        if self.observations and 'observations' in self.weights:
            obs_objective = 0.0
            for var, obs_value in self.observations.items():
                if var in self.emulators:
                    pred_value = self.emulators[var].predict(param_df)[0]
                    obs_objective += (pred_value - obs_value) ** 2
            
            total_objective += self.weights['observations'] * obs_objective
        
        # 2. Coexistence constraint
        if 'coexistence' in self.weights:
            coexist_objective = self._evaluate_coexistence(param_df)
            total_objective += self.weights['coexistence'] * coexist_objective
        
        # 3. Ecological constraints
        if 'ecological' in self.weights:
            eco_objective = self._evaluate_ecological_constraints(params)
            total_objective += self.weights['ecological'] * eco_objective
        
        return total_objective
    
    def _evaluate_coexistence(self, param_df: pd.DataFrame) -> float:
        """Evaluate coexistence objective."""
        # Predict biomass ratio
        if 'PFTbiomass_r' in self.emulators:
            biomass_ratio = self.emulators['PFTbiomass_r'].predict(param_df)[0]
            
            # Penalty if outside coexistence range [0.1, 0.9]
            if biomass_ratio < 0.1:
                return (0.1 - biomass_ratio) ** 2 * 100
            elif biomass_ratio > 0.9:
                return (biomass_ratio - 0.9) ** 2 * 100
            else:
                return 0.0
        
        return 0.0
    
    def _evaluate_ecological_constraints(self, params: np.ndarray) -> float:
        """Evaluate ecological relationship constraints."""
        penalty = 0.0
        
        # Example: early successional should have higher Vcmax than late
        try:
            vcmax_e_idx = self.param_names.index('fates_leaf_vcmax25top_e')
            vcmax_l_idx = self.param_names.index('fates_leaf_vcmax25top_l')
            
            if params[vcmax_e_idx] <= params[vcmax_l_idx]:
                penalty += (params[vcmax_l_idx] - params[vcmax_e_idx]) ** 2
        except ValueError:
            pass
        
        return penalty


def calibrate_parameters(
    emulators: Dict,
    param_names: List[str],
    param_bounds: List[Tuple[float, float]],
    observations: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    method: str = 'differential_evolution',
    n_iterations: int = 1000,
    random_seed: int = 23
) -> Dict:
    """
    Calibrate parameters using emulators.
    
    Parameters
    ----------
    emulators : Dict
        Trained emulators
    param_names : List[str]
        Parameter names
    param_bounds : List[Tuple[float, float]]
        Parameter bounds
    observations : Optional[Dict[str, float]]
        Observed values to match
    weights : Optional[Dict[str, float]]
        Objective weights
    method : str
        Optimization method ('differential_evolution' or 'minimize')
    n_iterations : int
        Number of iterations
    random_seed : int
        Random seed
        
    Returns
    -------
    Dict
        Calibration results including best parameters and objective value
    """
    logger.info(f"Starting parameter calibration with {method}")
    
    # Create objective function
    objective = ObjectiveFunction(
        emulators=emulators,
        param_names=param_names,
        param_bounds=param_bounds,
        observations=observations,
        weights=weights
    )
    
    # Run optimization
    if method == 'differential_evolution':
        result = differential_evolution(
            objective,
            bounds=param_bounds,
            maxiter=n_iterations,
            seed=random_seed,
            disp=True,
            workers=1
        )
    elif method == 'minimize':
        # Initial guess (middle of bounds)
        x0 = np.array([(b[0] + b[1]) / 2 for b in param_bounds])
        
        result = minimize(
            objective,
            x0=x0,
            bounds=param_bounds,
            method='L-BFGS-B'
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Prepare results
    best_params = dict(zip(param_names, result.x))
    
    # Predict outputs with best parameters
    param_df = pd.DataFrame([result.x], columns=param_names)
    predictions = {}
    for var, emulator in emulators.items():
        predictions[var] = float(emulator.predict(param_df)[0])
    
    calibration_results = {
        'best_parameters': best_params,
        'objective_value': result.fun,
        'predictions': predictions,
        'success': result.success if hasattr(result, 'success') else True,
        'message': result.message if hasattr(result, 'message') else 'Optimization complete',
        'n_iterations': result.nit if hasattr(result, 'nit') else n_iterations
    }
    
    logger.info(f"Calibration complete: objective = {result.fun:.6f}")
    logger.info(f"Success: {calibration_results['success']}")
    
    return calibration_results


def multi_start_calibration(
    emulators: Dict,
    param_names: List[str],
    param_bounds: List[Tuple[float, float]],
    observations: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    n_starts: int = 5,
    n_iterations_per_start: int = 200,
    random_seed: int = 23
) -> List[Dict]:
    """
    Run calibration from multiple starting points.
    
    Helps avoid local minima in optimization.
    
    Parameters
    ----------
    emulators : Dict
        Trained emulators
    param_names : List[str]
        Parameter names
    param_bounds : List[Tuple[float, float]]
        Parameter bounds
    observations : Optional[Dict[str, float]]
        Observed values
    weights : Optional[Dict[str, float]]
        Objective weights
    n_starts : int
        Number of different starting points
    n_iterations_per_start : int
        Iterations per start
    random_seed : int
        Random seed
        
    Returns
    -------
    List[Dict]
        List of calibration results, sorted by objective value
    """
    logger.info(f"Multi-start calibration with {n_starts} starts")
    
    results = []
    
    for i in range(n_starts):
        logger.info(f"\n{'='*60}")
        logger.info(f"Start {i+1}/{n_starts}")
        logger.info(f"{'='*60}")
        
        result = calibrate_parameters(
            emulators=emulators,
            param_names=param_names,
            param_bounds=param_bounds,
            observations=observations,
            weights=weights,
            method='differential_evolution',
            n_iterations=n_iterations_per_start,
            random_seed=random_seed + i
        )
        
        results.append(result)
    
    # Sort by objective value
    results.sort(key=lambda x: x['objective_value'])
    
    logger.info(f"\n{'='*60}")
    logger.info("Multi-start calibration complete")
    logger.info(f"Best objective: {results[0]['objective_value']:.6f}")
    logger.info(f"Worst objective: {results[-1]['objective_value']:.6f}")
    logger.info(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("FATES Parameter Calibration Example")
    print("=" * 60)
    
    print("\nThis module performs parameter calibration using emulators.")
    print("Key functions:")
    print("  - calibrate_parameters(): Single optimization run")
    print("  - multi_start_calibration(): Multiple starting points")
    print("  - ObjectiveFunction: Flexible objective with multiple terms")
    
    print("\nExample workflow:")
    print("  1. Train emulators for GPP, ET, biomass ratio")
    print("  2. Define observations and weights")
    print("  3. Run calibration")
    print("  4. Validate with full FATES simulation")

