"""
XGBoost Emulator Training using AutoML (FLAML)

This module provides functions for training machine learning emulators of FATES
outputs using automated machine learning (FLAML) for hyperparameter optimization.

All paths and configurations are passed as arguments - no hardcoded paths.
"""

import pickle
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

logger = logging.getLogger(__name__)


class FATESEmulator:
    """
    FATES emulator using AutoML for training XGBoost models.
    
    Attributes
    ----------
    model : AutoML
        Trained FLAML AutoML model
    target_variable : str
        Name of the target output variable
    feature_names : List[str]
        Names of input features (parameters)
    performance_metrics : Dict
        Training and test performance metrics
    training_time : float
        Time taken for training (seconds)
    """
    
    def __init__(self, target_variable: str, random_state: int = 23):
        """
        Initialize FATES emulator.
        
        Parameters
        ----------
        target_variable : str
            Name of output variable to emulate (e.g., 'GPP', 'ET', 'AGB')
        random_state : int
            Random seed for reproducibility
        """
        self.target_variable = target_variable
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.performance_metrics = {}
        self.training_time = 0.0
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        time_budget: int = 600,
        estimator_list: Optional[List[str]] = None,
        metric: str = 'r2',
        n_jobs: int = -1,
        verbose: int = 1
    ) -> Dict:
        """
        Train emulator using FLAML AutoML.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (parameters)
        y_train : pd.Series
            Training target (FATES output)
        X_test : Optional[pd.DataFrame]
            Test features (if None, will split from training data)
        y_test : Optional[pd.Series]
            Test target
        time_budget : int
            Time budget for AutoML search in seconds (default 600 = 10 min)
        estimator_list : Optional[List[str]]
            List of estimators to try. Default: ['xgboost', 'rf', 'lgbm']
        metric : str
            Metric to optimize ('r2', 'rmse', 'mae')
        n_jobs : int
            Number of parallel jobs (-1 = all cores)
        verbose : int
            Verbosity level (0-3)
            
        Returns
        -------
        Dict
            Performance metrics on training and test sets
        """
        logger.info(f"Training emulator for {self.target_variable}")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Split data if test set not provided
        if X_test is None or y_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, 
                test_size=0.1, 
                random_state=self.random_state
            )
        
        # Default estimator list: focus on tree-based methods
        if estimator_list is None:
            estimator_list = ['xgboost', 'rf', 'lgbm', 'extra_tree']
        
        # Initialize AutoML
        self.model = AutoML()
        
        # Configure and train
        start_time = time.time()
        
        logger.info(f"Starting AutoML training with {time_budget}s budget")
        logger.info(f"Estimators: {estimator_list}")
        
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            task='regression',
            time_budget=time_budget,
            estimator_list=estimator_list,
            metric=metric,
            n_jobs=n_jobs,
            verbose=verbose,
            seed=self.random_state,
            early_stop=True,
            log_file_name=None  # Disable file logging, use logger instead
        )
        
        self.training_time = time.time() - start_time
        
        # Evaluate performance
        self.performance_metrics = self._evaluate_performance(
            X_train, y_train, X_test, y_test
        )
        
        logger.info(f"Training completed in {self.training_time:.2f}s")
        logger.info(f"Best estimator: {self.model.best_estimator}")
        logger.info(f"Test R²: {self.performance_metrics['test_r2']:.4f}")
        logger.info(f"Test RMSE: {self.performance_metrics['test_rmse']:.4f}")
        
        return self.performance_metrics
    
    def _evaluate_performance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """Evaluate model performance on train and test sets."""
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Metrics
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'train_rmse': mean_squared_error(y_train, y_train_pred, squared=False),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'test_rmse': mean_squared_error(y_test, y_test_pred, squared=False),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'best_estimator': self.model.best_estimator,
            'best_config': self.model.best_config,
            'training_time': self.training_time,
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict FATES output from parameters.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features (parameters)
            
        Returns
        -------
        np.ndarray
            Predicted output values
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(X)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save trained emulator to file.
        
        Parameters
        ----------
        filepath : Union[str, Path]
            Path to save file (.pkl)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model': self.model,
            'target_variable': self.target_variable,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'training_time': self.training_time,
            'random_state': self.random_state,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Emulator saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'FATESEmulator':
        """
        Load trained emulator from file.
        
        Parameters
        ----------
        filepath : Union[str, Path]
            Path to saved file (.pkl)
            
        Returns
        -------
        FATESEmulator
            Loaded emulator object
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        emulator = cls(
            target_variable=save_dict['target_variable'],
            random_state=save_dict.get('random_state', 23)
        )
        emulator.model = save_dict['model']
        emulator.feature_names = save_dict['feature_names']
        emulator.performance_metrics = save_dict['performance_metrics']
        emulator.training_time = save_dict['training_time']
        
        logger.info(f"Emulator loaded from {filepath}")
        
        return emulator
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.Series:
        """
        Get feature importance from trained model.
        
        Parameters
        ----------
        importance_type : str
            Type of importance ('gain', 'weight', 'cover')
            Only applicable for tree-based models
            
        Returns
        -------
        pd.Series
            Feature importance values
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        try:
            # Try to get feature importance from the model
            if hasattr(self.model.model, 'feature_importances_'):
                importance = self.model.model.feature_importances_
            elif hasattr(self.model.model.estimator, 'feature_importances_'):
                importance = self.model.model.estimator.feature_importances_
            else:
                logger.warning("Feature importance not available for this model type")
                return pd.Series(index=self.feature_names)
            
            return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return pd.Series(index=self.feature_names)


def train_multiple_emulators(
    df_params: pd.DataFrame,
    df_outputs: pd.DataFrame,
    output_variables: List[str],
    output_dir: Union[str, Path],
    param_columns: Optional[List[str]] = None,
    time_budget_per_model: int = 600,
    test_size: float = 0.1,
    random_state: int = 23,
    n_jobs: int = -1
) -> Dict[str, FATESEmulator]:
    """
    Train emulators for multiple output variables.
    
    Parameters
    ----------
    df_params : pd.DataFrame
        Parameter samples
    df_outputs : pd.DataFrame
        FATES output variables
    output_variables : List[str]
        List of output variables to train emulators for
    output_dir : Union[str, Path]
        Directory to save trained models
    param_columns : Optional[List[str]]
        Columns to use as features (if None, use all)
    time_budget_per_model : int
        Time budget per model in seconds
    test_size : float
        Fraction of data for testing
    random_state : int
        Random seed
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    Dict[str, FATESEmulator]
        Dictionary of trained emulators
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if param_columns is None:
        param_columns = list(df_params.columns)
    
    X = df_params[param_columns]
    emulators = {}
    
    logger.info(f"Training emulators for {len(output_variables)} output variables")
    logger.info(f"Training data: {len(X)} samples × {len(param_columns)} parameters")
    
    for var in output_variables:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training emulator: {var}")
        logger.info(f"{'='*60}")
        
        if var not in df_outputs.columns:
            logger.warning(f"Variable {var} not found in outputs, skipping")
            continue
        
        y = df_outputs[var]
        
        # Remove missing values
        mask = ~(y.isna() | (y <= -1000))
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(y_clean) < 50:
            logger.warning(f"Too few valid samples for {var} ({len(y_clean)}), skipping")
            continue
        
        logger.info(f"Valid samples: {len(y_clean)} / {len(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean,
            test_size=test_size,
            random_state=random_state
        )
        
        # Train emulator
        emulator = FATESEmulator(target_variable=var, random_state=random_state)
        
        try:
            emulator.train(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                time_budget=time_budget_per_model,
                n_jobs=n_jobs,
                verbose=1
            )
            
            # Save emulator
            save_path = output_dir / f"{var}_emulator.pkl"
            emulator.save(save_path)
            
            emulators[var] = emulator
            
        except Exception as e:
            logger.error(f"Failed to train emulator for {var}: {e}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete: {len(emulators)} / {len(output_variables)} successful")
    logger.info(f"{'='*60}")
    
    # Save summary
    summary = []
    for var, emulator in emulators.items():
        summary.append({
            'variable': var,
            'best_estimator': emulator.performance_metrics['best_estimator'],
            'train_r2': emulator.performance_metrics['train_r2'],
            'test_r2': emulator.performance_metrics['test_r2'],
            'test_rmse': emulator.performance_metrics['test_rmse'],
            'training_time': emulator.training_time,
        })
    
    df_summary = pd.DataFrame(summary)
    summary_path = output_dir / 'emulator_summary.csv'
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")
    
    return emulators


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("FATES Emulator Example with FLAML AutoML")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(23)
    n_samples = 500
    
    # Synthetic parameters
    X = pd.DataFrame({
        'vcmax_e': np.random.uniform(40, 105, n_samples),
        'vcmax_l': np.random.uniform(40, 105, n_samples),
        'sla_e': np.random.uniform(0.005, 0.04, n_samples),
        'sla_l': np.random.uniform(0.005, 0.04, n_samples),
    })
    
    # Synthetic output (simple relationship for demo)
    y = 10 + 0.5 * X['vcmax_e'] + 0.3 * X['vcmax_l'] + 100 * X['sla_e'] + np.random.normal(0, 5, n_samples)
    y = pd.Series(y, name='GPP')
    
    print(f"\nGenerated {len(X)} samples")
    print(f"Features: {list(X.columns)}")
    print(f"Target: {y.name}")
    
    # Train emulator
    emulator = FATESEmulator(target_variable='GPP', random_state=23)
    metrics = emulator.train(
        X_train=X,
        y_train=y,
        time_budget=60,  # 1 minute for demo
        verbose=2
    )
    
    print(f"\nTraining Results:")
    print(f"  Best model: {metrics['best_estimator']}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"  Training time: {metrics['training_time']:.2f}s")
    
    # Feature importance
    print(f"\nFeature Importance:")
    importance = emulator.get_feature_importance()
    print(importance)

