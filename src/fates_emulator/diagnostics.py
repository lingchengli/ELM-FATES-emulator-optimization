"""
Diagnostics and interpretability tools for FATES emulators

Includes SHAP analysis, performance metrics, and visualization functions.
All plotting functions save to specified output paths - no hardcoded directories.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap

logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("ticks")
sns.set_context("paper")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


def compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive performance metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    Dict[str, float]
        Dictionary of performance metrics
    """
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'rmse': mean_squared_error(y_true, y_pred, squared=False),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'bias': np.mean(y_pred - y_true),
        'rel_bias': np.mean((y_pred - y_true) / (y_true + 1e-10)),
    }
    
    return metrics


def plot_prediction_scatter(
    y_train: pd.Series,
    y_train_pred: np.ndarray,
    y_test: pd.Series,
    y_test_pred: np.ndarray,
    output_path: Union[str, Path],
    variable_name: str = 'Variable',
    figsize: Tuple[int, int] = (6, 6)
) -> None:
    """
    Plot scatter plot of predictions vs observations.
    
    Parameters
    ----------
    y_train : pd.Series
        Training observations
    y_train_pred : np.ndarray
        Training predictions
    y_test : pd.Series
        Test observations
    y_test_pred : np.ndarray
        Test predictions
    output_path : Union[str, Path]
        Path to save figure
    variable_name : str
        Name of variable for labels
    figsize : Tuple[int, int]
        Figure size
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Training data
    metrics_train = compute_performance_metrics(y_train, y_train_pred)
    ax.scatter(y_train, y_train_pred, alpha=0.6, s=20, label=f"Train: R²={metrics_train['r2']:.3f}, RMSE={metrics_train['rmse']:.2f}")
    
    # Test data
    metrics_test = compute_performance_metrics(y_test, y_test_pred)
    ax.scatter(y_test, y_test_pred, alpha=0.6, s=20, label=f"Test: R²={metrics_test['r2']:.3f}, RMSE={metrics_test['rmse']:.2f}")
    
    # 1:1 line
    all_vals = np.concatenate([y_train, y_test])
    lims = [all_vals.min(), all_vals.max()]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='1:1 line')
    
    ax.set_xlabel(f"Simulated {variable_name}")
    ax.set_ylabel(f"Emulator Prediction")
    ax.legend(fontsize=9, loc='upper left', frameon=False)
    ax.set_title(variable_name, fontsize=12, loc='left')
    
    # Clean up spines
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved prediction scatter plot to {output_path}")


def compute_shap_values(
    model,
    X: pd.DataFrame,
    background_samples: int = 100
) -> shap.Explanation:
    """
    Compute SHAP values for model interpretability.
    
    Parameters
    ----------
    model
        Trained model (must have .predict method)
    X : pd.DataFrame
        Input features
    background_samples : int
        Number of background samples for SHAP explainer
        
    Returns
    -------
    shap.Explanation
        SHAP explanation object
    """
    logger.info("Computing SHAP values...")
    
    # Sample background data if dataset is large
    if len(X) > background_samples:
        X_background = X.sample(n=background_samples, random_state=23)
    else:
        X_background = X
    
    # Create explainer
    try:
        # Try TreeExplainer for tree-based models (faster)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
    except:
        # Fall back to KernelExplainer for other models
        logger.info("Using KernelExplainer (slower)")
        explainer = shap.Explainer(model.predict, X_background)
        shap_values = explainer(X, check_additivity=False)
    
    logger.info("SHAP values computed")
    
    return shap_values


def get_shap_feature_importance(
    shap_values: shap.Explanation,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get feature importance from SHAP values.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values from compute_shap_values
    feature_names : Optional[List[str]]
        Feature names (if None, use from shap_values)
        
    Returns
    -------
    pd.DataFrame
        Feature importance with correlations
    """
    if feature_names is None:
        feature_names = shap_values.feature_names
    
    # Mean absolute SHAP values
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    
    # Correlation between SHAP values and feature values
    correlations = []
    for i in range(len(feature_names)):
        corr = np.corrcoef(shap_values.values[:, i], shap_values.data[:, i])[0, 1]
        correlations.append(corr)
    
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': shap_importance,
        'correlation': correlations,
        'direction': ['positive' if c > 0 else 'negative' for c in correlations]
    })
    
    df_importance = df_importance.sort_values('importance', ascending=False)
    
    return df_importance


def plot_shap_summary(
    shap_values: shap.Explanation,
    output_path: Union[str, Path],
    variable_name: str = 'Variable',
    max_display: int = 15,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot SHAP summary (beeswarm) plot.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values
    output_path : Union[str, Path]
        Path to save figure
    variable_name : str
        Name of target variable
    max_display : int
        Maximum number of features to display
    figsize : Tuple[int, int]
        Figure size
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP Summary: {variable_name}", fontsize=12, loc='left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved SHAP summary plot to {output_path}")


def plot_shap_bar(
    df_importance: pd.DataFrame,
    output_path: Union[str, Path],
    variable_name: str = 'Variable',
    max_display: int = 15,
    figsize: Tuple[int, int] = (6, 8)
) -> None:
    """
    Plot SHAP feature importance as horizontal bar chart.
    
    Parameters
    ----------
    df_importance : pd.DataFrame
        Feature importance from get_shap_feature_importance
    output_path : Union[str, Path]
        Path to save figure
    variable_name : str
        Name of target variable
    max_display : int
        Maximum number of features to display
    figsize : Tuple[int, int]
        Figure size
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select top features
    df_plot = df_importance.head(max_display).copy()
    
    # Color by correlation direction
    colors = ['red' if c > 0 else 'blue' for c in df_plot['correlation']]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(df_plot))
    ax.barh(y_pos, df_plot['importance'], color=colors, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['feature'])
    ax.set_xlabel("mean(|SHAP Value|)", fontsize=11)
    ax.set_title(f"Feature Importance: {variable_name}", fontsize=12, loc='left')
    
    # Add value labels
    for i, (importance, corr) in enumerate(zip(df_plot['importance'], df_plot['correlation'])):
        ax.text(importance, i, f' {importance:.3f}', va='center', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Positive correlation'),
        Patch(facecolor='blue', alpha=0.7, label='Negative correlation')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=False, fontsize=10)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved SHAP bar plot to {output_path}")


def plot_shap_dependence(
    shap_values: shap.Explanation,
    feature_name: str,
    output_path: Union[str, Path],
    interaction_feature: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 5)
) -> None:
    """
    Plot SHAP dependence plot for a single feature.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values
    feature_name : str
        Feature to plot
    output_path : Union[str, Path]
        Path to save figure
    interaction_feature : Optional[str]
        Feature to color by (interaction)
    figsize : Tuple[int, int]
        Figure size
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=figsize)
    shap.dependence_plot(
        feature_name,
        shap_values.values,
        shap_values.data,
        feature_names=shap_values.feature_names,
        interaction_index=interaction_feature,
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved SHAP dependence plot to {output_path}")


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Union[str, Path],
    variable_name: str = 'Variable',
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot residual diagnostics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    output_path : Union[str, Path]
        Path to save figure
    variable_name : str
        Name of variable
    figsize : Tuple[int, int]
        Figure size
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("(a) Residuals vs Predicted", loc='left')
    
    # Residuals vs observed
    axes[1].scatter(y_true, residuals, alpha=0.5, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1].set_xlabel("Observed")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("(b) Residuals vs Observed", loc='left')
    
    # Histogram of residuals
    axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[2].axvline(x=0, color='r', linestyle='--', linewidth=1)
    axes[2].set_xlabel("Residuals")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("(c) Residual Distribution", loc='left')
    
    fig.suptitle(f"Residual Diagnostics: {variable_name}", fontsize=12, y=1.02)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved residual plots to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("FATES-Emulator Diagnostics Example")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(23)
    n_samples = 500
    
    y_true = np.random.normal(50, 10, n_samples)
    y_pred = y_true + np.random.normal(0, 5, n_samples)
    
    # Compute metrics
    metrics = compute_performance_metrics(y_true, y_pred)
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nDiagnostic plots would be saved to specified output paths.")

