# FATES-Emulator Workflows

This directory contains the three-step workflow for FATES parameter calibration using machine learning emulators.

## Overview

```
Step 1: Sensitivity Analysis
  ↓
Step 2: Emulator Training (AutoML)
  ↓
Step 3: Parameter Calibration
```

## Step 1: Sensitivity Analysis

**Purpose**: Generate parameter samples and run FATES simulations to create training data.

**Scripts**:
- `1.0_generate_parameters.py` - Generate Latin Hypercube samples with ecological constraints
- `1.1_create_fates_configs.py` - Create FATES parameter files and case directories
- `1.2_submit_fates_runs.sh` - Submit FATES simulations to HPC queue
- `1.3_extract_outputs.py` - Extract variables from FATES NetCDF outputs

**Typical Usage**:
```bash
cd step1_sensitivity_analysis

# 1. Generate parameter samples (~1500 samples)
python 1.0_generate_parameters.py --config ../../examples/manaus_k34/config.yaml \
    --output params/parameter_samples.csv

# 2. Create FATES cases
python 1.1_create_fates_configs.py \
    --params params/parameter_samples.csv \
    --base-param /path/to/fates_params_base.nc \
    --output-dir cases/

# 3. Submit simulations (customize for your HPC system)
./1.2_submit_fates_runs.sh cases/

# 4. Extract outputs once simulations complete
python 1.3_extract_outputs.py \
    --cases-dir cases/ \
    --output outputs/fates_outputs.csv
```

**Output**:
- `parameter_samples.csv` - Parameter values for each simulation
- `fates_outputs.csv` - Extracted FATES variables

---

## Step 2: Emulator Training

**Purpose**: Train machine learning emulators using FLAML AutoML.

**Scripts**:
- `2.0_prepare_training_data.py` - Clean and prepare data for training
- `2.1_train_emulators.py` - Train emulators with FLAML AutoML
- `2.2_evaluate_emulators.py` - Evaluate emulator performance
- `2.3_shap_analysis.py` - SHAP interpretability analysis

**Typical Usage**:
```bash
cd step2_emulator_training

# 1. Prepare training data (optional, can skip if data is clean)
python 2.0_prepare_training_data.py \
    --params ../step1_sensitivity_analysis/params/parameter_samples.csv \
    --outputs ../step1_sensitivity_analysis/outputs/fates_outputs.csv \
    --output prepared_data.csv

# 2. Train emulators (10 min AutoML budget per variable)
python 2.1_train_emulators.py \
    --params ../step1_sensitivity_analysis/params/parameter_samples.csv \
    --outputs ../step1_sensitivity_analysis/outputs/fates_outputs.csv \
    --output-dir models/ \
    --time-budget 600 \
    --variables GPP ET AGB PFTbiomass_r

# 3. Evaluate performance
python 2.2_evaluate_emulators.py \
    --models-dir models/ \
    --output-dir evaluation/

# 4. SHAP analysis
python 2.3_shap_analysis.py \
    --models-dir models/ \
    --output-dir shap_analysis/
```

**Output**:
- `models/*.pkl` - Trained emulator models
- `training_summary.csv` - Performance metrics
- Diagnostic plots (scatter, residuals, SHAP)

---

## Step 3: Parameter Calibration

**Purpose**: Optimize parameters using emulators to match observations and ensure coexistence.

**Scripts**:
- `3.0_define_objectives.py` - Setup calibration objectives
- `3.1_optimize_parameters.py` - Run parameter optimization
- `3.2_validate_calibration.py` - Validate with full FATES runs
- `3.3_run_final_fates.sh` - Run FATES with optimized parameters

**Typical Usage**:
```bash
cd step3_calibration

# 1. Setup objectives (edit observations in config.yaml first)
python 3.0_define_objectives.py \
    --config ../../examples/manaus_k34/config.yaml

# 2. Optimize parameters
python 3.1_optimize_parameters.py \
    --models-dir ../step2_emulator_training/models/ \
    --config ../../examples/manaus_k34/config.yaml \
    --output calibrated_params.csv \
    --n-starts 5

# 3. Validate with FATES
python 3.2_validate_calibration.py \
    --params calibrated_params.csv \
    --base-param /path/to/fates_params_base.nc \
    --output-dir validation/

# 4. Run final simulation
./3.3_run_final_fates.sh validation/best_params.nc
```

**Output**:
- `calibrated_params.csv` - Optimized parameter sets
- `optimization_history.csv` - Optimization trajectory
- Validation plots comparing emulator vs FATES

---

## Quick Start with Example

For Manaus K34 site with example data:

```bash
cd examples/manaus_k34

# Edit config.yaml with your paths
nano config.yaml

# Run complete workflow (if you have pre-computed data)
python run_complete_workflow.py --config config.yaml
```

## Configuration

All workflows use YAML configuration files. See `examples/manaus_k34/config.yaml` for a complete example.

Key configuration sections:
- **paths**: Input/output directories
- **parameters**: Parameter names, bounds, PFT mapping
- **emulator**: AutoML settings (time budget, estimators)
- **calibration**: Observations, weights, constraints
- **fates**: Simulation settings for HPC

## Computational Requirements

| Step | Time | Cores | Storage |
|------|------|-------|---------|
| **Step 1** | 3-10 days | 300-600 | 500 GB - 2 TB |
| **Step 2** | 1-3 hours | 40 | 1-5 GB |
| **Step 3** | 1-5 hours | 10 | < 1 GB |

*Step 1 can be parallelized across many nodes*

## Notes

- **Step 1** is computationally expensive but only done once
- **Step 2** benefits from multiple cores (AutoML parallelization)
- **Step 3** can explore many parameter combinations rapidly
- Always validate final parameters with full FATES simulations

## Customization

To adapt for your site:
1. Copy `examples/template_site/` to your site name
2. Edit `config.yaml` with your paths and settings
3. Provide base FATES parameter file
4. Modify parameter bounds if needed
5. Update observations in config

## Getting Help

- See main documentation: `docs/`
- Example walkthrough: `docs/06_example_manaus.md`
- Open an issue on GitHub

## Citation

Li, L., et al. (2023). A machine learning approach targeting parameter estimation for plant functional type coexistence modeling using ELM-FATES (v2.0). Geosci. Model Dev., 16, 4017–4040. https://doi.org/10.5194/gmd-16-4017-2023

