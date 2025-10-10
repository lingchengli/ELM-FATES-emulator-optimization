# FATES-Emulator Quick Start Guide

## 🚀 Start Using in 5 Minutes!

Your FATES-Emulator framework is **ready to use right now**. Here's how to get started:

---

## Step 1: Install (2 minutes)

```bash
cd /qfs/people/lili400/compy/project/NGT/manaus/script/3.7_git_repo

# Create conda environment
conda env create -f environment.yml
conda activate fates-emulator

# Install package
pip install -e .

# Verify installation
python -c "from fates_emulator import sampling, emulator; print('✓ Installation successful!')"
```

**Expected output**: `✓ Installation successful!`

---

## Step 2: Run Example (3 minutes)

### Option A: Complete Workflow Demo

```bash
python examples/manaus_k34/run_example.py
```

This demonstrates:
- ✅ Parameter sampling (500 samples)
- ✅ AutoML emulator training (2 models)
- ✅ SHAP interpretability analysis
- ✅ Parameter calibration
- ✅ Coexistence checking

**Time**: ~3 minutes  
**Output**: Complete workflow with metrics

### Option B: Individual Components

#### Generate Parameters
```bash
cd workflows/step1_sensitivity_analysis
python 1.0_generate_parameters.py --n-samples 100 --output test_params.csv
```

#### Train Emulator
```bash
cd workflows/step2_emulator_training
python 2.1_train_emulators.py \
  --params your_params.csv \
  --outputs your_outputs.csv \
  --output-dir models/ \
  --time-budget 300
```

#### Calibrate Parameters
```bash
cd workflows/step3_calibration
python 3.1_optimize_parameters.py \
  --models-dir ../step2_emulator_training/models/ \
  --output calibrated_params.csv
```

---

## Step 3: Use with Your Data

### Load Your Existing Data

```python
from fates_emulator.emulator import FATESEmulator
import pandas as pd

# Load your parameter samples and FATES outputs
df_params = pd.read_csv('/path/to/your/parameter_samples.csv')
df_outputs = pd.read_csv('/path/to/your/fates_outputs.csv')

# Train emulator with AutoML (10 min budget)
emulator = FATESEmulator(target_variable='GPP', random_state=23)
emulator.train(
    X_train=df_params,
    y_train=df_outputs['GPP'],
    time_budget=600,  # 10 minutes
    verbose=2
)

# Save emulator
emulator.save('GPP_emulator.pkl')

# Evaluate performance
print(f"Test R²: {emulator.performance_metrics['test_r2']:.4f}")
print(f"Best model: {emulator.performance_metrics['best_estimator']}")
```

### Run SHAP Analysis

```python
from fates_emulator import diagnostics

# Compute SHAP values
shap_values = diagnostics.compute_shap_values(emulator.model, df_params)

# Get feature importance
importance = diagnostics.get_shap_feature_importance(shap_values)
print(importance.head())

# Save plots
diagnostics.plot_shap_summary(shap_values, 'shap_summary.pdf', 'GPP')
diagnostics.plot_shap_bar(importance, 'shap_importance.pdf', 'GPP')
```

### Calibrate Parameters

```python
from fates_emulator import calibration

# Define observations
observations = {
    'GPP': 8.5,  # gC/m²/day
    'ET': 3.2    # mm/day
}

# Load emulators
emulators = {
    'GPP': FATESEmulator.load('GPP_emulator.pkl'),
    'ET': FATESEmulator.load('ET_emulator.pkl')
}

# Run calibration
result = calibration.calibrate_parameters(
    emulators=emulators,
    param_names=list(df_params.columns),
    param_bounds=[(40, 105), (40, 105), ...],  # Your bounds
    observations=observations,
    weights={'observations': 1.0, 'coexistence': 1.0},
    n_iterations=1000
)

# Print results
print(f"Objective: {result['objective_value']:.6f}")
print(f"Best parameters: {result['best_parameters']}")
print(f"Predictions: {result['predictions']}")
```

---

## 📚 Documentation

- **Overview**: `docs/00_overview.md` - Complete workflow description
- **Installation**: `docs/01_installation.md` - Detailed setup guide
- **FATES Setup**: `docs/02_fates_setup.md` - FATES configuration
- **Workflows**: `workflows/README.md` - Step-by-step usage
- **API**: Check docstrings in source code

---

## 🎯 Common Use Cases

### 1. Quick Parameter Screening
```python
from fates_emulator import sampling

# Generate samples
samples = sampling.generate_fates_parameter_samples(n_samples=100)
samples.to_csv('quick_test_params.csv')
```

### 2. Train Multiple Emulators
```python
from fates_emulator.emulator import train_multiple_emulators

emulators = train_multiple_emulators(
    df_params=df_params,
    df_outputs=df_outputs,
    output_variables=['GPP', 'ET', 'AGB', 'PFTbiomass_r'],
    output_dir='models/',
    time_budget_per_model=600
)
```

### 3. Batch Calibration
```python
from fates_emulator.calibration import multi_start_calibration

results = multi_start_calibration(
    emulators=emulators,
    param_names=param_names,
    param_bounds=param_bounds,
    observations=observations,
    n_starts=10  # Try 10 different starting points
)

# Best result
best = results[0]
print(f"Best objective: {best['objective_value']:.6f}")
```

---

## ⚙️ Configuration

### Edit Site Config

Copy and edit the example configuration:

```bash
cp examples/manaus_k34/config.yaml my_site_config.yaml
nano my_site_config.yaml
```

Key sections to update:
- `paths.project_dir` - Your working directory
- `paths.fates_params_base` - Your FATES parameter file
- `paths.observations` - Your observation data
- `parameters.param_list` - Your parameter ranges
- `calibration.observations` - Your target values

### Load Config in Python

```python
from fates_emulator import utils

config = utils.load_config('my_site_config.yaml')
params = config['parameters']['param_list']
observations = config['calibration']['observations']
```

---

## 🔧 Troubleshooting

### Import Errors
```bash
# Make sure you're in the right environment
conda activate fates-emulator

# Reinstall if needed
pip install -e .
```

### FLAML Issues
```bash
# FLAML might not install via conda on some systems
pip install flaml
```

### Memory Errors
```python
# Reduce n_jobs if running out of memory
emulator.train(..., n_jobs=10)  # Instead of -1 (all cores)
```

---

## 📊 Performance Tips

### Speed Up Training
- Use `time_budget=300` (5 min) for quick tests
- Use `time_budget=1800` (30 min) for production
- Reduce `n_samples` in data if > 2000

### Improve Accuracy
- Increase training data (>1000 samples recommended)
- Longer AutoML budget (>10 min)
- Check for failed simulations in data

### Save Results
```python
# Save everything for reproducibility
emulator.save('model.pkl')
df_params.to_csv('params.csv')
importance.to_csv('importance.csv')
```

---

## 🎓 Learn More

### Methodology Paper
Li, L., et al. (2023). A machine learning approach targeting parameter estimation for plant functional type coexistence modeling using ELM-FATES (v2.0). Geosci. Model Dev., 16, 4017–4040. https://doi.org/10.5194/gmd-16-4017-2023

### Example Papers Using Similar Methods
- Check `docs/` for references
- See Li et al. 2023 GMD for full methodology

### FATES Documentation
- https://fates-users-guide.readthedocs.io/
- https://e3sm.org/

---

## 💬 Get Help

1. **Check documentation**: `docs/` folder
2. **Run example**: `python examples/manaus_k34/run_example.py`
3. **Open issue**: GitHub issues (when repo is public)
4. **Email**: lingcheng.li@pnnl.gov

---

## ✅ Quick Checklist

Before your first real run:

- [ ] Installed conda environment
- [ ] Ran example successfully
- [ ] Prepared your parameter samples CSV
- [ ] Prepared your FATES outputs CSV
- [ ] Edited config.yaml with your paths
- [ ] Checked parameter columns match
- [ ] Set reasonable time budgets

---

**Ready to go!** 🚀

Start with the example, then adapt for your data. Good luck with your FATES calibration!

**Location**: `/qfs/people/lili400/compy/project/NGT/manaus/script/3.7_git_repo/`

