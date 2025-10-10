# Installation Guide

This guide covers installation of the FATES-Emulator framework on different systems.

## System Requirements

### Minimum Requirements
- **OS**: Linux (tested on RHEL/CentOS 7+, Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 16 GB minimum, 32 GB recommended
- **Disk Space**: 50 GB for examples, 500 GB+ for full workflow
- **CPU**: Multi-core processor (emulator training uses 40 cores by default)

### For FATES Simulations
- **HPC Access**: Required for Step 1 (Sensitivity Analysis)
- **E3SM/FATES**: Pre-installed E3SM with FATES component
- **Scheduler**: SLURM (scripts provided) or PBS/LSF (adapt scripts)

## Installation Methods

### Method 1: Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/fates-emulator.git
cd fates-emulator

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate fates-emulator

# Verify installation
python -c "import xgboost, shap, SALib; print('All packages imported successfully!')"
```

### Method 2: Pip with Virtual Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/fates-emulator.git
cd fates-emulator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Method 3: Manual Installation

```bash
# Install dependencies individually
pip install numpy pandas scipy
pip install xgboost scikit-learn scikit-optimize
pip install shap SALib
pip install xarray netCDF4
pip install matplotlib seaborn
pip install pyyaml tqdm jupyter
```

## HPC-Specific Setup

### On NERSC (Cori/Perlmutter)

```bash
# Load modules
module load python/3.9
module load gcc/11.2.0

# Create conda environment in your home directory
cd $HOME
conda env create -f fates-emulator/environment.yml

# Activate in your job scripts
conda activate fates-emulator
```

### On NCAR (Cheyenne/Casper)

```bash
# Load modules
module load ncarenv/1.3
module load python/3.9.9
module load ncarcompilers/0.5.0

# Create conda environment
conda env create -f fates-emulator/environment.yml
conda activate fates-emulator
```

### On PNNL (Compy)

```bash
# Load modules
module load python/3.9.7
module load gcc/9.3.0

# Create conda environment
conda env create -f fates-emulator/environment.yml
conda activate fates-emulator
```

## Verify Installation

Run the check script to verify all dependencies:

```bash
cd fates-emulator
python scripts/check_dependencies.py
```

Expected output:
```
Checking FATES-Emulator dependencies...
✓ Python 3.8+
✓ NumPy 1.21.0+
✓ Pandas 1.3.0+
✓ XGBoost 1.5.0+
✓ scikit-learn 1.0.0+
✓ SHAP 0.40.0+
✓ SALib 1.4.0+
✓ xarray 0.19.0+
✓ netCDF4 1.5.7+
✓ Matplotlib 3.4.0+

All dependencies satisfied!
```

## Test Installation

### Quick Test

```python
# test_installation.py
from fates_emulator import sampling, emulator, diagnostics

# Generate test parameter samples
params = sampling.generate_lhs_samples(
    param_names=['vcmax', 'sla'],
    bounds=[[40, 105], [0.005, 0.04]],
    n_samples=100
)
print(f"Generated {len(params)} parameter samples")
print("Installation successful!")
```

Run the test:
```bash
python test_installation.py
```

### Run Example

```bash
cd examples/manaus_k34
python run_complete_workflow.py --quick-test
```

This runs a quick test with pre-computed data (~5 minutes).

## Troubleshooting

### Common Issues

#### 1. XGBoost Import Error

**Error**: `ImportError: libgomp.so.1: cannot open shared object file`

**Solution**:
```bash
# Install OpenMP library
# Ubuntu/Debian:
sudo apt-get install libgomp1

# RHEL/CentOS:
sudo yum install libgomp

# Or use conda:
conda install -c conda-forge libgomp
```

#### 2. NetCDF4 Installation Fails

**Error**: `OSError: NetCDF: HDF5 error`

**Solution**:
```bash
# Use conda for netCDF4
conda install -c conda-forge netcdf4

# Or set library paths
export HDF5_DIR=/path/to/hdf5
export NETCDF4_DIR=/path/to/netcdf
pip install netCDF4
```

#### 3. SHAP Compilation Issues

**Error**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution**:
```bash
# Use pre-built wheels
pip install --only-binary :all: shap

# Or use conda
conda install -c conda-forge shap
```

#### 4. Memory Errors During Training

**Error**: `MemoryError` or `Killed` during XGBoost training

**Solution**:
- Reduce training data size in config
- Reduce `n_jobs` parameter (default 40 → 10)
- Use a machine with more RAM
- Enable memory-mapped mode in XGBoost

#### 5. Permission Errors on HPC

**Error**: `Permission denied` when creating directories

**Solution**:
```bash
# Ensure you're working in your scratch or home directory
cd $SCRATCH  # or cd $HOME
git clone https://github.com/yourusername/fates-emulator.git
```

### Getting Help

If you encounter issues:

1. **Check the FAQ**: See [docs/faq.md](faq.md)
2. **Search issues**: https://github.com/yourusername/fates-emulator/issues
3. **Ask for help**: Open a new issue with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce
4. **Email**: your.email@institution.edu

## Optional Components

### Jupyter Lab (for Notebooks)

```bash
conda install -c conda-forge jupyterlab
jupyter lab
```

Then open notebooks in `examples/manaus_k34/notebooks/`

### ParFlow Coupling (Advanced)

If using coupled FATES-ParFlow:

```bash
pip install parflowio
```

### Development Tools

For contributing to the codebase:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Pre-commit hooks
pip install pre-commit
pre-commit install

# Code formatting
black src/
flake8 src/
```

## Update Installation

To update to the latest version:

```bash
cd fates-emulator
git pull origin main

# Update dependencies
conda env update -f environment.yml
# or
pip install -r requirements.txt --upgrade
```

## Uninstall

```bash
# Remove conda environment
conda env remove -n fates-emulator

# Or remove virtual environment
rm -rf venv/

# Remove repository
cd ..
rm -rf fates-emulator/
```

## Next Steps

- **Setup FATES**: [FATES Setup Guide](02_fates_setup.md)
- **Run Example**: [Manaus K34 Case Study](06_example_manaus.md)
- **Understand Workflow**: [Overview](00_overview.md)

## System-Specific Notes

### MacOS (Apple Silicon M1/M2)

XGBoost and other packages work on ARM64, but you may need:

```bash
# Install Rosetta 2 if needed
softwareupdate --install-rosetta

# Use x86_64 conda
CONDA_SUBDIR=osx-64 conda env create -f environment.yml
conda activate fates-emulator
conda config --env --set subdir osx-64
```

### Windows (WSL2)

Install Windows Subsystem for Linux 2, then follow Linux instructions:

```bash
# In Windows PowerShell (admin)
wsl --install -d Ubuntu

# Then in WSL Ubuntu
git clone https://github.com/yourusername/fates-emulator.git
cd fates-emulator
conda env create -f environment.yml
```

Note: FATES itself requires Linux, so WSL or Docker is necessary.

