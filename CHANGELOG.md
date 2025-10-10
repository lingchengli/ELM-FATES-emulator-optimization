# Changelog

All notable changes to FATES-Emulator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-09

### Added
- Initial release of FATES-Emulator framework
- Core emulator module with FLAML AutoML integration
- Parameter sampling with Latin Hypercube and ecological constraints
- SHAP-based interpretability analysis
- Multi-objective calibration with coexistence constraints
- Preprocessing modules for FATES I/O
- Configuration-based workflow (no hardcoded paths)
- Complete documentation for installation and FATES setup
- Example configuration for Manaus K34 site
- Workflow scripts for Steps 1-3
- Command-line interfaces for key operations
- Professional package structure (pip/conda installable)

### Features
- **AutoML**: FLAML integration for automatic hyperparameter optimization
- **Path-agnostic**: All paths via configuration or function arguments
- **Interpretable**: SHAP analysis built-in
- **Flexible**: Modular design for different sites and objectives
- **HPC-ready**: SLURM scripts and parallel computing support
- **Citable**: Proper attribution to Li et al. (2023) GMD paper

### Documentation
- README with badges and citations
- Installation guide for multiple systems
- FATES setup guide
- Workflow overview with diagrams
- API documentation in docstrings
- Contributing guidelines
- Example scripts and notebooks

### Core Modules
- `fates_emulator.sampling` - Parameter space exploration
- `fates_emulator.emulator` - FLAML AutoML training
- `fates_emulator.diagnostics` - SHAP and performance metrics
- `fates_emulator.calibration` - Multi-objective optimization
- `fates_emulator.sensitivity` - Sensitivity analysis
- `fates_emulator.utils` - Configuration and utilities
- `preprocessing.fates_output` - Extract FATES NetCDF
- `preprocessing.parameter_handler` - Modify parameter files
- `preprocessing.data_prep` - ML data preparation

### Workflow Scripts
- `1.0_generate_parameters.py` - Parameter sampling
- `2.1_train_emulators.py` - AutoML training
- `3.1_optimize_parameters.py` - Calibration
- `run_example.py` - Complete workflow demo

### Dependencies
- Python 3.8+
- FLAML 1.0+ (AutoML)
- XGBoost 1.5+
- LightGBM 3.3+
- scikit-learn 1.0+
- SHAP 0.40+
- SALib 1.4+
- xarray, netCDF4, pandas, numpy

### Citation
Li, L., Fang, Y., Zheng, Z., Shi, M., Longo, M., Koven, C. D., Holm, J. A., Fisher, R. A., McDowell, N. G., Chambers, J., and Leung, L. R.: A machine learning approach targeting parameter estimation for plant functional type coexistence modeling using ELM-FATES (v2.0), Geosci. Model Dev., 16, 4017–4040, https://doi.org/10.5194/gmd-16-4017-2023, 2023.

## [0.0.1] - Development

### In Progress
- Additional workflow scripts for Step 1 (FATES submission)
- More documentation (step-by-step guides)
- Jupyter notebook tutorials
- Unit tests with pytest
- Example datasets
- CI/CD with GitHub Actions

---

## Version History

### Version Numbering
- Major version (X.0.0): Breaking changes
- Minor version (0.X.0): New features, backwards compatible
- Patch version (0.0.X): Bug fixes

### Release Process
1. Update CHANGELOG.md
2. Update version in setup.py
3. Tag release in git
4. Create GitHub release
5. Archive on Zenodo

---

## Links
- [GitHub Repository](https://github.com/your-org/fates-emulator)
- [Documentation](https://github.com/your-org/fates-emulator/tree/main/docs)
- [Issues](https://github.com/your-org/fates-emulator/issues)
- [GMD Paper](https://doi.org/10.5194/gmd-16-4017-2023)

