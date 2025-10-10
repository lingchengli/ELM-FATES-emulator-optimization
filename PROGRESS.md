# FATES-Emulator Repository: Build Progress

## Completed ✅

### Phase 1: Repository Foundation
- ✅ Main `README.md` with DOI badges and proper citation to Li et al. (2023) GMD paper
- ✅ `LICENSE` (MIT)
- ✅ `.gitignore` (Python, data, HPC-specific)
- ✅ `setup.py` with proper author information
- ✅ `environment.yml` with FLAML AutoML
- ✅ `requirements.txt` with FLAML AutoML
- ✅ Directory structure created

### Phase 1: Documentation
- ✅ `docs/00_overview.md` - Complete workflow overview
- ✅ `docs/01_installation.md` - Installation guide for various systems
- ✅ `docs/02_fates_setup.md` - FATES configuration guide

### Phase 2: Core Framework (AutoML-based, no hardcoded paths)
- ✅ `src/fates_emulator/__init__.py` - Package initialization
- ✅ `src/fates_emulator/sampling.py` - Parameter sampling with ecological constraints
- ✅ `src/fates_emulator/emulator.py` - **FLAML AutoML** for XGBoost training
- ✅ `src/fates_emulator/utils.py` - Configuration loading, logging, helper functions
- ✅ `src/fates_emulator/diagnostics.py` - SHAP analysis and performance metrics
- ✅ `src/preprocessing/__init__.py` - Preprocessing module initialization

### Key Features Implemented
1. **AutoML Integration**: Uses FLAML instead of manual hyperparameter tuning
2. **No Hardcoded Paths**: All functions accept paths as parameters or load from config files
3. **Configuration-Based**: YAML config files for site-specific settings
4. **Proper Citation**: Li et al. (2023) GMD paper properly cited
5. **Professional Structure**: Following Python package best practices

## In Progress 🚧

### Phase 2: Core Framework (Remaining)
- 🚧 `src/fates_emulator/calibration.py` - Parameter optimization
- 🚧 `src/fates_emulator/sensitivity.py` - Sensitivity analysis tools  
- 🚧 `src/preprocessing/fates_output.py` - Extract FATES NetCDF outputs
- 🚧 `src/preprocessing/parameter_handler.py` - Manipulate FATES parameter files
- 🚧 `src/preprocessing/data_prep.py` - Data preparation for ML

### Phase 1: Documentation (Remaining)
- 🚧 `docs/03_sensitivity_analysis.md`
- 🚧 `docs/04_emulator_training.md`
- 🚧 `docs/05_calibration.md`
- 🚧 `docs/06_example_manaus.md`
- 🚧 `docs/07_api_reference.md`

## Pending 📋

### Phase 3: Workflow Scripts
- ⏳ `workflows/step1_sensitivity_analysis/` - Complete Step 1 scripts
- ⏳ `workflows/step2_emulator_training/` - Complete Step 2 scripts
- ⏳ `workflows/step3_calibration/` - Complete Step 3 scripts

### Phase 4: Examples
- ⏳ `examples/manaus_k34/` - K34 site complete example with data
- ⏳ `examples/manaus_k34/notebooks/` - Jupyter notebooks
- ⏳ `examples/template_site/` - Template for new sites

### Phase 5: Polish & Testing
- ⏳ Unit tests in `tests/`
- ⏳ `CONTRIBUTING.md`
- ⏳ `CHANGELOG.md`
- ⏳ GitHub issue templates
- ⏳ CI/CD configuration

## Next Steps

### Immediate (Continue Building)
1. Complete remaining core framework modules:
   - `calibration.py` - Multi-objective optimization
   - `sensitivity.py` - Sensitivity analysis wrapper
   - Preprocessing modules for FATES I/O

2. Create workflow scripts for all 3 steps

3. Add Manaus K34 example with real data subset

### Before GitHub Push
1. Complete documentation
2. Add example notebooks
3. Create template for new sites
4. Add unit tests
5. Test installation on clean environment
6. Prepare small example dataset (< 100 MB for GitHub)

## Notes

### Key Design Decisions
- **FLAML AutoML**: Automatic hyperparameter optimization, saves time
- **Path Agnostic**: No hardcoded paths, all via config or function arguments
- **Modular**: Each step can run independently
- **HPC-Ready**: SLURM scripts included
- **Professional**: Following Python packaging standards

### Data Strategy for GitHub
- Include small example datasets in `examples/manaus_k34/data/`
- Provide download scripts for full datasets
- Document data sources and access

### Testing Strategy
- Unit tests for core functions
- Integration test with example workflow
- Verify on clean conda environment

## References

**Primary Citation**:
Li, L., Fang, Y., Zheng, Z., Shi, M., Longo, M., Koven, C. D., Holm, J. A., Fisher, R. A., McDowell, N. G., Chambers, J., and Leung, L. R.: A machine learning approach targeting parameter estimation for plant functional type coexistence modeling using ELM-FATES (v2.0), Geosci. Model Dev., 16, 4017–4040, https://doi.org/10.5194/gmd-16-4017-2023, 2023.

