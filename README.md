# ELM-FATES-Emulator: Machine Learning Framework for Land Surface and Ecosystem Model Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5194%2Fgmd--16--4017--2023-blue)](https://doi.org/10.5194/gmd-16-4017-2023)
[![GMD Paper](https://img.shields.io/badge/Paper-GMD%202023-green)](https://gmd.copernicus.org/articles/16/4017/2023/)

## 📖 Citation

**This framework was developed and published via:**

Li, L., Fang, Y., Zheng, Z., Shi, M., Longo, M., Koven, C. D., Holm, J. A., Fisher, R. A., McDowell, N. G., Chambers, J., and Leung, L. R.: A machine learning approach targeting parameter estimation for plant functional type coexistence modeling using ELM-FATES (v2.0), *Geosci. Model Dev.*, 16, 4017–4040, https://doi.org/10.5194/gmd-16-4017-2023, 2023.

<details>
<summary>BibTeX</summary>

```bibtex
@article{Li2023,
  author = {Li, L. and Fang, Y. and Zheng, Z. and Shi, M. and Longo, M. and Koven, C. D. and Holm, J. A. and Fisher, R. A. and McDowell, N. G. and Chambers, J. and Leung, L. R.},
  title = {A machine learning approach targeting parameter estimation for plant functional type coexistence modeling using ELM-FATES (v2.0)},
  journal = {Geoscientific Model Development},
  volume = {16},
  year = {2023},
  pages = {4017--4040},
  doi = {10.5194/gmd-16-4017-2023}
}
```
</details>

---

## 🌟 Overview

A comprehensive AutoML-powered framework for calibrating the Functionally Assembled Terrestrial Ecosystem Simulator (FATES). This approach uses **FLAML (Fast Lightweight AutoML)** to automatically train optimized machine learning emulators, dramatically reducing computational costs while enabling optimization for ecosystem coexistence and multiple ecological objectives.

## 🌟 Key Features

- **AutoML-Powered**: Uses **FLAML (Fast Lightweight AutoML)** for automatic model selection and hyperparameter optimization - no manual tuning needed!
- **Fast Surrogate Modeling**: ML emulators replace expensive FATES simulations
- **Ecosystem Coexistence Focus**: Optimize parameters to maintain multi-PFT (Plant Functional Type) coexistence and biodiversity
- **Interpretable AI**: SHAP (SHapley Additive exPlanations) values reveal how parameters influence model outputs
- **Modular Workflow**: Three-stage pipeline adaptable to different sites and calibration objectives
- **HPC-Ready**: Includes SLURM job submission scripts for high-performance computing environments
- **Validated Approach**: Tested and validated at Manaus K34 flux tower site in the Amazon

## 🔬 Workflow

Traditional ecosystem model calibration requires thousands of simulations, making it computationally prohibitive. FATES-Emulator addresses this by:

1. **Sensitivity Analysis**: Sample parameter space and run FATES simulations to understand parameter impacts
2. **Emulator Training**: Use **FLAML AutoML** to automatically train and optimize ML models (XGBoost, LightGBM, etc.) that predict FATES outputs from parameters (GPP, ET, biomass, mortality, etc.)
3. **Calibration**: Use emulators to rapidly explore parameter space and optimize for observations and coexistence

### Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: Sensitivity Analysis                                       │
│  ├─ Generate parameter samples (Latin Hypercube)                   │
│  ├─ Create FATES parameter files                                   │
│  ├─ Run FATES simulations (~1500 runs)                             │
│  └─ Extract outputs (GPP, ET, biomass, mortality, etc.)            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: Emulator Training (AutoML)                                 │
│  ├─ Prepare training data (parameters → outputs)                   │
│  ├─ Use FLAML AutoML to train models for each output variable      │
│  ├─ Automatic model selection (XGBoost, LightGBM, RF, etc.)        │
│  ├─ Automatic hyperparameter optimization                          │
│  ├─ Evaluate model performance (R², RMSE)                          │
│  └─ SHAP analysis for interpretability                             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: Parameter Calibration                                      │
│  ├─ Define objectives (match observations, ensure coexistence)     │
│  ├─ Use emulators to explore parameter space                       │
│  ├─ Optimize parameters (gradient-free methods)                    │
│  ├─ Validate with full FATES simulations                           │
│  └─ Analyze trade-offs and uncertainty                             │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/lingchengli/fates-emulator-optimization.git
cd fates-emulator-optimization

# Create conda environment
conda env create -f environment.yml
conda activate fates-emulator

# Or install with pip
pip install -r requirements.txt
```

### Run the Manaus K34 Example

```bash
cd examples/manaus_k34
python run_example.py
```

This will:
- Load pre-computed sensitivity analysis data
- Use **FLAML AutoML** to train optimized emulators for GPP, ET, AGB, and biomass ratios
- Optimize parameters for flux tower observations and PFT coexistence
- Generate diagnostic plots and SHAP analyses

## 📚 Documentation

- **[Installation Guide](docs/01_installation.md)** - Detailed setup instructions
- **[FATES Setup](docs/02_fates_setup.md)** - Configure E3SM/FATES for your site
- **[Sensitivity Analysis](docs/03_sensitivity_analysis.md)** - Step 1 details
- **[Emulator Training](docs/04_emulator_training.md)** - Step 2 details
- **[Calibration](docs/05_calibration.md)** - Step 3 details
- **[Manaus K34 Case Study](docs/06_example_manaus.md)** - Complete walkthrough
- **[API Reference](docs/07_api_reference.md)** - Code documentation

## 🌴 Example: Manaus K34 Amazon Site

The repository includes a complete case study calibrating FATES for the K34 flux tower in the Amazon rainforest:

- **Location**: 2.61°S, 60.21°W, Central Amazon
- **Ecosystem**: Tropical moist broadleaf forest
- **PFTs**: Early successional vs. late successional species
- **Observations**: Eddy covariance GPP, ET, sensible heat flux (2002-2020)
- **Key Challenge**: Maintaining coexistence between competing plant functional types

See `examples/manaus_k34/` for data and notebooks.

## 📊 Key Parameters Calibrated

| Parameter | Description | PFT | Range |
|-----------|-------------|-----|-------|
| `fates_leaf_vcmax25top` | Maximum carboxylation rate | Early/Late | 40-105 μmol/m²/s |
| `fates_leaf_slatop` | Specific leaf area | Early/Late | 0.005-0.04 m²/gC |
| `fates_mort_bmort` | Background mortality rate | Early/Late | 0.005-0.05 /year |
| `fates_wood_density` | Wood density | Early/Late | 0.2-1.0 g/cm³ |
| `fates_leaf_long` | Leaf longevity | Early/Late | 0.2-3.0 years |
| `fates_alloc_storage_cushion` | Storage allocation cushion | Both | 0.8-1.5 |

## 🎯 Calibration Objectives

The framework supports multiple calibration objectives:

1. **Match Observations**: Minimize difference between simulated and observed fluxes (GPP, ET, etc.)
2. **Ensure Coexistence**: Maintain biomass ratios within 0.1-0.9 range for both PFTs
3. **Ecological Realism**: Constrain parameter relationships (e.g., early successional has higher Vcmax)
4. **Multi-objective**: Combine objectives with weights or Pareto optimization

## 🔧 Customization

To adapt this framework for your site:

1. **Prepare your data**: FATES parameter file, forcing data, observations (optional)
2. **Configure**: Edit `config.yaml` with site information and parameter ranges
3. **Run sensitivity**: Generate parameter samples and FATES outputs
4. **Train emulators**: Use your simulation data
5. **Calibrate**: Define objectives and optimize

See `examples/template_site/` for a starting template.

## 📈 Performance

Typical performance on HPC systems:

- **Sensitivity Analysis**: ~1500 FATES runs × 2-6 hours = 3000-9000 CPU-hours
- **Emulator Training**: ~10-30 minutes for all variables
- **Calibration**: ~1-5 hours for optimization (vs. weeks for direct calibration)

**Speedup**: 100-1000× faster than traditional methods for parameter optimization.

## 🧪 Requirements

### Software Dependencies

- Python 3.8+
- **FLAML >= 1.0.0** (AutoML framework)
- XGBoost >= 1.5.0 (ML model)
- LightGBM >= 3.3.0 (ML model)
- scikit-learn >= 1.0.0
- SHAP >= 0.40.0 (interpretability)
- pandas, numpy, xarray
- netCDF4
- matplotlib, seaborn
- SALib (for parameter sampling)

### FATES/E3SM Setup

- E3SM version with FATES support (tested with E3SM v2.0+)
- FATES parameter file for your PFTs
- Site-specific forcing data (DATM format)
- Domain and surface datasets

See [FATES Setup Guide](docs/02_fates_setup.md) for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Lingcheng Li** - Lead Developer - Pacific Northwest National Laboratory
- Contributors welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

## 🙏 Acknowledgments

This framework builds upon the methodology published in:
> Li et al. (2023). A machine learning approach targeting parameter estimation for plant functional type coexistence modeling using ELM-FATES (v2.0). Geoscientific Model Development, 16, 4017–4040.

Special thanks to:
- FATES Development Team (R. Fisher, J. Holm, M. Longo, and contributors)
- E3SM Land Model (ELM) Development Team
- K34 flux tower operators and data providers (LBA Program)
- Co-authors: Y. Fang, Z. Zheng, M. Shi, C. D. Koven, N. G. McDowell, J. Chambers, L. R. Leung
- Funding: U.S. Department of Energy, Office of Science, Biological and Environmental Research

## 📧 Contact

For questions and support:
- Open an issue on GitHub: [Issues](https://github.com/lingchengli/fates-emulator-optimization/issues)
- Email: lingcheng.li@pnnl.gov

## 🔗 Related Resources

- [FATES Documentation](https://fates-users-guide.readthedocs.io/)
- [E3SM Documentation](https://e3sm.org/)
- [FLAML Documentation](https://microsoft.github.io/FLAML/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)

