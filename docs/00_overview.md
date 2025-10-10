# FATES-Emulator Framework Overview

## Introduction

The FATES-Emulator framework provides a computationally efficient approach to calibrating the Functionally Assembled Terrestrial Ecosystem Simulator (FATES) using machine learning surrogate models. This framework addresses a fundamental challenge in ecosystem modeling: parameter calibration requires thousands of expensive simulations, making traditional optimization methods impractical.

## The Challenge

### Traditional FATES Calibration

- **High computational cost**: Each FATES simulation takes 2-6 hours on HPC systems
- **Large parameter space**: 10+ parameters with complex interactions
- **Multiple objectives**: Must match observations while maintaining ecosystem realism
- **Coexistence constraints**: Parameters must support multi-PFT coexistence
- **Result**: Traditional methods require 10,000+ simulations = 20,000-60,000 CPU-hours

### Our Solution

FATES-Emulator replaces expensive FATES simulations with fast XGBoost machine learning models that predict ecosystem outputs from parameters. This enables:

- **100-1000× speedup** in parameter optimization
- **Rapid exploration** of parameter space
- **Interpretability** via SHAP analysis
- **Flexible objectives** including coexistence constraints

## Three-Stage Workflow

### Stage 1: Sensitivity Analysis

**Goal**: Understand parameter impacts and generate training data

**Steps**:
1. Define parameter ranges and ecological constraints
2. Generate parameter samples using Latin Hypercube Sampling
3. Apply ecological relationships (e.g., early successional > late successional Vcmax)
4. Create FATES parameter files for each sample
5. Run ~1500 FATES simulations on HPC
6. Extract outputs: GPP, ET, biomass, mortality, etc.

**Outputs**:
- Parameter samples CSV
- FATES simulation outputs (NetCDF)
- Initial parameter sensitivity analysis

**Computational cost**: ~3000-9000 CPU-hours (one-time cost)

---

### Stage 2: Emulator Training

**Goal**: Train ML models to predict FATES outputs from parameters

**Steps**:
1. Prepare training data (parameters → outputs)
2. Split into training/testing sets
3. Train separate XGBoost models for each output variable:
   - Gross Primary Production (GPP)
   - Evapotranspiration (ET)
   - Above-Ground Biomass (AGB)
   - Sensible heat flux (FSH)
   - PFT biomass fractions
   - Mortality rates
4. Hyperparameter optimization using Bayesian search
5. Evaluate model performance (R², RMSE)
6. SHAP analysis for interpretability

**Outputs**:
- Trained XGBoost models (.dat files)
- Performance metrics and validation plots
- SHAP importance rankings

**Computational cost**: ~10-30 minutes (one-time training)

---

### Stage 3: Parameter Calibration

**Goal**: Find optimal parameters using emulators

**Steps**:
1. Define calibration objectives:
   - Match flux tower observations (GPP, ET, FSH)
   - Maintain PFT coexistence (biomass ratio 0.1-0.9)
   - Minimize mortality rates
   - Ecological realism constraints
2. Use emulators to rapidly evaluate parameter sets
3. Optimize using gradient-free methods:
   - Genetic algorithms
   - Bayesian optimization
   - Multi-objective optimization
4. Validate best parameters with full FATES runs
5. Analyze trade-offs and uncertainty

**Outputs**:
- Optimized parameter sets
- Calibration diagnostics
- Validation against observations
- Uncertainty quantification

**Computational cost**: ~1-5 hours for optimization + validation runs

---

## Key Components

### Parameter Space

**11 Primary Parameters** (values for early and late successional PFTs):

1. **fates_leaf_vcmax25top**: Maximum carboxylation rate (40-105 μmol/m²/s)
   - Controls photosynthetic capacity
   - Early successional typically > late successional

2. **fates_leaf_slatop**: Specific leaf area (0.005-0.04 m²/gC)
   - Leaf economic spectrum trait
   - Early successional typically > late successional

3. **fates_mort_bmort**: Background mortality (0.005-0.05 /year)
   - Base mortality rate independent of stress
   - Early successional typically > late successional

4. **fates_wood_density**: Wood density (0.2-1.0 g/cm³)
   - Trade-off with growth rate
   - Early successional typically < late successional

5. **fates_leaf_long**: Leaf longevity (0.2-3.0 years)
   - Leaf lifetime before senescence
   - Early successional typically < late successional

6. **fates_alloc_storage_cushion**: Storage allocation (0.8-1.5)
   - Carbon storage buffer
   - Affects stress tolerance

### Output Variables

**Flux Variables**:
- GPP: Gross Primary Production (gC/m²/day)
- ET: Evapotranspiration (mm/day)
- FSH: Sensible heat flux (W/m²)

**State Variables**:
- AGB: Above-ground biomass (kgC/m²)
- PFTbiomass_e: Early successional biomass fraction
- PFTbiomass_l: Late successional biomass fraction
- PFTbiomass_r: Biomass ratio (early/(early+late))

**Demographic Variables**:
- MORTALITY_e/l: Mortality rates by PFT
- RECRUITMENT: New individuals per year

### XGBoost Emulators

**Why XGBoost?**
- Handles non-linear parameter interactions
- Robust to overfitting
- Fast predictions (~milliseconds)
- Built-in feature importance

**Model Architecture**:
- Regression task (continuous outputs)
- Hyperparameters tuned via Bayesian optimization:
  - learning_rate: 0.001-1.0
  - max_depth: 3-8
  - n_estimators: 10-600
  - subsample: 0.5-0.9

**Performance**:
- Typical R² > 0.9 for well-sampled outputs
- RMSE within observational uncertainty
- Cross-validation ensures generalization

### SHAP Analysis

**What is SHAP?**
SHapley Additive exPlanations provide:
- Global feature importance
- Directional relationships (positive/negative)
- Interaction effects

**Use in FATES-Emulator**:
- Identify most influential parameters
- Understand parameter-output relationships
- Validate against ecological knowledge
- Guide calibration strategy

---

## Calibration Objectives

### 1. Match Observations

Minimize difference between emulator predictions and flux tower data:

```
Objective_obs = Σ weights[i] * (predicted[i] - observed[i])²
```

For:
- Daily/monthly GPP
- Daily/monthly ET  
- Sensible heat flux
- Seasonal patterns

### 2. Ensure Coexistence

Maintain both PFTs with realistic biomass fractions:

```
Objective_coex = penalty if (biomass_ratio < 0.1 or biomass_ratio > 0.9)
```

This prevents competitive exclusion.

### 3. Ecological Realism

Enforce known ecological relationships:
- Early successional: higher Vcmax, SLA, mortality
- Late successional: higher wood density, leaf longevity
- Trade-offs between growth and survival

### 4. Multi-objective Optimization

Combine objectives with weights or use Pareto optimization:

```
Total_objective = w1*Objective_obs + w2*Objective_coex + w3*Objective_constraints
```

---

## Computational Efficiency

### Cost Comparison

| Method | Simulations | Time per Sim | Total Time |
|--------|-------------|--------------|------------|
| **Traditional calibration** | 10,000 | 4 hours | 40,000 hours |
| **FATES-Emulator** |
| - Sensitivity analysis | 1,500 | 4 hours | 6,000 hours |
| - Training | N/A | N/A | 0.5 hours |
| - Calibration | 50 (validation) | 4 hours | 200 hours |
| **Total** | **1,550** | | **6,200 hours** |
| **Speedup** | | | **6.5×** |

*Plus ability to explore parameter space and test objectives rapidly*

### Why This Matters

- Enables multi-site calibration
- Supports uncertainty quantification
- Allows rapid testing of hypotheses
- Makes calibration accessible to more researchers

---

## Assumptions and Limitations

### Assumptions

1. **Parameter space is well-sampled**: Emulators interpolate, not extrapolate
2. **FATES outputs are smooth**: Sudden discontinuities are hard to learn
3. **Training simulations reach quasi-equilibrium**: Transient dynamics may differ
4. **Parameter interactions are learnable**: XGBoost can capture non-linearities

### Limitations

1. **Emulator accuracy**: Not perfect substitutes for full FATES runs
2. **Validation needed**: Always validate with full simulations
3. **Site-specific**: Models trained for one site may not transfer
4. **Output-specific**: Need separate models for each output variable
5. **Computational setup**: Requires HPC for initial sensitivity analysis

### Best Practices

- Use 1000-2000 training samples for robust emulators
- Always validate calibrated parameters with full FATES runs
- Check emulator performance (R² > 0.8) before using for calibration
- Include coexistence constraints explicitly in objectives
- Test multiple optimization algorithms
- Quantify uncertainty in optimal parameters

---

## Workflow Diagram (Detailed)

```
┌─────────────────────────────────────────────────────────┐
│ Inputs                                                  │
│ ├─ Base FATES parameter file                           │
│ ├─ Site forcing data (DATM format)                     │
│ ├─ Domain and surface files                            │
│ └─ Parameter ranges and constraints                    │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ Step 1: Sensitivity Analysis                            │
│                                                          │
│ 1.0_generate_parameters.py                              │
│   ├─ Latin Hypercube Sampling                          │
│   ├─ Apply ecological constraints                      │
│   └─ Save parameter samples CSV                        │
│                                                          │
│ 1.1_create_fates_configs.py                            │
│   ├─ Read parameter samples                            │
│   ├─ Create FATES .nc parameter files                  │
│   └─ Create case directories                           │
│                                                          │
│ 1.2_submit_fates_runs.sh                               │
│   ├─ Submit SLURM jobs                                 │
│   ├─ Run FATES simulations (parallel)                  │
│   └─ Wait for completion                               │
│                                                          │
│ 1.3_extract_outputs.py                                 │
│   ├─ Read FATES output NetCDF files                   │
│   ├─ Extract variables (GPP, ET, biomass, etc.)       │
│   └─ Create training dataset                           │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Emulator Training                               │
│                                                          │
│ 2.0_prepare_training_data.py                           │
│   ├─ Load parameters and outputs                       │
│   ├─ Clean data (remove failures)                      │
│   ├─ Train/test split (90/10)                         │
│   └─ Compute derived variables (ratios)                │
│                                                          │
│ 2.1_train_xgb_models.py                                │
│   ├─ For each output variable:                         │
│   │   ├─ Hyperparameter optimization                   │
│   │   ├─ Train XGBoost model                          │
│   │   └─ Save trained model                           │
│                                                          │
│ 2.2_evaluate_emulators.py                              │
│   ├─ Compute R², RMSE on test set                     │
│   ├─ Generate prediction plots                         │
│   └─ Residual analysis                                 │
│                                                          │
│ 2.3_shap_analysis.py                                   │
│   ├─ Compute SHAP values                              │
│   ├─ Feature importance rankings                       │
│   └─ Dependence plots                                  │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: Parameter Calibration                           │
│                                                          │
│ 3.0_define_objectives.py                               │
│   ├─ Load observations                                  │
│   ├─ Define objective functions                        │
│   └─ Set optimization constraints                      │
│                                                          │
│ 3.1_optimize_parameters.py                             │
│   ├─ Load trained emulators                           │
│   ├─ Run optimization algorithm                        │
│   ├─ Evaluate candidates with emulators               │
│   ├─ Track best parameters                            │
│   └─ Save optimization history                         │
│                                                          │
│ 3.2_validate_calibration.py                            │
│   ├─ Run FATES with optimized parameters              │
│   ├─ Compare to observations                           │
│   ├─ Check coexistence                                 │
│   └─ Generate diagnostic plots                         │
│                                                          │
│ 3.3_run_final_fates.sh                                 │
│   ├─ Long simulation with best parameters             │
│   └─ Full evaluation against observations              │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ Outputs                                                  │
│ ├─ Calibrated parameter file (fates_params_calib.nc)   │
│ ├─ Trained emulator models (*.pkl)                     │
│ ├─ Performance metrics and plots                       │
│ ├─ SHAP importance analysis                            │
│ └─ Validation against observations                     │
└─────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Installation**: See [Installation Guide](01_installation.md)
2. **Setup FATES**: See [FATES Setup](02_fates_setup.md)
3. **Run Example**: See [Manaus K34 Case Study](06_example_manaus.md)
4. **Adapt for your site**: Use template in `examples/template_site/`

---

## References

- Fisher, R. A., et al. (2015). Taking off the training wheels: the properties, dynamics, and behaviors of FATES. Geoscientific Model Development, 8(10), 3593-3619.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. ACM SIGKDD.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
- Saltelli, A., et al. (2008). Global Sensitivity Analysis: The Primer. Wiley.

