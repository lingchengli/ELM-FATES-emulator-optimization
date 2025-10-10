# FATES Setup Guide

This guide explains how to set up FATES (Functionally Assembled Terrestrial Ecosystem Simulator) for use with the FATES-Emulator framework.

## Prerequisites

### E3SM with FATES

You need a working E3SM installation with the FATES component enabled. The framework has been tested with:

- **E3SM v2.0** or later
- **FATES API 13+**
- **ELM** (E3SM Land Model) as host land model

### Check Your Installation

```bash
# Navigate to your E3SM directory
cd /path/to/E3SM

# Check for FATES component
ls components/elm/src/external_models/fates

# Check FATES version/API
grep "FATES_API_VERSION" components/elm/src/external_models/fates/main/FatesVersionMod.F90
```

## Site Setup Requirements

### 1. Domain File

A domain file defines the spatial grid for your site.

**Required variables**:
- `xc`, `yc`: Longitude and latitude
- `xv`, `yv`: Vertex coordinates
- `mask`: Land/ocean mask
- `area`: Grid cell area
- `frac`: Land fraction

**For single-point simulations**:
```bash
# Example: 1x1 grid at K34 tower
Longitude: -60.2091°W → 299.7909°E
Latitude: -2.6091°S
```

### 2. Surface Dataset

Surface dataset contains:
- Soil properties (texture, organic matter, pH)
- Topography
- Land use  
- PFT weights

**Key variables** for FATES:
- `PCT_NAT_PFT`: Natural PFT percentages
- `SOIL_COLOR`: Soil color class
- `ORGANIC`: Organic matter density
- `PCT_SAND`, `PCT_CLAY`: Soil texture

### 3. Forcing Data (DATM)

Atmospheric forcing in DATM (Data Atmosphere) format:

**Required variables**:
- `TBOT`: Air temperature (K)
- `PRECT`: Precipitation rate (mm/s)
- `QBOT`: Specific humidity (kg/kg)
- `WIND`: Wind speed (m/s)
- `FSDS`: Downward shortwave radiation (W/m²)
- `FLDS`: Downward longwave radiation (W/m²)
- `PSRF`: Surface pressure (Pa)

**Temporal resolution**: Hourly or 3-hourly recommended

**Example forcing file naming**:
```
TPHWL3Hrly/clmforc.k34.TBOT.2000-01.nc
TPHWL3Hrly/clmforc.k34.PRECT.2000-01.nc
...
```

### 4. FATES Parameter File

Base parameter file with default PFT traits.

**Download default parameters**:
```bash
wget https://github.com/NGEET/fates/raw/main/parameter_files/fates_params_default.cdl

# Convert CDL to NetCDF
ncgen -o fates_params_base.nc fates_params_default.cdl
```

**Or use site-specific parameters** (Manaus K34 example):
```bash
cp examples/manaus_k34/data/fates_params_base.nc .
```

## Creating a FATES Case

### Option 1: Using E3SM Scripts

```bash
# Set E3SM root
export E3SM_ROOT=/path/to/E3SM

# Navigate to case creation script
cd $E3SM_ROOT/cime/scripts

# Create a case
./create_newcase \
  --case /path/to/cases/manaus_k34_test \
  --compset IELMFATES \
  --res ELM_USRDAT \
  --machine your_machine \
  --project your_project \
  --run-unsupported

cd /path/to/cases/manaus_k34_test
```

### Option 2: Using Workflow Scripts

The framework provides helper scripts:

```bash
cd workflows/step1_sensitivity_analysis
./1.1_create_fates_configs.py --site manaus_k34 --param-file params_001.nc
```

## Configuration

### XML Settings

Key namelists to configure:

```bash
# Case setup
./xmlchange NTASKS=40
./xmlchange STOP_N=20
./xmlchange STOP_OPTION=nyears
./xmlchange REST_N=10
./xmlchange DATM_CLMNCEP_YR_START=2000
./xmlchange DATM_CLMNCEP_YR_END=2020

# Output frequency
./xmlchange HIST_OPTION=nmonths
./xmlchange HIST_N=1
```

### user_nl_elm (ELM Namelist)

```bash
cat >> user_nl_elm << EOF
! Domain and surface data
fsurdat = '/path/to/surfdata_k34_1x1.nc'
 
! FATES specific
use_fates = .true.
fates_paramfile = '/path/to/fates_params_base.nc'

! Output variables
hist_fincl1 = 'GPP', 'EFLX_LH_TOT', 'FSH', 'TLAI', 'FATES_GPP_PF', 
              'FATES_VEGC_PF', 'FATES_MORTALITY_PF', 'FATES_NPLANT_PF'

! Spinup settings (for sensitivity analysis)
hist_nhtfrq = 0
hist_mfilt = 12
EOF
```

### user_nl_datm (DATM Namelist)

```bash
cat >> user_nl_datm << EOF
! Forcing data streams
taxmode = 'cycle', 'cycle', 'cycle'
mapalgo = 'bilinear','bilinear','bilinear'
streams = "datm.streams.txt.CLM1PT.ELM_USRDAT"
EOF
```

### DATM Streams File

Create `user_datm.streams.txt.CLM1PT.ELM_USRDAT`:

```xml
<?xml version="1.0"?>
<file id="stream" version="1.0">
  <dataSource>
    GENERIC
  </dataSource>
  <domainInfo>
    <variableNames>
      time    time
      xc      lon
      yc      lat
      area    area
      mask    mask
    </variableNames>
    <filePath>
      /path/to/domain
    </filePath>
    <fileNames>
      domain_k34_1x1.nc
    </fileNames>
  </domainInfo>
  <fieldInfo>
    <variableNames>
      TBOT   tbot
      PRECT  prec
      QBOT   shum
      WIND   wind
      FSDS   srad
      FLDS   lrad
      PSRF   pres
    </variableNames>
    <filePath>
      /path/to/forcing
    </filePath>
    <fileNames>
      forcing_k34_2000.nc
      forcing_k34_2001.nc
      ...
    </fileNames>
    <offset>
      0
    </offset>
  </fieldInfo>
</file>
```

## PFT Setup for FATES-Emulator

### Two-PFT Configuration

The framework uses a two-PFT setup for tropical forests:

**PFT 1: Early Successional**
- Higher growth rate (high Vcmax, SLA)
- Lower wood density
- Shorter leaf lifespan
- Higher mortality

**PFT 2: Late Successional**
- Lower growth rate
- Higher wood density
- Longer leaf lifespan
- Lower mortality

### Modifying PFT Parameters

```python
import netCDF4 as nc

# Open parameter file
params = nc.Dataset('fates_params_base.nc', 'r+')

# Modify parameters (example)
params['fates_leaf_vcmax25top'][0] = 75.0  # Early successional
params['fates_leaf_vcmax25top'][1] = 50.0  # Late successional

params.close()
```

Or use the framework's parameter handler:

```python
from fates_emulator.preprocessing import parameter_handler

param_values = {
    'fates_leaf_vcmax25top_e': 75.0,
    'fates_leaf_vcmax25top_l': 50.0,
    'fates_wood_density_e': 0.4,
    'fates_wood_density_l': 0.8,
}

parameter_handler.update_fates_params(
    'fates_params_base.nc',
    'fates_params_001.nc',
    param_values
)
```

## Running a Test Simulation

### Build and Submit

```bash
cd /path/to/case/manaus_k34_test

# Setup case
./case.setup

# Build
./case.build

# Submit
./case.submit
```

### Monitor Progress

```bash
# Check queue status
squeue -u $USER

# Check log files
tail -f run/elmlog.txt

# Monitor in real-time
watch -n 10 tail -20 run/elmlog.txt
```

### Verify Output

```bash
# Check output directory
ls archive/elm/hist/

# Quick check with NCO
ncks -M archive/elm/hist/case.elm.h0.2000-01.nc | grep GPP

# Plot with Python
python -c "
import xarray as xr
ds = xr.open_dataset('archive/elm/hist/case.elm.h0.2000-01.nc')
print('GPP range:', ds.GPP.values.min(), '-', ds.GPP.values.max())
"
```

## Typical Runtime

For sensitivity analysis (~1500 simulations):

| Configuration | Time per Sim | Total Time (parallel) |
|---------------|--------------|----------------------|
| **20-year spinup** | 3-4 hours | ~5 days (300 cores) |
| **50-year spinup** | 6-8 hours | ~10 days (300 cores) |

## Troubleshooting

### Common Issues

#### 1. Case Build Fails

**Error**: `Could not find fates_paramfile`

**Solution**: Check path in `user_nl_elm` is absolute and file exists

#### 2. Simulation Crashes

**Error**: `FATES: Negative carbon pool`

**Solution**: 
- Check parameter values are within realistic bounds
- Reduce timestep if needed
- Check for parameter file corruption

#### 3. No Output Files

**Error**: No `.h0.` files in archive

**Solution**:
- Check `hist_fincl1` in user_nl_elm
- Verify simulation ran past first output time
- Check disk quota: `quota -s`

#### 4. PFT Goes Extinct

**Issue**: One PFT completely dominates or disappears

**Explanation**: This is expected in parameter space exploration! The emulator will learn which parameter combinations lead to coexistence.

## Example: Manaus K34 Setup

Complete setup for K34 site is in `examples/manaus_k34/`:

```bash
examples/manaus_k34/
├── data/
│   ├── domain_k34_1x1.nc          # Domain file
│   ├── surfdata_k34_1x1.nc        # Surface data
│   ├── forcing_data/              # DATM forcing (2000-2020)
│   └── fates_params_base.nc       # Base parameters
├── config.yaml                     # Site configuration
└── user_datm.streams.txt.CLM1PT.ELM_USRDAT  # DATM streams
```

To use:

```bash
cp examples/manaus_k34/config.yaml mysite_config.yaml
# Edit paths and settings
./workflows/step1_sensitivity_analysis/1.1_create_fates_configs.py \
  --config mysite_config.yaml
```

## Site-Specific Considerations

### Tropical Sites
- Long spinup needed (50+ years)
- High biomass equilibrium
- Strong seasonal precipitation

### Temperate Sites  
- Phenology important (deciduous vs evergreen)
- Freeze-thaw dynamics
- Shorter spinup possible

### Boreal Sites
- Cold tolerance parameters critical
- Fire disturbance
- Permafrost (if coupled)

## Next Steps

- **Run Sensitivity Analysis**: [Step 1 Guide](03_sensitivity_analysis.md)
- **Configure Site**: Edit `config.yaml` in `examples/template_site/`
- **Review Example**: [Manaus K34 Case Study](06_example_manaus.md)

## Additional Resources

- [FATES Users Guide](https://fates-users-guide.readthedocs.io/)
- [E3SM Tutorial](https://e3sm.org/resources/tutorials/)
- [CTSM/CLM5 Guide](https://escomp.github.io/ctsm-docs/)
- [DATM User Guide](https://esmci.github.io/cime/versions/master/html/users_guide/datm.html)

