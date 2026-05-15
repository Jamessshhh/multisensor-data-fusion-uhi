# Urban Heat Island AI — Agent Instructions (v3)

## Architecture Overview

**Purpose:** Research platform analyzing urban heat islands (UHI) using satellite thermal data with evidence-based interventions.

**Service Boundaries:**
- **`app.py`** (Streamlit MVC Controller): Dashboard UI, recommendations engine, metric visualization
- **`data_processing.py`** (Data Layer): Hybrid thermal architecture using MODIS (1km absolute LST) + Landsat (30m zone delineation) + Sentinel-2 (10m spectral indices)
- **`ml_model.py`** (ML Core): 5-class random forest classifier (Water/Vegetation/Soil/Urban/Agriculture) on 14-band feature space
- **`auth.py`** (Secrets Management): Google Earth Engine API auth with service-account fallback chain
- **`export_maps.py`** (Report Export): Standalone script for high-res map exports (NDVI/NDBI/LST PNGs)

**Data Flow:**
```
[ROI coordinates] → get_satellite_data() → S2/L8 imagery
                    → run_random_forest() → 5-class land cover
                    → calculate_physics() → LST, emissivity, WBGT
                    → Streamlit UI → Recommendations engine
                    → export_maps.py → Publication-grade PNGs
```

## Critical Technical Patterns

### 1. **Thermal Data Authority**
- **MODIS (MOD11A2)**: 1km, 8-day composite → *authoritative* absolute LST (used for reporting, WBGT, UHI intensity)
- **Landsat C2 L2**: 30m → spatial zone masks *only* (uses percentile-relative thresholds, not absolute °C, to sidestep aerosol bias)
- **Sentinel-2**: 10m → all spectral indices (NDVI, NDBI, BSI, etc. for classification)

**Why:** MODIS is accurate absolute temp but coarse. Landsat 30m resolves streets/rooftops. S2 10m provides spectral richness.

### 2. **Emissivity Correction (Physics Critical)**
File: `data_processing.py::_compute_emissivity()`
- Fractional vegetation from NDVI (0–1)
- ε = 0.97 (soil) to 0.99 (veg), weighted by Pv
- Water override: MNDWI > 0 OR NDWI > 0 → ε = 0.992
- Directly inputs LST retrieval accuracy (±2–3°C without it)

### 3. **5-Class ML Model (Imbalanced Stratification)**
File: `ml_model.py::run_random_forest()`
- **Classes**: 0=Water, 1=Vegetation, 2=BareSoil, 3=BuiltUp, 4=Agriculture
- **Features**: 14 indices (NDVI, NDBI, BSI, EVI, UI, NBI, IBI, RI, SAVI, NDWI, MNDWI, BU, NDSI, VgNIRBI)
- **Tuning constants** (top of file):
    - `_N_TREES=120` (diminishing returns >150)
    - `_SAMPLE_PIXELS=500` per class (stratified)
    - `_SMOOTH_RADIUS=3` (morphological closing, noise removal)
    - `_MIN_MAPPING_UNIT=5` pixels (sieve filter)
    - `_VALIDATION_SPLIT=0.2`
- **Key safeguard**: Post-classification NDBI-based confidence refinement (prevents soil/urban confusion)

### 4. **Cloud Masking is Two-Stage**
- **Sentinel-2**: SCL band filtering (keep 4=veg, 5=soil, 6=water, 7=unclass, 11=snow; reject cirrus via B10)
- **Landsat**: QA_PIXEL bitwise decode (remove bits 1,2,3,4,5 for dilated cloud/cirrus/cloud/shadow/snow)
- MODIS: QC_Day bits 0–1 must equal "00" (good)

### 5. **Date Range is Hemisphere-Aware (Seasonal Hottest)**
File: `data_processing.py::_dynamic_date_range()`
- **Northern** (lat > 0): April–October (NH summer)
- **Southern** (lat ≤ 0): October–April (crosses year boundary)
- Falls back to full-year if imagery count too low

## Earth Engine API Patterns

### Authentication Fallback Chain
File: `auth.py::_get_service_account_credentials()`
1. Streamlit secrets (`st.secrets.get()`)
2. Env var path (`GEE_SERVICE_ACCOUNT_KEY` → file)
3. Conventional paths (`credentials/service_account.json`, `.streamlit/service_account.json`)

**Streamlit Cloud note:** Must set `GEE_SERVICE_ACCOUNT_KEY` secret in Streamlit dashboard UI.

### Common EE Patterns in Codebase
```python
# Normalized difference (universal for spectral indices)
img.normalizedDifference(['B8', 'B4'])

# Cloud masking (updateMask chaining)
img.updateMask(clear_mask).updateMask(qa_good)

# Spatial filtering
img.focal_median(radius=3, kernelType='circle')

# Percentile composition (P95 for peak heat)
col.reduce(ee.Reducer.percentile([95]))
```

## Development Workflows

### Running Locally
```bash
# Install deps
pip install -r requirements.txt

# Set credentials (choose one)
export GEE_SERVICE_ACCOUNT_KEY="/path/to/service_account.json"
# OR copy to: ~/.streamlit/service_account.json

# Run dashboard
streamlit run app.py

# Export maps (standalone, requires auth)
python export_maps.py
```

### Testing Thermal Calculations
File: `data_processing.py::calculate_physics()`
- LST in **Kelvin** initially, converted to **Celsius** at export
- WBGT (Wet-Bulb Globe Temp) uses lstC + relative humidity assumption
- Returns: (emissivity, lst_celsius, uhi_intensity, wbgt)

### ML Model Serialization
`run_random_forest()` returns an EE classifier (not pickled). Retraining requires:
1. Sample generation (stratified by class)
2. Feature engineering (14 indices)
3. Classifier training (OOB error logged)
4. Post-filtering (majority filter + sieve)

## Project-Specific Conventions

- **Logging**: All modules use `logging.getLogger("ModuleName_v3")` with ISO timestamps
- **Constants**: Module-level `_SCREAMING_SNAKE_CASE` (e.g., `_N_TREES`, `_MODIS_LST`)
- **Version tracking**: v3 throughout codebase (v2 had single thermal source, early accuracy issues)
- **UI Colors**: Dark sci-fi theme (dark navy/slate) with accent cyan (`#38bdf8`) + status pills (red/orange/green)
- **Metric cards**: Monospace fonts for data (Space Mono), sans-serif headers (Syne)
- **Uncertainty**: All outputs include confidence/uncertainty metrics (not just point estimates)

## When Modifying Core Logic

**Red flags — require caution:**
- Changing `_SAMPLE_PIXELS`, `_N_TREES`: impacts classification accuracy + compute time
- Altering emissivity formula: breaks LST physics (±2–3°C error per step)
- Cloud masking thresholds: increases commission errors (false clear pixels)
- Thermal data source swap: must recalibrate against ground truth

**Safe changes:**
- UI layout/colors (app.py styles)
- Export format/resolution (export_maps.py)
- Logging verbosity
- Constants within ±10% (e.g., _SMOOTH_RADIUS: 3→2 or 3→4)
