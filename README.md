# Multisensor Data Fusion for Urban Heat Island Analysis

Streamlit application that fuses multi-sensor satellite data (Landsat thermal + Sentinel-2 optical) to map and analyze the Urban Heat Island (UHI) effect, with an ML-driven recommendations engine for evidence-based mitigation interventions.

## Features

- **Emissivity-corrected LST** via Split-Window Algorithm
- **5-class land cover classification** (Water, Vegetation, Soil, Urban, Agriculture)
- **UHI Intensity metric** (urban − rural ΔT)
- **WBGT heat stress index** estimation
- **Scientific Recommendations Engine** with ROI estimates, CO₂ budgets, and difficulty ratings
- **Uncertainty / confidence scoring** per metric
- Interactive Folium maps + Plotly visualizations

## Project structure

```
.
├── app.py                  # Streamlit controller + view layer
├── auth.py                 # Google Earth Engine authentication
├── data_processing.py      # Sensor data fusion & preprocessing
├── ml_model.py             # ML classification + UHI metrics
├── features.py             # Feature engineering
├── make_academic_maps.py   # Generates academic-style maps
├── export_maps.py          # Map export utilities
├── requirements.txt
└── .streamlit/
    └── secrets.toml.example
```

## Setup

### 1. Clone & install

```bash
git clone https://github.com/Jamessshhh/multisensor-data-fusion-uhi.git
cd multisensor-data-fusion-uhi
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and add your API keys:

- **Anthropic API key** — https://console.anthropic.com/settings/keys
- **Gemini API key** — https://aistudio.google.com/apikey
- **Google Earth Engine** — register a service account at https://console.cloud.google.com and paste the JSON under `GEE_SERVICE_ACCOUNT_KEY`

### 3. Run

```bash
streamlit run app.py
```

## Security notice

`.streamlit/secrets.toml` is gitignored. Never commit real API keys to this repo. If a key is ever exposed in a commit, revoke it immediately at the provider's console — git history retains it even after deletion.

## Stack

- Streamlit · Google Earth Engine · Anthropic Claude · Google Gemini · Folium · Plotly · Pandas
