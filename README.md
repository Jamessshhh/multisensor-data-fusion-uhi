# Multisensor Data Fusion for Urban Heat Island Analysis

**AI-driven satellite data fusion for identifying and prioritizing urban cooling interventions**

I built this to answer a question that most UHI mapping tools stop short of: once you've found the hot spots, what do you actually *do* about them? This app fuses Sentinel-2 optical and Landsat-8 thermal satellite data through Google Earth Engine to map Urban Heat Island intensity, then goes a step further and recommends specific interventions — cool roofs, solar PV zones, afforestation sites — at 10 m resolution, with area totals and impact estimates attached.

The method behind it was published at ICIRCA 2026:

> R. Mahara, K. Tandon, and C. Venkatesh, **"AI-Driven Satellite Data Fusion for Urban Heat Island Mitigation,"** *Proceedings of the 7th International Conference on Inventive Research in Computing Applications (ICIRCA 2026)*, IEEE, pp. 453–460, 2026.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Results at a glance

| Metric | Value |
|---|---|
| Random Forest overall accuracy | **91.3%** (Kappa = 0.88) |
| vs. SVM (RBF) baseline | 87.2% OA |
| vs. Naïve Bayes baseline | 83.6% OA |
| Misclassification reduction from sensor fusion | **18.4%** vs. best single-sensor baseline |
| Rooftop false-positive reduction from GLCM texture | **23.1%** |
| Study area | SRM IST campus, Kattankulathur, India (12.82°N, 80.04°E) |

| Intervention Zone | Area | Estimated Impact |
|---|---|---|
| Cool Roofs | ~45,000 m² | LST reduction: 3–5 °C |
| Solar PV Potential | ~72,000 m² | Capacity: ~10,800 kW |
| Afforestation | ~38,000 m² | CO₂ offset: 83,600 kg/yr |

---

## System architecture

Five stages, start to finish — data acquisition, pre-processing, classification, decision-level fusion, and results delivery — all running inside Earth Engine before anything hits the Streamlit UI.

![System architecture: data sources through pre-processing, classification, fusion engine, and results](assets/architecture.png)

---

## How it works

1. **Multi-sensor acquisition** — Pull Sentinel-2 SR Harmonized (10 m, bands B2–B4, B8, B11, B12) and Landsat-8 Collection 2 Level-2 thermal data (Band ST_B10, 30 m) for the region of interest, with cloud masking below 30% cover.
2. **Feature engineering** — Compute NDVI and NDBI, combine with the six raw bands into an 8-dimensional feature space. This alone improves built-up/bare-soil separability by 18.4% over either sensor used on its own.
3. **Random Forest classification** — A 50-tree Random Forest (the tree count came out of an OOB-error convergence study, not a guess) classifies each pixel as Water, Vegetation, or Built-up, trained on 1,000 stratified samples from ESA WorldCover labels.
4. **GLCM texture analysis** — GLCM entropy on the NIR band tells apart structurally smooth surfaces (rooftops, roads) from rough ones (canopy, scrubland) using a 3.5-bit entropy threshold. This cuts rooftop false positives by 23.1%.
5. **Thermal calibration & decision fusion** — Landsat-8 digital numbers get converted to calibrated LST (°C), then combined with the RF land-cover mask and the GLCM roughness layer to assign each eligible pixel to an intervention zone based on adjustable thermal thresholds (`T_crit`, `T_high`).
6. **Delivery** — Everything lands in an interactive Streamlit dashboard: a color-coded action map plus square-meter-level analytics — area totals, predicted impact, temperature distribution.

## Sample outputs

These are from the primary ROI — SRM IST campus, Kattankulathur, Tamil Nadu.

**NDVI (Vegetation Index)** — low NDVI (<0.2) marks built-up or bare-soil surfaces.
![NDVI map of the Kattankulathur study area](assets/ndvi.png)

**NDBI (Built-Up Index)** — positive values (>0.1) mark concrete or asphalt surfaces.
![NDBI map of the Kattankulathur study area](assets/ndbi.png)

**Land Surface Temperature** — calibrated LST shows thermal hotspots up to 44 °C in built-up zones, roughly 10 °C above adjacent vegetation and water.
![Land surface temperature map of the Kattankulathur study area](assets/lst.png)

## Features

- **Emissivity-corrected LST** via radiometric Split-Window calibration of Landsat-8 Band 10
- **Land cover classification** (Water, Vegetation, Built-up) via a 50-tree Random Forest on an 8-dimensional spectral feature space
- **GLCM texture entropy** layer for smooth-vs-rough surface discrimination
- **UHI Intensity metric** (urban − rural ΔT)
- **WBGT heat stress index** estimation
- **Scientific Recommendations Engine** — cool roof / solar PV / afforestation zoning with area totals, ROI estimates, CO₂ budgets, and difficulty ratings
- **Uncertainty / confidence scoring** per metric
- Interactive Folium maps + Plotly visualizations

## Project structure

```
.
├── app.py                  # Streamlit controller + view layer
├── auth.py                 # Google Earth Engine authentication
├── data_processing.py      # Sensor data fusion & preprocessing (NDVI/NDBI, GLCM, LST)
├── ml_model.py              # Random Forest classification + UHI metrics
├── features.py              # Feature engineering
├── make_academic_maps.py   # Publication-style map rendering (used to generate the figures above)
├── export_maps.py           # Map export utilities
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

Needs Python 3.10+ and a Google Earth Engine–enabled Google Cloud project.

### 2. Configure secrets

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and add:

- **Google Earth Engine** — register a service account at [console.cloud.google.com](https://console.cloud.google.com) and paste the JSON under `GEE_SERVICE_ACCOUNT_KEY`. This one's non-negotiable — nothing runs without it.
- **Anthropic API key** — [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys). Powers the Recommendations Engine's narrative intervention summaries.
- **Gemini API key** — [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Secondary provider for the Recommendations Engine.

### 3. Run

```bash
streamlit run app.py
```

## Methodology notes

- **Random Forest over SVM / Naïve Bayes / k-NN**: it handles the class imbalance well (built-up pixels are only ~12% of the ROI), doesn't need feature scaling across NDVI/NDBI/reflectance ranges, and gives out-of-bag error estimates for free — useful when labeled GEE samples are limited.
- **Tree count (k = 50)**: came from a convergence study — OOB error plateaus around 50 trees (8.7%) with no statistically significant accuracy gain out to 200 trees (p > 0.05, paired t-test, 5-fold CV), while inference time nearly quadruples (2.3 s → 9.1 s per ROI tile). 50 was the sweet spot.
- **GLCM entropy threshold (H = 3.5 bits)**: calibrated empirically on the Kattankulathur site, then cross-checked on a secondary semi-arid ROI without recalibration — it held up, which suggests it generalizes across bioclimatic zones rather than being overfit to one campus.
- **LST calibration**: `LST (°C) = DN × 0.00341802 + 149.0 − 273.15`, applied to Landsat-8 Collection 2 Level-2 Band 10.

Full derivations, ablations, and cross-validation results are in the paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/11570461).

## Limitations

- 10 m / 30 m sensor resolution can't resolve sub-pixel features (skylights, rooftop equipment) that matter for individual-building solar feasibility.
- Satellite LST is radiant surface temperature, not ambient air temperature — don't compare it directly to weather-station readings.
- Monsoon cloud cover biases temporal coverage toward the pre-monsoon season (March–June) for the primary ROI.
- The GLCM entropy threshold and decision thresholds (`T_crit`, `T_high`) are empirically derived and should be validated against in-situ measurements before deploying anywhere new.

## Citation

If this is useful for your own work, here's the BibTeX:

```bibtex
@inproceedings{mahara2026uhi,
  title     = {AI-Driven Satellite Data Fusion for Urban Heat Island Mitigation},
  author    = {Mahara, Ritesh and Tandon, Khyati and Venkatesh, C.},
  booktitle = {Proceedings of the 7th International Conference on Inventive Research in Computing Applications (ICIRCA)},
  pages     = {453--460},
  year      = {2026},
  publisher = {IEEE},
  url       = {https://ieeexplore.ieee.org/document/11570461}
}
```

## Stack

Streamlit · Google Earth Engine · Anthropic Claude · Google Gemini · Folium · Plotly · Pandas

## License

[MIT](LICENSE)
