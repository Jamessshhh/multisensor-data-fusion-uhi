<div align="center">

# Multisensor Data Fusion for Urban Heat Island Analysis

### From hot-spot maps to a *where-to-act* plan: satellite fusion + machine learning for urban cooling interventions

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-powered-4285F4?logo=googleearth&logoColor=white)](https://earthengine.google.com/)
[![IEEE ICIRCA 2026](https://img.shields.io/badge/IEEE-ICIRCA%202026-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/11570461)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/Jamessshhh/multisensor-data-fusion-uhi)](https://github.com/Jamessshhh/multisensor-data-fusion-uhi/commits/main)

[**Paper**](https://ieeexplore.ieee.org/document/11570461) ·
[**Quick start**](#quick-start) ·
[**How it works**](#how-it-works) ·
[**Methodology**](#methodology) ·
[**Limitations**](#limitations-and-honest-notes) ·
[**Cite**](#citation)

</div>

---

## Overview

Most Urban Heat Island (UHI) tools stop at the heat map. This project asks the next question: **once you have found the hot spots, what should you actually do about them?**

It fuses **Sentinel-2** (10 m optical), **Landsat 8/9** (30 m thermal) and **MODIS** (1 km land surface temperature) through **Google Earth Engine**, classifies land cover with a **Random Forest**, adds **GLCM texture** analysis, and combines everything at decision level to delineate three kinds of intervention zones:

| Zone | What it means | Where it comes from |
|---|---|---|
| **Cool roof** (red) | Impervious rooftops in the hottest 20% of the study area | Built-up gate ∩ Landsat LST ≥ P80 |
| **Solar PV** (orange) | Impervious rooftops in the hottest 40% (excluding cool-roof zones) | Built-up gate ∩ Landsat LST ≥ P60 |
| **Afforestation** (purple) | Rough, sparsely vegetated, heat-stressed urban land | RF urban mask ∩ GLCM roughness ∩ NDVI/NDBI limits ∩ LST ≥ P60 |

A Streamlit dashboard then turns those zones into square-metre area totals, impact estimates (cooling load, PV capacity, CO₂), a prioritised list of literature-backed recommendations, and downloadable planner deliverables (interactive map, PNG snapshot, CSV report).

> [!NOTE]
> **Origin.** This was my B.Tech final-year major project at SRM Institute of Science and Technology. The method was published as:
>
> R. Mahara, K. Tandon, and C. Venkatesh, **"AI-Driven Satellite Data Fusion for Urban Heat Island Mitigation,"** *Proc. 7th Int. Conf. on Inventive Research in Computing Applications (ICIRCA 2026)*, IEEE, pp. 453–460, 2026. [IEEE Xplore](https://ieeexplore.ieee.org/document/11570461)

---

## Contents

- [Results at a glance](#results-at-a-glance)
- [Sample outputs](#sample-outputs)
- [System architecture](#system-architecture)
- [Features](#features)
- [How it works](#how-it-works)
- [Methodology](#methodology)
- [Paper vs. this repository](#paper-vs-this-repository)
- [Quick start](#quick-start)
- [Using the dashboard](#using-the-dashboard)
- [Project structure](#project-structure)
- [Limitations and honest notes](#limitations-and-honest-notes)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Author](#author)
- [Data sources and acknowledgements](#data-sources-and-acknowledgements)
- [License](#license)

---

## Results at a glance

Headline numbers **as reported in the ICIRCA 2026 paper** for the primary study area (SRM IST campus, Kattankulathur, Tamil Nadu, India: 12.82°N, 80.04°E).

| Metric | Value |
|---|---|
| Random Forest overall accuracy | **91.3%** (Kappa = 0.88) |
| vs. SVM (RBF) baseline | 87.2% OA |
| vs. Naïve Bayes baseline | 83.6% OA |
| Misclassification reduction from sensor fusion | **18.4%** vs. best single-sensor baseline |
| Rooftop false-positive reduction from GLCM texture | **23.1%** |

| Intervention zone | Area | Estimated impact |
|---|---|---|
| Cool roofs | ~45,000 m² | LST reduction of 3–5 °C |
| Solar PV potential | ~72,000 m² | ~10,800 kW capacity |
| Afforestation | ~38,000 m² | ~83,600 kg CO₂ offset per year |

> [!IMPORTANT]
> These figures come from the paper's experimental configuration. The code in this repository is a later, extended implementation (v5) with a different classifier setup and different impact-estimation constants, so running the app will **not** reproduce these exact numbers. See [Paper vs. this repository](#paper-vs-this-repository) for the differences.

---

## Sample outputs

Maps of the Kattankulathur study area (≈3 km radius around 12.823°N, 80.044°E), styled with `make_academic_maps.py`.

<table>
<tr>
<td align="center" width="33%"><img src="assets/ndvi.png" alt="NDVI map of the Kattankulathur study area" width="100%"><br><b>NDVI</b><br><sub>Vegetation vigour. Low values (&lt; 0.2) mark built-up or bare surfaces.</sub></td>
<td align="center" width="33%"><img src="assets/ndbi.png" alt="NDBI map of the Kattankulathur study area" width="100%"><br><b>NDBI</b><br><sub>Built-up intensity. Positive values (&gt; 0.1) mark concrete and asphalt.</sub></td>
<td align="center" width="33%"><img src="assets/lst.png" alt="Land surface temperature map of the Kattankulathur study area" width="100%"><br><b>Land surface temperature</b><br><sub>Built-up zones reach about 44 °C, roughly 10 °C above adjacent vegetation and water.</sub></td>
</tr>
</table>

---

## System architecture

The pipeline from the paper: data acquisition, pre-processing, classification, decision-level fusion, and results delivery.

![System architecture: data sources through pre-processing, classification, fusion engine, and results](assets/architecture.png)

The diagram shows the published 3-class configuration. The flowchart in [How it works](#how-it-works) shows the current implementation in this repository.

---

## Features

**Remote sensing and physics**
- Three-sensor fusion: Sentinel-2 SR Harmonized (10 m), Landsat 8 + 9 Collection 2 Level-2 (30 m), MODIS MOD11A2 v061 (1 km)
- Pixel-level cloud and quality masking for all three sensors (Sentinel-2 SCL, Landsat QA_PIXEL bit decode, MODIS QC_Day)
- Land surface temperature from the USGS Collection 2 `ST_B10` scale and offset
- NDVI-threshold emissivity model (Sobrino et al., 2004) as a display layer
- Hemisphere-aware seasonal compositing (hottest 6-month window of the previous year)

**Machine learning**
- 5-class Random Forest (Water, Vegetation, Bare soil, Built-up, Agriculture) on a 20-feature space (6 bands + 14 spectral indices), trained on ESA WorldCover labels
- Spectral supplement for sparse WorldCover built-up labels in peri-urban areas
- Two-of-four **water voting** (MNDWI, NDWI, AWEInsh, RF class) to avoid flooding the map with moist-vegetation false positives
- GLCM entropy on NIR to separate smooth surfaces from rough ones

**Decision support**
- Mutually exclusive intervention zones with square-metre area totals
- Rule-based recommendation engine (7 intervention types) with evidence notes, cost band, difficulty and time frame
- UHI intensity, LST distribution (P5 / P50 / P95 / σ), heat-stress category, automatic sanity-check warnings
- Planner deliverables: interactive HTML action map, 1600 px PNG snapshot, CSV report

**Engineering**
- Everything heavy runs server-side in Earth Engine; the dashboard only fetches summaries and tiles
- Three-tier authentication fallback (service account, cached OAuth, interactive OAuth)
- Auto-capping of implausible zone areas (35% of ROI general cap, 15% for afforestation) with visible warnings

---

## How it works

```mermaid
flowchart TB
    ROI["Place name → Nominatim geocoding → 5 km radius ROI"]

    subgraph ACQ["1 · Data acquisition (Google Earth Engine)"]
        S2["Sentinel-2 SR Harmonized<br/>10 m · B2 B3 B4 B8 B11 B12"]
        LS["Landsat 8 + 9 C2 L2<br/>30 m · ST_B10"]
        MD["MODIS MOD11A2 v061<br/>1 km · LST_Day_1km"]
        WC["ESA WorldCover v200<br/>training labels"]
    end

    ROI --> ACQ

    S2 --> FE["2 · Feature engineering<br/>6 bands + 14 spectral indices"]
    S2 --> GL["GLCM texture<br/>NIR entropy → roughness"]
    S2 --> EM["NDVI-based emissivity"]
    WC --> RF
    FE --> RF["3 · Random Forest<br/>5 classes · 120 trees"]
    RF --> LC["Land-cover map + urban mask"]

    LS --> PR["Percentile ranks within ROI<br/>P60 hot · P80 critical"]
    MD --> ST["P5 / P50 / P95 / σ<br/>UHI intensity · WBGT proxy"]

    LC --> FU
    GL --> FU
    PR --> FU["4 · Decision-level fusion<br/>rooftop gate + water vote<br/>+ thermal rank + texture"]
    FU --> Z["Zones: Cool roof · Solar PV · Afforestation"]

    Z --> REC["5 · Rule-based recommendation engine"]
    ST --> REC
    REC --> OUT["Streamlit dashboard · interactive maps · CSV · PNG"]
```

**Step by step**

1. **Acquisition.** Geocode the place name, buffer to a 5 km radius ROI (~78.5 km²), and build seasonal composites from three sensors (previous year, April–October in the northern hemisphere and October–April in the southern).
2. **Feature engineering.** Compute 14 spectral indices from Sentinel-2 and stack them with the six raw bands.
3. **Classification.** Train a 120-tree Random Forest on stratified samples labelled from ESA WorldCover, supplemented with spectrally identified built-up pixels where WorldCover under-labels urban areas. Smooth with a majority filter.
4. **Fusion.** Gate rooftops with NDBI + IBI, remove water by majority vote, rank pixels by Landsat LST within the ROI, apply GLCM roughness for afforestation sites, and resolve overlaps with a fixed priority (cool roof > solar > afforestation).
5. **Delivery.** Compute zonal areas and MODIS statistics, run sanity checks, generate recommendations, and render the analyst map, the planner map, charts and downloads.

---

## Methodology

<details>
<summary><b>Data acquisition and cloud masking</b></summary>

<br>

| Sensor | Collection | Resolution | Used for |
|---|---|---|---|
| Sentinel-2 | `COPERNICUS/S2_SR_HARMONIZED` | 10 m | All spectral indices, classification, GLCM texture |
| Landsat 8 | `LANDSAT/LC08/C02/T1_L2` | 30 m | Spatial delineation of hot / critical zones |
| Landsat 9 | `LANDSAT/LC09/C02/T1_L2` | 30 m | Merged with Landsat 8 for denser temporal coverage |
| MODIS Terra | `MODIS/061/MOD11A2` | 1 km | Absolute LST statistics, UHI intensity, WBGT proxy |
| ESA WorldCover | `ESA/WorldCover/v200` | 10 m | Training labels for the Random Forest |

- **Sentinel-2:** scenes filtered by `CLOUDY_PIXEL_PERCENTAGE` (user-adjustable, default 25%), then masked with the Scene Classification Layer (kept classes: vegetation, bare soil, water, unclassified, snow/ice). A median composite is built from bands B2, B3, B4, B8, B11, B12; the SWIR bands (B11, B12) use the 10th-percentile composite, which improves water detection.
- **Landsat:** scenes with `CLOUD_COVER < 30`, masked by `QA_PIXEL` bits for dilated cloud, cirrus, cloud, shadow and snow. A median LST composite is taken over the window.
- **MODIS:** only pixels with `QC_Day` quality flag "good" are kept; the 95th percentile across the window is used as the peak-heat metric.
- **LST conversion (Landsat C2 L2):** `LST (°C) = DN × 0.00341802 + 149.0 − 273.15`.

</details>

<details>
<summary><b>Feature space (20 features)</b></summary>

<br>

Six raw reflectance bands (B2, B3, B4, B8, B11, B12) plus 14 indices:

| Index | Formula | Purpose |
|---|---|---|
| NDVI | (NIR − Red) / (NIR + Red) | Vegetation vs. non-vegetation |
| NDBI | (SWIR1 − NIR) / (SWIR1 + NIR) | Built-up surfaces, the key UHI signal |
| MNDWI | (Green − SWIR1) / (Green + SWIR1) | Open water |
| NDWI | (Green − NIR) / (Green + NIR) | Water and moisture |
| BSI | (SWIR1 + Red − NIR − Blue) / (SWIR1 + Red + NIR + Blue) | Bare soil vs. urban |
| EVI | 2.5 (NIR − Red) / (NIR + 6 Red − 7.5 Blue + 1) | Dense vegetation |
| SAVI | 1.5 (NIR − Red) / (NIR + Red + 0.5) | Sparse vegetation and agriculture |
| NBI | Red × SWIR1 / NIR | Built-up and bare soil |
| UI | (SWIR2 − NIR) / (SWIR2 + NIR) | Impervious surfaces |
| BU | NDBI − NDVI | Built-up vs. vegetation |
| RI | (Red − Green) / (Red + Green) | Red laterite soil vs. vegetation |
| IBI | (NDBI − (SAVI + MNDWI)/2) / (NDBI + (SAVI + MNDWI)/2) | Impervious built-up, reduces soil confusion |
| VgNIRBI | (Green − NIR) / (Green + NIR) | Urban-vegetation boundary |
| NDSI | (Green − SWIR1) / (Green + SWIR1) | Soil moisture proxy |

</details>

<details>
<summary><b>Random Forest classifier</b></summary>

<br>

| Setting | Value |
|---|---|
| Implementation | Earth Engine `smileRandomForest` |
| Trees | 120 |
| Variables per split | 5 |
| Min leaf population | 8 |
| Bag fraction | 0.7 |
| Seed | 2025 |
| Training samples | Stratified, 400 per class and 800 for built-up (up to 2,400 in total), sampled at 10 m |
| Post-processing | Circular majority filter (radius 2 px); built-up additionally requires NDBI > −0.2 |

**Label mapping from ESA WorldCover:**

| WorldCover classes | Target class |
|---|---|
| Tree cover, shrubland, grassland, wetland, mangroves, moss | Vegetation |
| Cropland | Agriculture |
| Built-up | Built-up |
| Bare / sparse vegetation, snow and ice | Bare soil |
| Permanent water | Water |

**Spectral urban supplement.** WorldCover under-labels built-up land in many peri-urban South Asian landscapes. Wherever NDBI > 0.12 **and** IBI > 0.06, the training label is forced to Built-up, and the built-up class is sampled at 2× weight to counter class imbalance.

</details>

<details>
<summary><b>Water detection, rooftop gate and zone logic</b></summary>

<br>

**Water (2-of-4 majority vote).** A pixel is water only if at least two of these agree: `MNDWI > 0.1`, `NDWI > 0.1`, `AWEInsh > 0`, RF class = Water. A single-index "OR" rule flagged moist paddy fields as water in testing, so independent confirmation is required.

**Rooftop / impervious gate.**
`(NDBI > 0.20 AND IBI > 0.10)` **OR** `(RF = Built-up AND NDBI > 0.15 AND IBI > 0.05)`, excluding water. The IBI condition keeps bare red laterite soil, which has positive NDBI, out of the rooftop class.

**Thermal ranking.** Landsat LST is thresholded by *percentile rank within the ROI* (hot = top 40%, critical = top 20%), not by absolute °C. Aerosol and atmospheric bias shift all pixels by roughly the same amount, so relative ranking is robust to it while the 30 m spatial detail is preserved. Absolute temperature statistics come from MODIS.

**Zones (mutually exclusive, priority: cool roof > solar > afforestation):**

| Zone | Rule |
|---|---|
| Cool roof | Rooftop AND Landsat LST ≥ P80 |
| Solar PV | Rooftop AND Landsat LST ≥ P60 AND not cool roof |
| Afforestation | RF urban mask AND GLCM roughness ≥ threshold AND NDBI < 0.15 AND 0.05 < NDVI < 0.40 AND LST ≥ P60 AND not water AND not cool roof / solar |

**GLCM texture.** NIR (B8) is scaled to 16-bit integers, a grey-level co-occurrence matrix is computed with a 7×7 window, and the entropy is rescaled ×2.5 into a "roughness" layer. The afforestation rule requires roughness ≥ the sidebar threshold (default 3.5, matching the paper's 3.5-bit entropy threshold).

**Sanity checks.** Any zone larger than 35% of the ROI, or afforestation larger than 15%, is capped for derived metrics and flagged in the dashboard. A zone total above 80% of the ROI raises an overlap warning.

</details>

<details>
<summary><b>Thermal metrics, impact estimates and recommendations</b></summary>

<br>

**UHI intensity** is the mean MODIS LST of the warm half of the ROI minus the mean of the cool half (split at the median), classified as Negligible (< 1 °C), Weak, Moderate, Strong, or Extreme (> 6 °C).

**Heat stress** uses a simple proxy, `WBGT ≈ 0.72 × P95 LST`, binned into Low / Moderate / High / Very High / Extreme.

**Impact assumptions (planning-level, not simulations):**

| Quantity | Assumption |
|---|---|
| Solar PV capacity | Zone area × 0.18 panel efficiency × 0.70 packing factor = 0.126 kW/m² |
| Annual PV energy | ~1,600 sun-hours per year; grid factor 0.82 kg CO₂/kWh (India) |
| Cool roof cooling load | ~10 W/m² annual average (≈ 40 W/m² peak, Akbari et al. 2009) |
| Afforestation CO₂ | 1 kg CO₂/m²/yr over 60% canopy cover of the plantable area |
| Tree count | One tree per 25 m² of planting zone |

**Recommendation rules** (each carries an evidence note, cost band, difficulty rating and time frame, and is ranked by priority then area):

| Recommendation | Trigger |
|---|---|
| Emergency cool-roof programme | Critical heat and cool-roof area > 1,000 m² |
| Rooftop solar PV | Solar zone > 500 m² |
| Afforestation and green corridors | Afforestation zone > 1,000 m² |
| Permeable pavement / water-sensitive design | UHI intensity > 2 °C and bare surface > 5,000 m² |
| Green-roof policy | Roof area > 2,000 m² and low vegetation-to-built ratio |
| Heat early-warning and public-health response | WBGT category Very High or Extreme |
| Agricultural heat adaptation | Agricultural area > 5,000 m² and P95 LST > 38 °C |

The engine is **deterministic and rule-based**. No language model or external API is involved.

</details>

---

## Paper vs. this repository

The paper describes the method as published. The code here is a later iteration (v5) that adapts it to messier real-world conditions (peri-urban India, aerosol-biased Landsat temperatures, sparse WorldCover built-up labels).

| Aspect | Paper (ICIRCA 2026) | This repository (v5) |
|---|---|---|
| Land-cover classes | 3: Water, Vegetation, Built-up | 5: adds Bare soil and Agriculture |
| Random Forest | 50 trees (OOB-error convergence study) | 120 trees |
| Feature space | 8-D: 6 bands + NDVI + NDBI | 20-D: 6 bands + 14 indices |
| Training labels | 1,000 stratified WorldCover samples | Up to 2,400 stratified samples, WorldCover plus spectral urban supplement |
| Thermal data | Landsat 8 `ST_B10` | MODIS P95 for absolute statistics, Landsat 8 + 9 for 30 m zoning |
| Heat thresholds | Adjustable `T_crit`, `T_high` in °C | Percentile-rank zoning (P60 / P80); °C thresholds drive gauges and recommendations |
| Impact constants | Paper-specific estimates | 0.126 kW/m² PV, 1 kg CO₂/m²/yr × 60% canopy (see above) |
| In-app accuracy assessment | Reported in the paper | Not computed in the app yet (see [Roadmap](#roadmap)) |

---

## Quick start

### Prerequisites

- Python **3.10+**
- A Google account with [Earth Engine access](https://earthengine.google.com/signup/) and a Google Cloud project registered for Earth Engine

### 1. Clone and install

```bash
git clone https://github.com/Jamessshhh/multisensor-data-fusion-uhi.git
cd multisensor-data-fusion-uhi

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Authenticate with Earth Engine

Pick **one** of the two routes.

**Option A: service account (recommended; also what you need for deployment)**

1. In your Google Cloud project, enable the Earth Engine API and create a service account with Earth Engine access. See Google's [service-account guide](https://developers.google.com/earth-engine/guides/service_account).
2. Download the JSON key and add it to `.streamlit/secrets.toml`:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

```toml
GEE_SERVICE_ACCOUNT_KEY = '''
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "your-sa@your-project-id.iam.gserviceaccount.com"
}
'''
```

The project ID is read from the key automatically. `auth.py` also accepts a path in the `GEE_SERVICE_ACCOUNT_KEY` environment variable, or a key file at `credentials/service_account.json`.

**Option B: your own Google login (quick local testing)**

```bash
earthengine authenticate
```

Then set `_DEFAULT_PROJECT` at the top of `auth.py` to **your** Earth Engine-enabled Cloud project ID.

> [!NOTE]
> No Anthropic, Gemini or other LLM keys are required. The recommendation engine is rule-based.

### 3. Run

```bash
streamlit run app.py
```

> [!WARNING]
> Never commit `.streamlit/secrets.toml` or any service-account key. Both are already covered by `.gitignore`.

### Deploying on Streamlit Community Cloud

Push the repo, create a new app pointing at `app.py`, and paste the same `GEE_SERVICE_ACCOUNT_KEY` value into the app's **Secrets** panel.

---

## Using the dashboard

1. Type a city, district or campus name into the search bar (the default is *Chennai*).
2. Tune the sidebar if needed (defaults in the table below), then press **Analyse**.
3. Explore the results top to bottom:
   - **Geospatial view:** analyst map with toggleable layers (Sentinel-2 true colour, MODIS and Landsat LST, emissivity, NDBI, 5-class land cover, intervention zones, water) and four basemaps
   - **Thermal analytics:** peak-LST gauge, LST distribution, UHI intensity gauge, land-cover donut, zone-area bars
   - **Recommendations:** ranked cards with impact, cost, difficulty, time frame and an expandable evidence note
   - **Intervention action map:** a clean planner map showing only the three action zones
   - **Downloads:** interactive HTML map, PNG snapshot, full CSV report

| Sidebar parameter | Default | Range | Effect |
|---|---|---|---|
| Max cloud cover | 25% | 5–60% | Sentinel-2 scene filter |
| GLCM entropy threshold | 3.5 | 1.0–8.0 | Smooth vs. rough split for afforestation sites |
| Actionable heat | 38 °C | any | Gauge colouring and recommendation logic |
| Critical heat | 42 °C | must exceed actionable | Gauge colouring and cool-roof trigger |

---

## Project structure

```text
.
├── app.py                   # Streamlit dashboard: pipeline controller, recommendation engine, charts, exports
├── auth.py                  # Earth Engine authentication (service account → cached OAuth → interactive)
├── data_processing.py       # Acquisition, cloud masking, LST, emissivity, GLCM texture, UHI intensity
├── features.py              # Spectral-index feature engineering helpers
├── ml_model.py              # 5-class Random Forest, training-label construction, urban mask
├── make_academic_maps.py    # Re-styles raw map exports into publication-style figures
├── export_maps.py           # Early standalone exporter for NDVI / NDBI / LST PNGs
├── assets/                  # Images used by this README (architecture diagram, sample maps)
├── ndvi.png, ndbi.png, lst.png                      # Raw map exports
├── ndvi_academic.png, ndbi_academic.png, lst_academic.png   # Publication-style versions
├── requirements.txt
└── .streamlit/
    └── secrets.toml.example
```

**Regenerating the figures.** `make_academic_maps.py` additionally needs `matplotlib` and `numpy` (`pip install matplotlib numpy`).

---

## Limitations and honest notes

Scientific tools are most useful when their boundaries are clear:

- **Surface, not air, temperature.** Satellite LST is radiant skin temperature at the overpass time (daytime, mid-morning). It is not comparable to weather-station air temperature, and it says nothing about the often stronger *night-time* UHI.
- **Zones are relative by design.** Hot and critical zones are the top 40% and top 20% of Landsat LST *within the ROI*. That makes them robust to atmospheric bias, but it also means every ROI will have "hot" zones, even a cool one. Read them together with the absolute MODIS statistics.
- **Coarse UHI metric.** MODIS is 1 km, so a 5 km-radius ROI contains only about 80 pixels. UHI intensity is a coarse, ROI-level indicator, not a street-level one.
- **Resolution.** At 10 m and 30 m, individual roof features (skylights, equipment, shading) cannot be resolved. Solar potential here is a screening estimate, not a feasibility study.
- **Weak supervision.** Training labels come from WorldCover plus a spectral rule that uses indices which are also model features. This is pragmatic where labelled data is scarce, but it is not a substitute for independent ground truth.
- **No in-app accuracy report yet.** The accuracy figures above are from the paper; the dashboard does not currently compute a confusion matrix.
- **Planning-level impact numbers.** Cooling, PV and CO₂ estimates use literature-based constants. The WBGT value is a simple proxy (`0.72 × P95 LST`), not a measured wet-bulb globe temperature.
- **Heuristic thresholds.** The NDBI / IBI gates, GLCM threshold and sanity caps were tuned on South Indian sites. Validate them with in-situ measurements before using the tool elsewhere.
- **Seasonal bias.** Monsoon cloud cover limits usable imagery to the drier, hotter months in tropical ROIs.
- **No automated tests.** Validation so far has been by inspection and sanity-check caps.

---

## Roadmap

- [ ] In-app accuracy assessment (hold-out split, confusion matrix, Kappa)
- [ ] Validation against in-situ or air-temperature measurements
- [ ] Multi-year and seasonal comparison, plus night-time thermal analysis
- [ ] Refresh `export_maps.py` for the current four-value `get_satellite_data()` return
- [ ] Split `app.py` into UI, recommendation and pipeline modules; add unit tests and CI
- [ ] Dockerfile and a hosted demo

---

## Citation

If this project helps your work, please cite the paper:

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

---

## Author

**Ritesh Mahara**
B.Tech, Computer Science and Engineering, SRM Institute of Science and Technology
Research interests: geospatial intelligence and machine learning for remote sensing

GitHub: [@Jamessshhh](https://github.com/Jamessshhh)

Paper co-authors: Khyati Tandon and C. Venkatesh.

---

## Data sources and acknowledgements

- **Sentinel-2** imagery: contains modified Copernicus Sentinel data (ESA / European Union)
- **Landsat 8/9** Collection 2: U.S. Geological Survey / NASA
- **MODIS MOD11A2**: NASA LP DAAC
- **ESA WorldCover v200**: © ESA WorldCover project, CC BY 4.0
- **Geocoding:** OpenStreetMap Nominatim · **Basemaps:** OpenStreetMap, CARTO, Esri World Imagery
- Built with [Google Earth Engine](https://earthengine.google.com/), [Streamlit](https://streamlit.io/), [Folium](https://python-visualization.github.io/folium/), [Plotly](https://plotly.com/python/) and [pandas](https://pandas.pydata.org/)

---

## License

Released under the [MIT License](LICENSE).

<div align="center">
<sub>If you find this useful, a ⭐ on the repo is much appreciated.</sub>
</div>
