"""
Urban Heat Island AI - Controller & Dashboard (v3)
MVC architecture: Controller + View layer.
Model logic: data_processing.py | ml_model.py | auth.py

Key improvements over v2:
  - Scientific Recommendations Engine: evidence-based, prioritized interventions
    with ROI estimates, CO₂ budgets, and implementation difficulty ratings
  - 5-class land cover breakdown chart (Water/Veg/Soil/Urban/Agriculture)
  - UHI Intensity metric (urban − rural ΔT) displayed prominently
  - LST statistical distribution (P5/P50/P95 + StdDev)
  - Emissivity-corrected LST from Split-Window Algorithm
  - Uncertainty / confidence scoring displayed per metric
  - WBGT (Wet-Bulb Globe Temperature) heat stress index estimate
  - Class accuracy info surfaced in debug panel
  - Streamlined, dark-science UI with monospace data readouts
  - Export includes all new metrics + recommendation summary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import ee
import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

from auth import initialize_earth_engine
from data_processing import get_satellite_data, calculate_physics, compute_uhi_intensity
from ml_model import run_random_forest

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("UHI_App_v3")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Urban Heat Island AI · Research Platform",
    page_icon="🌡️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

#MainMenu, footer { visibility: hidden; }

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Monospace data readouts */
code, .mono {
    font-family: 'Space Mono', monospace;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(15,20,30,0.95) 0%, rgba(20,30,45,0.95) 100%);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 8px;
    padding: 16px 20px;
    box-shadow: 0 2px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
}

/* Status pills */
.pill-critical { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4);
    color: #f87171; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.pill-warning  { background: rgba(251,146,60,0.15); border: 1px solid rgba(251,146,60,0.4);
    color: #fb923c; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.pill-ok       { background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.4);
    color: #4ade80; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }

/* Recommendation cards */
.rec-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.98) 0%, rgba(15,30,55,0.98) 100%);
    border-left: 3px solid;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.rec-priority-1 { border-color: #ef4444; }
.rec-priority-2 { border-color: #f97316; }
.rec-priority-3 { border-color: #eab308; }
.rec-priority-4 { border-color: #22c55e; }

/* Info bar */
.info-bar {
    background: rgba(14,165,233,0.08);
    border: 1px solid rgba(14,165,233,0.2);
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 0.85rem;
    margin-bottom: 12px;
}

/* Section headings */
h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; letter-spacing: -0.5px; }

/* Sidebar tweaks */
section[data-testid="stSidebar"] {
    background: rgba(8,12,24,0.98);
    border-right: 1px solid rgba(56,189,248,0.1);
}
</style>
""", unsafe_allow_html=True)


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class PipelineResult:
    """Immutable result bundle produced by one full pipeline run."""
    city:         str
    lat:          float
    lon:          float

    # Thermal
    peak_lst:     float     # P95 LST °C
    lst_p5:       float     # P5 °C
    lst_p50:      float     # Median °C
    lst_std:      float     # StdDev °C
    uhi_intensity: float    # Urban − Rural mean ΔT °C
    urban_mean_lst: float   # Mean LST in urban pixels °C
    rural_mean_lst: float   # Mean LST in rural/veg pixels °C

    # Zones (sq m)
    solar_area:   float
    roof_area:    float
    trees_area:   float
    water_area:   float
    veg_area:     float
    bare_area:    float
    agri_area:    float

    # Rendering
    map_html:     str

    # Fields with defaults must all come after fields without defaults
    urban_area:        float      = 0.0   # spectral urban (NDBI+IBI) for donut
    errors:            list[str]  = field(default_factory=list)
    roi_area_m2:       float      = 0.0
    modis_lst_p95:     float      = 0.0
    zones_validated:   bool       = True
    sanity_warnings:   list[str]  = field(default_factory=list)

    # Intervention-only deliverables (focused map + downloadable artifacts)
    # These are separate from the analyst map (`map_html`) so that planners
    # who only need the action plan can grab a clean, single-purpose visual
    # without having to toggle off six analytical overlays.
    intervention_map_html:  str    = ""    # self-contained Folium HTML string
    intervention_map_png:   bytes  = b""   # PNG snapshot bytes (Sentinel-2 + zones)

    # ── Derived metrics ────────────────────────────────────────────────────────
    @property
    def co2_offset_kg(self) -> float:
        """
        Afforestation CO₂ sequestration.
        Realistic rate for tropical plantations: 8–12 t CO₂/ha/yr (IPCC AR6).
        Using 10 t CO₂/ha/yr = 1 kg/m²/yr (conservative middle estimate).
        Assumes 60% canopy coverage of plantable area.
        """
        plantable = self.trees_area * 0.60   # 60% canopy coverage
        return plantable * 1.0               # 1 kg CO₂/m²/yr

    @property
    def solar_kw(self) -> float:
        """
        Realistic solar PV capacity.
        - Panel efficiency: 18% (commercial monocrystalline, 2024 standard)
        - Packing factor: 0.70 (panels don't cover 100% of roof — walkways, HVAC)
        - 1 kWp per 5.56 m² at 18% efficiency
        Net: solar_area × 0.18 × 0.70 = solar_area × 0.126
        """
        return self.solar_area * 0.126

    @property
    def cool_roof_cooling_kw(self) -> float:
        """
        Cooling load reduction from high-SRI cool roof coatings.
        Based on: Akbari et al. (2009) — ~40 W/m² peak reduction in tropical climates.
        Annual average ~25% of peak = 10 W/m² = 0.01 kW/m².
        """
        return self.roof_area * 0.01   # kW annual average

    @property
    def wbgt_estimate(self) -> float:
        """
        WBGT (Wet-Bulb Globe Temperature) rough estimate from LST.
        WBGT ≈ 0.7 * Tw + 0.2 * Tg + 0.1 * Ta
        Simplified: WBGT ≈ peak_lst * 0.72 (outdoor sunny conditions).
        """
        return self.peak_lst * 0.72

    @property
    def heat_stress_category(self) -> str:
        w = self.wbgt_estimate
        if w < 28: return "Low"
        if w < 32: return "Moderate"
        if w < 35: return "High"
        if w < 40: return "Very High"
        return "Extreme"

    def is_critical(self, t_crit: float) -> bool:
        # MODIS LST is ~36°C for Tamil Nadu peri-urban (accurate).
        # Use P95 > P50 + 2°C as "critical" if MODIS < t_crit (avoids dead zone).
        # Also trigger if MODIS P95 > 35°C (tropical UHI always warrants action).
        # Guard against missing MODIS data: only use the relative threshold when
        # P50 is plausibly populated (> 10°C) — otherwise fall back to t_crit.
        if self.lst_p50 > 10.0:
            relative_thresh = min(t_crit, self.lst_p50 + 2.0)
        else:
            relative_thresh = t_crit
        return self.peak_lst >= relative_thresh or self.peak_lst >= 35.0

    @property
    def uhi_class(self) -> str:
        i = self.uhi_intensity
        if i < 1:  return "Negligible (<1°C)"
        if i < 2:  return "Weak (1–2°C)"
        if i < 4:  return "Moderate (2–4°C)"
        if i < 6:  return "Strong (4–6°C)"
        return "Extreme (>6°C)"


# ── Scientific Recommendations Engine ─────────────────────────────────────────
@dataclass
class Recommendation:
    priority:    int       # 1 = most urgent
    title:       str
    description: str
    evidence:    str       # literature reference / mechanism
    area_m2:     float
    impact:      str       # quantified expected impact
    cost_band:   str       # Low / Medium / High / Very High
    difficulty:  str       # Easy / Moderate / Difficult / Expert
    timeframe:   str       # Short (0–2yr) / Medium (2–5yr) / Long (5–15yr)
    co2_kg:      float = 0.0
    energy_kw:   float = 0.0


def generate_recommendations(result: PipelineResult, t_crit: float) -> list[Recommendation]:
    """
    Evidence-based recommendation engine.  Priority is determined by:
      1. LST severity × affected area
      2. Cost-effectiveness (energy/CO₂ benefit per unit area)
      3. Implementation lead time (faster wins first)

    Literature references embedded in each recommendation.
    """
    recs = []
    uhi = result.uhi_intensity

    # ── 1. Emergency / Critical Heat Mitigation ───────────────────────────────
    if result.is_critical(t_crit) and result.roof_area > 1000:
        recs.append(Recommendation(
            priority=1,
            title="🔴 Emergency Cool Roof Programme",
            description=(
                f"{result.roof_area:,.0f} m² of rooftop surface exceeds the critical "
                f"threshold ({t_crit}°C). High-albedo coatings (SRI ≥ 78) can reduce "
                f"surface temperature by 15–40°C vs. conventional dark roofs."
            ),
            evidence=(
                "Synnefa et al. (2011): Cool roofs reduce indoor air temperature by "
                "1–4°C; Akbari et al. (2009): City-scale cool surface deployment can "
                "offset 150 Gt CO₂ globally."
            ),
            area_m2=result.roof_area,
            impact=f"Estimated −3 to −5°C local LST | {result.cool_roof_cooling_kw:,.0f} kW cooling load reduction",
            cost_band="Medium",
            difficulty="Easy",
            timeframe="Short (0–2 yr)",
            co2_kg=result.roof_area * 0.8,     # ~0.8 kg CO₂ offset / m² / yr via reduced AC
            energy_kw=result.cool_roof_cooling_kw
        ))

    # ── 2. Solar PV Deployment ────────────────────────────────────────────────
    if result.solar_area > 500:
        annual_kwh = result.solar_kw * 1600   # ~1600 sun-hours/yr (South India)
        co2_solar  = annual_kwh * 0.82        # India grid emission factor 0.82 kg CO₂/kWh (2023)
        recs.append(Recommendation(
            priority=2,
            title="🟠 Rooftop Solar PV Deployment",
            description=(
                f"{result.solar_area:,.0f} m² of confirmed impervious rooftop surface "
                f"(NDBI > 0.15 + IBI > 0.05) identified as viable for solar PV. "
                f"Estimated capacity: {result.solar_kw:,.0f} kW "
                f"(18% monocrystalline efficiency × 0.70 packing factor = 12.6% net). "
                f"Dual benefit: clean electricity generation + panel shading reduces "
                f"roof surface temperature by ~8–12°C."
            ),
            evidence=(
                "Masson et al. (2014): Building-scale solar deployment reduces UHI "
                "by 0.3–0.5°C at city scale. CEA India (2023): South Indian cities "
                "receive 5.0–5.8 kWh/m²/day solar irradiance."
            ),
            area_m2=result.solar_area,
            impact=f"{result.solar_kw:,.0f} kW capacity | {annual_kwh:,.0f} kWh/yr | −{co2_solar:,.0f} kg CO₂/yr",
            cost_band="High",
            difficulty="Moderate",
            timeframe="Short (0–2 yr)",
            co2_kg=co2_solar,
            energy_kw=result.solar_kw
        ))

    # ── 3. Urban Afforestation / Green Infrastructure ─────────────────────────
    if result.trees_area > 1000:
        trees_possible = int(result.trees_area / 25)  # 1 tree per 25 m² canopy
        recs.append(Recommendation(
            priority=3,
            title="🟣 Urban Afforestation & Green Corridors",
            description=(
                f"{result.trees_area:,.0f} m² of heat-stressed rough urban surface "
                f"suitable for tree planting. Potential: ~{trees_possible:,} trees. "
                f"Native species are preferred — they require no irrigation after "
                f"establishment and support local biodiversity."
            ),
            evidence=(
                "Bowler et al. (2010): Urban trees reduce ambient temperature by "
                "1–5°C within 100 m. Shashua-Bar & Hoffman (2000): Individual trees "
                "provide −0.9°C cooling over 70 m radius."
            ),
            area_m2=result.trees_area,
            impact=(
                f"~{trees_possible:,} trees | {result.co2_offset_kg:,.0f} kg CO₂ "
                f"sequestered/yr | evapotranspiration cooling"
            ),
            cost_band="Low",
            difficulty="Easy",
            timeframe="Medium (2–5 yr)",
            co2_kg=result.co2_offset_kg
        ))

    # ── 4. Permeable Pavement & Water-Sensitive Design ────────────────────────
    if uhi > 2 and result.bare_area > 5000:
        recs.append(Recommendation(
            priority=3,
            title="💧 Permeable Pavement & WSUD",
            description=(
                f"{result.bare_area:,.0f} m² of bare/impervious surfaces detected. "
                f"Replacing asphalt with permeable pavers + stormwater harvesting "
                f"increases latent heat flux, reducing surface temperature through "
                f"evaporative cooling."
            ),
            evidence=(
                "Stempihar et al. (2012): Permeable asphalt 6–10°C cooler than "
                "standard asphalt in summer. Huang et al. (2023): WSUD reduces "
                "runoff by 40–80% and peak LST by 2–4°C."
            ),
            area_m2=result.bare_area,
            impact="Estimated −2 to −4°C surface temp | improved stormwater retention",
            cost_band="High",
            difficulty="Difficult",
            timeframe="Medium (2–5 yr)"
        ))

    # ── 5. Green Roof Mandates ─────────────────────────────────────────────────
    if result.roof_area > 2000 and result.veg_area < result.solar_area * 0.5:
        recs.append(Recommendation(
            priority=4,
            title="🌿 Green Roof / Vertical Garden Policy",
            description=(
                f"Low vegetation-to-urban ratio detected. Mandating green roofs on "
                f"≥30% of {result.roof_area:,.0f} m² of critical-heat rooftops adds "
                f"evapotranspiration cooling, improves air quality, and provides "
                f"insulation benefits."
            ),
            evidence=(
                "Santamouris (2014): Green roofs reduce building energy consumption "
                "by 0.7–4.5% annually. Jim & Tsang (2011): Green roofs reduce "
                "indoor temperature by 3.3°C on average."
            ),
            area_m2=result.roof_area * 0.3,
            impact="−0.5 to −2°C local ambient air temperature | improved air quality",
            cost_band="Very High",
            difficulty="Expert",
            timeframe="Long (5–15 yr)"
        ))

    # ── 6. Heat Early Warning System ─────────────────────────────────────────
    if result.heat_stress_category in ["Very High", "Extreme"]:
        recs.append(Recommendation(
            priority=1,
            title="🚨 Heat Early Warning & Public Health Response",
            description=(
                f"WBGT estimate of {result.wbgt_estimate:.1f}°C indicates "
                f"{result.heat_stress_category} heat stress conditions. "
                f"Immediate public health protocols required: cooling centres, "
                f"elderly welfare checks, outdoor work restrictions, mist spraying "
                f"in high-density zones."
            ),
            evidence=(
                "WHO (2022): WBGT >32°C causes dangerous heat stress in unacclimatised "
                "persons. Robine et al. (2008): European 2003 heatwave caused 70,000 "
                "excess deaths — early warning systems cut mortality by 40–80%."
            ),
            area_m2=0,
            impact=f"Immediate mortality risk reduction | {result.heat_stress_category} WBGT class",
            cost_band="Low",
            difficulty="Easy",
            timeframe="Short (0–2 yr)"
        ))

    # ── 7. Agricultural Adaptation ────────────────────────────────────────────
    if result.agri_area > 5000 and result.peak_lst > 38:
        recs.append(Recommendation(
            priority=4,
            title="🌾 Agricultural Heat Adaptation",
            description=(
                f"{result.agri_area:,.0f} m² of agricultural land experiencing heat "
                f"stress (LST P95 = {result.peak_lst:.1f}°C). Recommend: shift to "
                f"heat-tolerant crop varieties, agroforestry buffers, and micro-irrigation "
                f"to reduce crop surface temperature through evapotranspiration."
            ),
            evidence=(
                "Lobell et al. (2011): Wheat yield declines 6% per °C above 30°C. "
                "Mbow et al. (2014): Agroforestry reduces crop surface temperature "
                "by 2–5°C and improves yield stability."
            ),
            area_m2=result.agri_area,
            impact=f"Crop yield protection | −2 to −5°C surface temp in agri zones",
            cost_band="Medium",
            difficulty="Moderate",
            timeframe="Medium (2–5 yr)"
        ))

    # Sort by priority
    recs.sort(key=lambda r: (r.priority, -r.area_m2))
    return recs


# ── Geocoding ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86_400, show_spinner=False)
def get_coordinates(city_name: str) -> Optional[tuple[float, float]]:
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city_name, "format": "json", "limit": 1},
            headers={"User-Agent": "UHI-AI-Research/3.0 (academic)"},
            timeout=8
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except requests.RequestException as exc:
        logger.error(f"Geocoding failed for '{city_name}': {exc}")
    return None


# ── EE layer renderer ─────────────────────────────────────────────────────────
def add_ee_layer(map_obj: folium.Map, img: ee.Image, vis: dict, name: str,
                 opacity: float = 0.75, shown: bool = True) -> None:
    """
    Adds an Earth Engine image as a tile layer.
      opacity : per-layer transparency (0–1).  Heavy thematic layers use ~0.6
                so the basemap roads/water remain partly visible underneath.
      shown   : whether the layer is enabled by default.  Most overlays are
                shown=False so the user opts-in via the LayerControl — this
                prevents water bodies from being buried under stacked overlays.
    """
    try:
        map_id = img.getMapId(vis)
        folium.TileLayer(
            tiles=map_id["tile_fetcher"].url_format,
            attr="Google Earth Engine",
            name=name, overlay=True, control=True,
            opacity=opacity, show=shown,
        ).add_to(map_obj)
    except Exception as exc:
        logger.warning(f"Could not add EE layer '{name}': {exc}")


# ── Area calculator ───────────────────────────────────────────────────────────
def zone_area_sqm(mask_img: ee.Image, roi: ee.Geometry) -> float:
    """Returns total area in sq m of non-masked pixels."""
    area_img = ee.Image.pixelArea().updateMask(mask_img)
    result = area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi, scale=10, maxPixels=1e9, bestEffort=True
    )
    val = result.getInfo()
    return float(next((v for v in val.values() if v is not None), 0) or 0.0)


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(city: str, cloud_thresh: int, t_high: float, t_crit: float,
                 texture_thresh: float) -> Optional[PipelineResult]:

    coords = get_coordinates(city)
    if not coords:
        st.error(f"❌ Could not geocode **{city}**. Try a more specific name.")
        return None

    lat, lon = coords
    ROI_RADIUS_M = 5000
    roi = ee.Geometry.Point([lon, lat]).buffer(ROI_RADIUS_M)
    # Theoretical ROI area: π × r²  (used for sanity-check %)
    import math
    ROI_AREA_M2 = math.pi * ROI_RADIUS_M ** 2   # ~78.54 km²
    errors: list[str] = []

    with st.status("🛰️ Running analysis pipeline…", expanded=True) as status:

        # Stage 1 — Data acquisition
        st.write("📡 Sentinel-2 SR + MODIS MOD11A2 + Landsat 8/9 C2 …")
        try:
            s2, lst_modis, lst_landsat, emissivity = get_satellite_data(
                roi, cloud_thresh, roi_lat=lat
            )
            # lst_modis   = MODIS P95 (1km) — accurate absolute °C for analytics
            # lst_landsat = Landsat median (30m) — spatial detail for zone delineation
            lst_celsius = lst_modis   # alias used by stats stage
        except Exception as exc:
            status.update(label="❌ Data acquisition failed.", state="error")
            st.error(f"Satellite query error: {exc}")
            return None

        # Stage 2 — Physics (GLCM texture + LST already in Celsius)
        st.write("⚙️ GLCM texture analysis (surface roughness) …")
        try:
            roughness, lst_celsius = calculate_physics(s2, lst_celsius)
        except Exception as exc:
            status.update(label="❌ Physics computation failed.", state="error")
            st.error(f"Physics error: {exc}")
            return None

        # Stage 3 — ML classifier (5-class RF)
        st.write("🧠 Training 5-class Random Forest (Urban/Veg/Water/Soil/Agri) …")
        try:
            urban_mask, class_img, accuracy_info = run_random_forest(s2, roi)
        except Exception as exc:
            status.update(label="❌ ML training failed.", state="error")
            st.error(f"Classifier error: {exc}")
            return None

        # Stage 4 — Spatial fusion
        st.write("🗺️ Fusing thermal + spectral + land-cover layers …")
        try:
            ndbi  = s2.normalizedDifference(["B11", "B8"]).rename("NDBI")
            ndvi  = s2.normalizedDifference(["B8",  "B4"]).rename("NDVI")
            mndwi = s2.normalizedDifference(["B3",  "B11"]).rename("MNDWI")
            ndwi  = s2.normalizedDifference(["B3",  "B8"]).rename("NDWI")

            b3  = s2.select("B3")
            b8  = s2.select("B8")
            b11 = s2.select("B11")
            b12 = s2.select("B12")
            b4  = s2.select("B4")
            b2  = s2.select("B2")

            # ── Index-Based Built-Up Index (IBI) ──────────────────────────────
            # IBI = (NDBI − (SAVI + MNDWI)/2) / (NDBI + (SAVI + MNDWI)/2)
            # Values > 0.1 reliably indicate impervious rooftop/pavement.
            # This is the critical gate that was missing — bare red laterite
            # soil in Tamil Nadu has NDBI > 0 but IBI < 0.05, so it was
            # previously counted as "urban rooftop" and bloating zone areas.
            savi = b8.subtract(b4).multiply(1.5).divide(b8.add(b4).add(0.5))
            ibi_num = ndbi.subtract(savi.add(mndwi).divide(2))
            ibi_den = ndbi.add(savi.add(mndwi).divide(2)).abs().add(1e-6)
            ibi = ibi_num.divide(ibi_den).rename("IBI")

            # ── AWEInsh water index ────────────────────────────────────────────
            awei_nsh = b3.multiply(4).subtract(
                b8.multiply(0.25).add(b12.multiply(2.75))
            ).rename("AWEInsh")

            # ── Water detection: 2-of-4 majority vote ─────────────────────────
            # PROBLEM WITH OR-of-4: NDWI > 0 fires on any moist vegetation
            # (paddy fields, irrigated crops), making 54% of ROI appear as water.
            # MNDWI > 0 has the same issue in tropical humid landscapes.
            #
            # FIX: require ≥ 2 independent detectors to agree.
            # Each vote scores 1; threshold at 2 means at least two independent
            # spectral signals must confirm water. This eliminates single-index
            # false positives from moist soil and vegetation.
            #
            # Conservative thresholds per index:
            #   MNDWI > 0.1   — clear/moderately turbid open water
            #   NDWI  > 0.1   — open water (slightly above noise floor)
            #   AWEInsh > 0   — water extraction index (standard threshold)
            #   RF class==0   — WorldCover-trained RandomForest water class
            #
            # Expected: water coverage drops from 54% → 5–10% (realistic)
            vote_mndwi = mndwi.gt(0.1).rename("v1")
            vote_ndwi  = ndwi.gt(0.1).rename("v2")
            vote_awei  = awei_nsh.gt(0.0).rename("v3")
            vote_rf    = class_img.eq(0).rename("v4")

            water_votes = (vote_mndwi.add(vote_ndwi)
                                     .add(vote_awei)
                                     .add(vote_rf))
            is_water  = water_votes.gte(2).rename("water_mask")
            not_water = is_water.Not()

            # ── Strict rooftop / impervious surface gate ───────────────────────
            # PRIMARY:   NDBI > 0.20 AND IBI > 0.10  (above Tamil Nadu soil range)
            # SECONDARY: RF class=3 AND NDBI > 0.15 AND IBI > 0.05
            savi = b8.subtract(b4).multiply(1.5).divide(b8.add(b4).add(0.5))
            ibi_num = ndbi.subtract(savi.add(mndwi).divide(2))
            ibi_den = ndbi.add(savi.add(mndwi).divide(2)).abs().add(1e-6)
            ibi = ibi_num.divide(ibi_den).rename("IBI")

            strict_spectral = ndbi.gt(0.20).And(ibi.gt(0.10))
            rf_buildup      = class_img.eq(3).And(ndbi.gt(0.15)).And(ibi.gt(0.05))
            is_rooftop      = strict_spectral.Or(rf_buildup).And(not_water)

            # ── Thermal masks: pull Landsat percentiles client-side first ────────
            try:
                _pct_info = lst_landsat.reduceRegion(
                    reducer=ee.Reducer.percentile([60, 80]),
                    geometry=roi, scale=30, maxPixels=1e9, bestEffort=True
                ).getInfo()
                _p60_val = float(
                    _pct_info.get("LST_p60") or
                    _pct_info.get("LST_p60_p60") or
                    next((v for v in _pct_info.values() if v and v > 30), 42.0)
                )
                _p80_val = float(
                    _pct_info.get("LST_p80") or
                    _pct_info.get("LST_p80_p80") or
                    next((v for v in sorted(_pct_info.values(), reverse=True) if v and v > 30), 46.0)
                )
            except Exception:
                _p60_val, _p80_val = 42.0, 46.0

            logger.info(f"Landsat thresholds: P60={_p60_val:.1f}°C | P80={_p80_val:.1f}°C")
            hot      = lst_landsat.gte(_p60_val)   # top 40% of scene = actionable heat
            critical = lst_landsat.gte(_p80_val)   # top 20% of scene = critical heat

            # ── Mutually exclusive zones ────────────────────────────────────────
            roof_mask  = is_rooftop.And(critical)
            solar_mask = is_rooftop.And(hot).And(roof_mask.Not())

            # ── Land cover classes for statistics & donut chart ───────────────
            is_urban_display = is_rooftop
            is_veg  = class_img.eq(1).And(ndvi.gt(0.25)).And(not_water).And(is_urban_display.Not())
            is_soil = class_img.eq(2).And(not_water).And(is_urban_display.Not())
            is_agri = class_img.eq(4).And(not_water).And(is_urban_display.Not())

            # ── Afforestation zone ─────────────────────────────────────────────
            # Uses urban_mask (RF+spectral) — more reliable than class_img.eq(3)
            # for peri-urban ROIs where WorldCover under-labels built-up areas.
            trees_mask = (
                urban_mask
                .And(roughness.gte(texture_thresh))
                .And(ndbi.lt(0.15))
                .And(ndvi.lt(0.40))
                .And(ndvi.gt(0.05))
                .And(hot)
                .And(not_water)
                .And(roof_mask.Not())
                .And(solar_mask.Not())
            )

            # Priority rendering: Cool Roof > Solar > Afforestation
            zones = (
                ee.Image(0)
                .where(trees_mask, 3)
                .where(solar_mask, 1)
                .where(roof_mask,  2)
            )
            zones_masked = zones.updateMask(zones.gt(0))

        except Exception as exc:
            status.update(label="❌ Spatial fusion failed.", state="error")
            st.error(f"Fusion error: {exc}")
            return None

        # Stage 5 — Zonal statistics + validation
        st.write("📊 Zonal statistics, MODIS cross-check & sanity validation …")
        try:
            solar_area  = zone_area_sqm(solar_mask.selfMask(),      roi)
            roof_area   = zone_area_sqm(roof_mask.selfMask(),       roi)
            trees_area  = zone_area_sqm(trees_mask.selfMask(),      roi)
            water_area  = zone_area_sqm(is_water.selfMask(),        roi)
            veg_area    = zone_area_sqm(is_veg.selfMask(),          roi)
            bare_area   = zone_area_sqm(is_soil.selfMask(),         roi)
            agri_area   = zone_area_sqm(is_agri.selfMask(),         roi)
            urban_area  = zone_area_sqm(is_urban_display.selfMask(),roi)  # for donut

            st.write(
                f"   ↳ Urban={urban_area:,.0f} | Roof={roof_area:,.0f} | "
                f"Solar={solar_area:,.0f} | Trees={trees_area:,.0f} | "
                f"Veg={veg_area:,.0f} | Water={water_area:,.0f} | Soil={bare_area:,.0f} m²"
            )

            # ── Sanity checks ──────────────────────────────────────────────────
            # Zone areas that individually exceed 40% of the ROI area are almost
            # certainly the result of spectral confusion (e.g. soil→urban).
            # We flag them clearly and cap display values so the dashboard doesn't
            # show nonsensical numbers.  The cap is conservative — 40% of a 5 km
            # radius ROI (~78.5 km²) = 31.4 km² which is already very large for
            # a single land-cover class in a mixed peri-urban landscape.
            SANITY_CAP_FRAC = 0.35   # 35% of ROI area
            SANITY_CAP_M2   = ROI_AREA_M2 * SANITY_CAP_FRAC
            sanity_warnings = []
            zones_validated = True

            def _check_and_cap(area: float, label: str) -> float:
                nonlocal zones_validated
                if area > SANITY_CAP_M2:
                    pct = area / ROI_AREA_M2 * 100
                    msg = (
                        f"⚠️ {label} area ({area/1e6:.1f} km², {pct:.0f}% of ROI) "
                        f"exceeds the 35% sanity cap — spectral confusion likely. "
                        f"Capped at {SANITY_CAP_M2/1e6:.1f} km² for derived metrics."
                    )
                    sanity_warnings.append(msg)
                    logger.warning(msg)
                    zones_validated = False
                    return SANITY_CAP_M2
                return area

            solar_area_capped  = _check_and_cap(solar_area, "Solar PV")
            roof_area_capped   = _check_and_cap(roof_area,  "Cool Roof")

            # Afforestation cap is stricter than the general 35%:
            # In a peri-urban 5 km ROI, realistically plantable urban land is
            # at most 15% of the total area (~11.8 km²).  Rough scrubland,
            # agricultural bunds, and undeveloped plots dominate the "rough urban"
            # texture class but are not all genuinely afforest-able urban sites.
            # We cap at 15% of ROI with a clear explanatory warning.
            TREES_CAP_FRAC = 0.15
            TREES_CAP_M2   = ROI_AREA_M2 * TREES_CAP_FRAC
            if trees_area > TREES_CAP_M2:
                pct = trees_area / ROI_AREA_M2 * 100
                sanity_warnings.append(
                    f"⚠️ Afforestation area ({trees_area/1e6:.1f} km², {pct:.0f}% of ROI) "
                    f"exceeds the 15% urban-plantable cap. Rough scrubland and agricultural "
                    f"bunds are included in the texture class but are not all genuinely "
                    f"urban afforestation sites. Capped at {TREES_CAP_M2/1e6:.1f} km²."
                )
                zones_validated = False
                trees_area_capped = TREES_CAP_M2
            else:
                trees_area_capped = trees_area

            # Also check that zone total doesn't exceed ROI (overlap = double-count)
            total_zone = solar_area + roof_area + trees_area
            if total_zone > ROI_AREA_M2 * 0.80:
                sanity_warnings.append(
                    f"⚠️ Zone total ({total_zone/1e6:.1f} km²) > 80% of ROI — "
                    f"zones may not be fully mutually exclusive. Review NDBI/IBI thresholds."
                )
                zones_validated = False

            # ── LST statistics from MODIS (authoritative absolute temperature) ──
            lst_stats = lst_modis.reduceRegion(
                reducer=ee.Reducer.percentile([5, 50, 95])
                                   .combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=roi, scale=1000, maxPixels=1e9, bestEffort=True
            ).getInfo()
            logger.info(f"MODIS LST stats: {lst_stats}")

            def _get(d, *keys):
                for k in keys:
                    if d.get(k) is not None:
                        return float(d[k])
                return 0.0

            peak_lst = _get(lst_stats, "LST_p95", "p95")
            lst_p5   = _get(lst_stats, "LST_p5",  "p5")
            lst_p50  = _get(lst_stats, "LST_p50", "p50")
            lst_std  = _get(lst_stats, "LST_stdDev", "stdDev")

            # ── Landsat P95 for transparency display only ──────────────────────
            modis_lst_p95 = 0.0   # field repurposed to store Landsat P95
            try:
                l_stats = lst_landsat.reduceRegion(
                    reducer=ee.Reducer.percentile([95]),
                    geometry=roi, scale=30, maxPixels=1e9, bestEffort=True
                ).getInfo()
                raw_l = next((v for v in l_stats.values() if v is not None), None)
                if raw_l:
                    modis_lst_p95 = float(raw_l)
                    logger.info(
                        f"Landsat C2 P95={modis_lst_p95:.1f}°C | MODIS P95={peak_lst:.1f}°C | "
                        f"Δ={modis_lst_p95-peak_lst:+.1f}°C (aerosol bias — zones use percentile-rank)"
                    )
            except Exception as exc_l:
                logger.warning(f"Landsat P95 stat failed: {exc_l}")

            # ── UHI Intensity (from MODIS — absolute temps) ────────────────────
            uhi_data = compute_uhi_intensity(lst_modis, roi)
            uhi_info = {k: v.getInfo() if hasattr(v, "getInfo") else v
                        for k, v in uhi_data.items()}
            uhi_intensity  = float(uhi_info.get("uhi_intensity") or 0.0)
            urban_mean_lst = float(uhi_info.get("urban_mean_lst") or peak_lst)
            rural_mean_lst = float(uhi_info.get("rural_mean_lst") or lst_p50)

            # Use capped values for derived metrics going into PipelineResult
            solar_area_final  = solar_area_capped
            roof_area_final   = roof_area_capped
            trees_area_final  = trees_area_capped

        except Exception as exc:
            status.update(label="❌ Statistics failed.", state="error")
            st.error(f"Statistics error: {exc}")
            return None

        # Stage 6 — Map
        st.write("🗺️ Rendering interactive geospatial map …")
        try:
            # ── Basemap strategy ──────────────────────────────────────────────
            # Default to CartoDB Voyager: light enough that OSM-rendered water
            # bodies (lakes, tanks, rivers) are immediately visible without any
            # overlay being toggled on.  Dark Matter remains available for
            # users who prefer the original "science console" aesthetic.
            m = folium.Map(
                location=[lat, lon], zoom_start=14,
                tiles=None, control_scale=True,
            )
            folium.TileLayer(
                tiles="CartoDB Voyager",
                name="Basemap · Voyager (light, shows water)",
                control=True, show=True,
            ).add_to(m)
            folium.TileLayer(
                tiles="OpenStreetMap",
                name="Basemap · OpenStreetMap (water bodies + roads)",
                control=True, show=False,
            ).add_to(m)
            # Esri World Imagery: real satellite background where water bodies
            # are unmistakable as dark blue/green polygons.
            folium.TileLayer(
                tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
                       "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
                attr="Esri · World Imagery",
                name="Basemap · Satellite (Esri Imagery)",
                control=True, show=False,
            ).add_to(m)
            folium.TileLayer(
                tiles="CartoDB dark_matter",
                name="Basemap · Dark Matter (science mode)",
                control=True, show=False,
            ).add_to(m)

            # ── Thematic overlays (all default OFF except zones + water) ──────
            # Why default-off: stacking 8 semi-opaque overlays buries the very
            # features the user cares about (water, roads, terrain).  The user
            # toggles in only what they need from the LayerControl on the right.
            add_ee_layer(
                m, s2,
                {"min": 0, "max": 0.3, "bands": ["B4", "B3", "B2"]},
                "Sentinel-2 True Colour",
                opacity=0.85, shown=False,
            )
            # MODIS: accurate absolute temperature (1km, analytics)
            add_ee_layer(
                m, lst_modis,
                {"min": 28, "max": 45,
                 "palette": ["#023858", "#4393c3", "#ffffbf", "#f46d43", "#a50026"]},
                "LST · MODIS P95 (1km, analytics)",
                opacity=0.65, shown=False,
            )
            # Landsat: 30m spatial detail for zone context
            add_ee_layer(
                m, lst_landsat,
                {"min": 28, "max": 55,
                 "palette": ["#023858", "#4393c3", "#ffffbf", "#f46d43", "#a50026"]},
                "LST · Landsat C2 (30m, spatial detail)",
                opacity=0.65, shown=False,
            )
            add_ee_layer(
                m, emissivity,
                {"min": 0.97, "max": 1.0,
                 "palette": ["#ffffff", "#74add1", "#313695"]},
                "Surface Emissivity (ε)",
                opacity=0.6, shown=False,
            )
            add_ee_layer(
                m, ndbi,
                {"min": -0.3, "max": 0.3,
                 "palette": ["#1a9641", "#ffffbf", "#d7191c"]},
                "NDBI (Built-Up Index)",
                opacity=0.6, shown=False,
            )
            add_ee_layer(
                m, class_img,
                {"min": 0, "max": 4,
                 "palette": ["#1976D2", "#4CAF50", "#FF9800", "#F44336", "#FFEB3B"]},
                "5-Class Land Cover (RF)",
                opacity=0.55, shown=False,
            )
            # Intervention zones — default ON because this is the headline output
            add_ee_layer(
                m, zones_masked,
                {"min": 1, "max": 3,
                 "palette": ["#FFA500", "#FF0000", "#800080"]},
                "AI Intervention Zones",
                opacity=0.78, shown=True,
            )

            # ── Water layers — added LAST so they sit on top of overlays ──────
            # We render water in two complementary ways for maximum legibility:
            #
            #   1. SOLID FILL (bright cyan) — the water mask painted as a single
            #      flat colour with full opacity.  This is the most legible
            #      representation: every detected water pixel becomes a vivid,
            #      unmistakable blob regardless of which basemap is selected.
            #
            #   2. MNDWI gradient — uses the MNDWI palette to show relative
            #      water "purity" (turbid → clear).  Useful for analysts.
            #
            # Both are added AFTER all other overlays so they always render on
            # top.  Solid fill is shown by default.
            water_solid = is_water.selfMask()
            add_ee_layer(
                m, water_solid,
                {"min": 1, "max": 1, "palette": ["#00BFFF"]},
                "💧 Water Bodies — Solid Fill (recommended)",
                opacity=1.0, shown=True,
            )

            mndwi_vis = s2.normalizedDifference(["B3", "B11"])
            add_ee_layer(
                m, mndwi_vis.updateMask(is_water),
                {"min": -0.1, "max": 0.5,
                 "palette": ["#a8d8ea", "#0077b6", "#03045e"]},
                "💧 Water Bodies — MNDWI gradient",
                opacity=0.95, shown=False,
            )

            # ROI boundary circle so the user knows what was analysed
            folium.Circle(
                location=[lat, lon], radius=ROI_RADIUS_M,
                color="#38bdf8", weight=2, fill=False, opacity=0.7,
                dash_array="6,6",
                popup=f"5 km ROI · {ROI_AREA_M2/1e6:.1f} km²",
            ).add_to(m)
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>{city}</b><br>{lat:.4f}, {lon:.4f}",
                tooltip=city,
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)

            # Legend — light card so it reads on every basemap
            legend_html = """
            <div style="position:absolute;bottom:24px;left:12px;z-index:9999;
                        background:rgba(255,255,255,0.96);
                        border:1px solid rgba(15,23,42,0.15);
                        box-shadow:0 4px 14px rgba(0,0,0,0.18);
                        border-radius:8px;padding:10px 14px;
                        font-family:'Space Mono',monospace;font-size:11px;color:#0f172a;
                        max-width:240px;line-height:1.55;">
              <b style="color:#0369a1">💧 WATER BODIES</b><br>
              <span style="color:#00BFFF;font-size:14px">■</span> Solid fill (always-on)<br>
              <span style="color:#0077b6;font-size:14px">■</span> MNDWI gradient (turbid → clear)<br>
              <hr style="border:none;border-top:1px solid #e2e8f0;margin:6px 0">
              <b style="color:#0369a1">AI INTERVENTION ZONES</b><br>
              <span style="color:#FFA500;font-size:14px">■</span> Solar PV potential<br>
              <span style="color:#FF0000;font-size:14px">■</span> Cool roof priority<br>
              <span style="color:#800080;font-size:14px">■</span> Afforestation zone<br>
              <hr style="border:none;border-top:1px solid #e2e8f0;margin:6px 0">
              <b style="color:#0369a1">5-CLASS LAND COVER</b><br>
              <span style="color:#1976D2;font-size:14px">■</span> Water
              <span style="color:#4CAF50;font-size:14px;margin-left:6px">■</span> Vegetation<br>
              <span style="color:#FF9800;font-size:14px">■</span> Bare soil
              <span style="color:#F44336;font-size:14px;margin-left:6px">■</span> Urban<br>
              <span style="color:#FFEB3B;font-size:14px;text-shadow:0 0 1px #888">■</span> Agriculture<br>
              <hr style="border:none;border-top:1px solid #e2e8f0;margin:6px 0">
              <span style="color:#475569;font-size:10px">
              Toggle layers via the control on the right →
              </span>
            </div>"""
            m.get_root().html.add_child(folium.Element(legend_html))
            folium.LayerControl(collapsed=False, position="topright").add_to(m)
            map_html = m._repr_html_()
        except Exception as exc:
            errors.append(f"Map render warning: {exc}")
            map_html = "<p style='color:orange'>Map unavailable — see logs.</p>"

        # ── Stage 6b — FOCUSED INTERVENTION MAP (planner deliverable) ─────────
        # The analyst map above carries 8+ scientific overlays; the intervention
        # map below is a single-purpose deliverable for city planners and estate
        # managers.  It shows ONLY the three action zones (Solar / Cool Roof /
        # Afforestation) over a clean OSM basemap, with a baked-in legend.
        st.write("🎯 Building focused intervention map …")
        intervention_map_html = ""
        intervention_map_png  = b""
        try:
            im = folium.Map(
                location=[lat, lon], zoom_start=14,
                tiles=None, control_scale=True,
            )
            # OSM basemap — water bodies, roads, building outlines visible by default
            folium.TileLayer(
                tiles="OpenStreetMap",
                name="Streets & Water (OSM)",
                control=True, show=True,
            ).add_to(im)
            folium.TileLayer(
                tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
                       "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
                attr="Esri · World Imagery",
                name="Satellite (Esri)",
                control=True, show=False,
            ).add_to(im)

            # Three SEPARATE zone layers (each toggleable individually) — planners
            # often want to look at solar PV alone, then cool-roof alone, etc.
            solar_layer = zones.eq(1).selfMask()
            roof_layer  = zones.eq(2).selfMask()
            trees_layer = zones.eq(3).selfMask()

            add_ee_layer(
                im, solar_layer,
                {"min": 1, "max": 1, "palette": ["#FFA500"]},
                "🟧 Solar PV Potential",
                opacity=0.85, shown=True,
            )
            add_ee_layer(
                im, roof_layer,
                {"min": 1, "max": 1, "palette": ["#FF0000"]},
                "🟥 Cool Roof Priority",
                opacity=0.85, shown=True,
            )
            add_ee_layer(
                im, trees_layer,
                {"min": 1, "max": 1, "palette": ["#800080"]},
                "🟪 Afforestation Zone",
                opacity=0.85, shown=True,
            )

            # ROI boundary + city marker
            folium.Circle(
                location=[lat, lon], radius=ROI_RADIUS_M,
                color="#0369a1", weight=2.5, fill=False, opacity=0.85,
                dash_array="6,6",
                popup=f"5 km ROI · {ROI_AREA_M2/1e6:.1f} km²",
            ).add_to(im)
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>{city}</b><br>{lat:.4f}, {lon:.4f}",
                tooltip=city,
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(im)

            # Header card with city name + actionable totals — appears as part of
            # the HTML so the standalone download is self-explanatory.
            header_html = f"""
            <div style="position:absolute;top:12px;left:60px;z-index:9999;
                        background:rgba(255,255,255,0.97);
                        border:1px solid rgba(15,23,42,0.18);
                        box-shadow:0 4px 16px rgba(0,0,0,0.15);
                        border-radius:10px;padding:12px 18px;
                        font-family:-apple-system,'Segoe UI',sans-serif;
                        max-width:520px;color:#0f172a;">
              <div style="font-weight:700;font-size:15px;color:#0369a1;
                          margin-bottom:6px;letter-spacing:0.3px;">
                🎯 INTERVENTION ACTION MAP — {city.upper()}
              </div>
              <div style="font-size:12px;line-height:1.55;">
                <span style="color:#FFA500;font-weight:700">●</span>
                Solar PV: <b>{solar_area_final:,.0f} m²</b> &nbsp;·&nbsp;
                <span style="color:#FF0000;font-weight:700">●</span>
                Cool Roof: <b>{roof_area_final:,.0f} m²</b> &nbsp;·&nbsp;
                <span style="color:#800080;font-weight:700">●</span>
                Afforestation: <b>{trees_area_final:,.0f} m²</b>
              </div>
            </div>"""
            im.get_root().html.add_child(folium.Element(header_html))

            # Bottom-left legend (mirrors the analyst map but zone-only)
            iv_legend = """
            <div style="position:absolute;bottom:24px;left:12px;z-index:9999;
                        background:rgba(255,255,255,0.97);
                        border:1px solid rgba(15,23,42,0.15);
                        box-shadow:0 4px 14px rgba(0,0,0,0.18);
                        border-radius:8px;padding:10px 14px;
                        font-family:-apple-system,'Segoe UI',sans-serif;
                        font-size:12px;color:#0f172a;line-height:1.7;">
              <b style="color:#0369a1;font-size:11px;
                         letter-spacing:0.5px">WHERE TO ACT</b><br>
              <span style="color:#FFA500;font-size:16px">■</span>
              &nbsp;Install solar PV (flat smooth roofs)<br>
              <span style="color:#FF0000;font-size:16px">■</span>
              &nbsp;Apply cool-roof coating (hot rooftops)<br>
              <span style="color:#800080;font-size:16px">■</span>
              &nbsp;Plant trees (rough heat-stressed land)<br>
              <span style="color:#0369a1;font-size:11px">┄┄</span>
              &nbsp;5 km Region of Interest
            </div>"""
            im.get_root().html.add_child(folium.Element(iv_legend))
            folium.LayerControl(collapsed=False, position="topright").add_to(im)
            intervention_map_html = im.get_root().render()

            # ── PNG snapshot via EE thumbnail (planner-ready static image) ────
            # We blend the zones over a Sentinel-2 RGB basemap inside Earth Engine
            # itself, then ask EE to render the composite as a 1600 px PNG.  This
            # avoids needing Selenium / headless-Chrome to screenshot Folium.
            try:
                # Sentinel-2 RGB composite (8-bit, gamma-corrected)
                s2_rgb = (s2.select(["B4", "B3", "B2"])
                            .visualize(min=0.02, max=0.30, gamma=1.2))
                # Three zone overlays in their own colours, each painted only where
                # the corresponding zone is true.
                solar_vis = solar_layer.visualize(palette=["#FFA500"], min=1, max=1)
                roof_vis  = roof_layer.visualize (palette=["#FF0000"], min=1, max=1)
                trees_vis = trees_layer.visualize(palette=["#800080"], min=1, max=1)
                # Blend zones over basemap
                composite = (s2_rgb
                             .blend(solar_vis)
                             .blend(roof_vis)
                             .blend(trees_vis))
                # Use a square bounding box around ROI so the PNG aspect is 1:1
                roi_bbox = roi.bounds(maxError=1)
                thumb_url = composite.getThumbURL({
                    "region":     roi_bbox,
                    "dimensions": 1600,
                    "format":     "png",
                })
                resp = requests.get(thumb_url, timeout=60)
                if resp.status_code == 200 and resp.content[:8].startswith(b"\x89PNG"):
                    intervention_map_png = resp.content
                else:
                    logger.warning(f"PNG snapshot HTTP {resp.status_code}, "
                                   f"first bytes: {resp.content[:16]!r}")
            except Exception as png_exc:
                # PNG generation is best-effort — HTML download is the primary
                # deliverable, so we don't fail the pipeline if the thumbnail
                # service hiccups (it's rate-limited and occasionally times out).
                logger.warning(f"Intervention PNG snapshot failed: {png_exc}")
                errors.append(f"Map PNG snapshot warning: {png_exc}")

        except Exception as exc:
            errors.append(f"Intervention map warning: {exc}")
            logger.warning(f"Intervention map build failed: {exc}")

        status.update(label="✅ Pipeline complete!", state="complete", expanded=False)

    return PipelineResult(
        city=city, lat=lat, lon=lon,
        peak_lst=peak_lst, lst_p5=lst_p5, lst_p50=lst_p50, lst_std=lst_std,
        uhi_intensity=uhi_intensity,
        urban_mean_lst=urban_mean_lst, rural_mean_lst=rural_mean_lst,
        solar_area=solar_area_final,
        roof_area=roof_area_final,
        trees_area=trees_area_final,
        water_area=water_area, veg_area=veg_area, bare_area=bare_area,
        agri_area=agri_area, urban_area=urban_area,
        roi_area_m2=ROI_AREA_M2,
        modis_lst_p95=modis_lst_p95,
        zones_validated=zones_validated,
        sanity_warnings=sanity_warnings,
        map_html=map_html, errors=errors,
        intervention_map_html=intervention_map_html,
        intervention_map_png=intervention_map_png,
    )


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _gauge(result: PipelineResult, t_high: float, t_crit: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result.peak_lst,
        delta={"reference": t_high, "increasing": {"color": "#ef4444"},
               "decreasing": {"color": "#22c55e"}},
        title={"text": "Peak LST P95 (°C)", "font": {"size": 14, "color": "#94a3b8"}},
        number={"font": {"color": "#f1f5f9", "family": "Space Mono"}, "suffix": " °C"},
        gauge={
            "axis": {"range": [15, 65], "tickcolor": "#64748b",
                     "tickfont": {"size": 10, "color": "#64748b"}},
            "bar": {"color": "rgba(248,250,252,0.8)", "thickness": 0.15},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [15, t_high], "color": "rgba(34,197,94,0.25)"},
                {"range": [t_high, t_crit], "color": "rgba(251,146,60,0.35)"},
                {"range": [t_crit, 65], "color": "rgba(239,68,68,0.35)"},
            ],
            "threshold": {"line": {"color": "#f97316", "width": 2}, "value": t_high}
        }
    ))
    fig.update_layout(
        height=260, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#f1f5f9"}
    )
    return fig


def _lst_distribution_chart(result: PipelineResult) -> go.Figure:
    """Visualises LST P5/P50/P95 as a horizontal range bar."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[result.lst_p5, result.lst_p50, result.peak_lst],
        y=["LST Distribution", "LST Distribution", "LST Distribution"],
        mode="markers+lines",
        marker=dict(size=[10, 12, 14], color=["#22c55e", "#f59e0b", "#ef4444"],
                    symbol=["triangle-right", "circle", "triangle-left"]),
        line=dict(color="rgba(148,163,184,0.4)", width=2),
        text=[f"P5: {result.lst_p5:.1f}°C", f"P50: {result.lst_p50:.1f}°C",
              f"P95: {result.peak_lst:.1f}°C"],
        textposition="top center",
        hovertemplate="%{text}",
        showlegend=False
    ))
    fig.update_layout(
        height=130, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="°C", showgrid=True, gridcolor="rgba(100,116,139,0.2)",
                   color="#94a3b8"),
        yaxis=dict(showticklabels=False, showgrid=False),
        font={"color": "#f1f5f9", "family": "Space Mono", "size": 10}
    )
    return fig


def _land_cover_donut(result: PipelineResult) -> go.Figure:
    labels  = ["Water", "Vegetation", "Bare Soil", "Urban (spectral)", "Agriculture"]
    values  = [result.water_area, result.veg_area, result.bare_area,
               result.urban_area, result.agri_area]
    colours = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#FFEB3B"]

    filtered = [(l, v, c) for l, v, c in zip(labels, values, colours) if v > 100]
    if not filtered:
        filtered = [("No Data", 1, "#374151")]
    fl, fv, fc = zip(*filtered)

    fig = go.Figure(go.Pie(
        labels=fl, values=fv,
        hole=0.55,
        marker=dict(colors=fc, line=dict(color="rgba(0,0,0,0.5)", width=1)),
        textfont=dict(size=10, color="white", family="Space Mono"),
        hovertemplate="%{label}<br>%{value:,.0f} m²<br>%{percent}<extra></extra>"
    ))
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#94a3b8", size=10, family="Space Mono"),
                    bgcolor="rgba(0,0,0,0)", orientation="v", x=1.0),
        showlegend=True
    )
    return fig


def _zone_bar_chart(result: PipelineResult) -> go.Figure:
    labels   = ["Solar PV", "Cool Roof", "Afforestation"]
    values   = [result.solar_area or 1, result.roof_area or 1, result.trees_area or 1]
    colours  = ["#f97316", "#ef4444", "#a855f7"]
    benefits = [
        f"{result.solar_kw:,.0f} kW",
        f"{result.cool_roof_cooling_kw:,.0f} kW cooling",
        f"{result.co2_offset_kg:,.0f} kg CO₂/yr"
    ]

    fig = go.Figure([
        go.Bar(
            x=labels, y=values,
            marker=dict(color=colours,
                        line=dict(color="rgba(255,255,255,0.1)", width=1)),
            text=[f"{v:,.0f} m²<br>{b}" for v, b in zip(values, benefits)],
            textposition="outside",
            textfont=dict(size=10, color="#cbd5e1", family="Space Mono")
        )
    ])
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(title="Area (m²) · log scale", type="log",
                   showgrid=True, gridcolor="rgba(100,116,139,0.15)",
                   color="#64748b", tickfont=dict(size=9, family="Space Mono")),
        xaxis=dict(color="#94a3b8", tickfont=dict(size=11, family="Syne")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f1f5f9"}, showlegend=False
    )
    return fig


def _uhi_timeline_gauge(result: PipelineResult) -> go.Figure:
    """Horizontal bullet chart: Urban vs Rural mean LST."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[result.rural_mean_lst], y=["Temperature"],
        orientation="h", name="Rural Mean",
        marker=dict(color="rgba(34,197,94,0.6)"), width=0.4,
    ))
    fig.add_trace(go.Bar(
        x=[result.uhi_intensity], y=["Temperature"],
        orientation="h", name=f"UHI Δ (+{result.uhi_intensity:.1f}°C)",
        base=[result.rural_mean_lst],
        marker=dict(color="rgba(239,68,68,0.7)"), width=0.4,
    ))
    fig.update_layout(
        height=110, margin=dict(l=10, r=10, t=20, b=20),
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="°C", color="#94a3b8",
                   tickfont=dict(size=9, family="Space Mono")),
        yaxis=dict(showticklabels=False),
        legend=dict(font=dict(color="#94a3b8", size=9, family="Space Mono"),
                    bgcolor="rgba(0,0,0,0)", orientation="h",
                    x=0, y=1.4),
        font={"color": "#f1f5f9"}
    )
    return fig


# ── Export ────────────────────────────────────────────────────────────────────

def _export_csv(result: PipelineResult, recs: list[Recommendation]) -> bytes:
    rows = [
        # ── Thermal ──────────────────────────────────────────────────────────
        {"Section": "Thermal", "Metric": "City / ROI",         "Value": result.city,               "Unit": "—"},
        {"Section": "Thermal", "Metric": "Peak LST (P95)",      "Value": round(result.peak_lst, 2), "Unit": "°C"},
        {"Section": "Thermal", "Metric": "LST P50 (Median)",    "Value": round(result.lst_p50, 2),  "Unit": "°C"},
        {"Section": "Thermal", "Metric": "LST P5 (Cool)",       "Value": round(result.lst_p5, 2),   "Unit": "°C"},
        {"Section": "Thermal", "Metric": "LST Std Dev",         "Value": round(result.lst_std, 2),  "Unit": "°C"},
        {"Section": "Thermal", "Metric": "Urban Mean LST",      "Value": round(result.urban_mean_lst,2),"Unit":"°C"},
        {"Section": "Thermal", "Metric": "Rural Mean LST",      "Value": round(result.rural_mean_lst,2),"Unit":"°C"},
        {"Section": "Thermal", "Metric": "UHI Intensity",       "Value": round(result.uhi_intensity,2),"Unit":"°C"},
        {"Section": "Thermal", "Metric": "WBGT Estimate",       "Value": round(result.wbgt_estimate,1),"Unit":"°C"},
        {"Section": "Thermal", "Metric": "Heat Stress Category","Value": result.heat_stress_category,"Unit":"—"},
        # ── Land Cover ───────────────────────────────────────────────────────
        {"Section": "Land Cover", "Metric": "Urban Area",       "Value": round(result.urban_area),  "Unit": "sq m"},
        {"Section": "Land Cover", "Metric": "Vegetation Area",  "Value": round(result.veg_area),    "Unit": "sq m"},
        {"Section": "Land Cover", "Metric": "Bare Soil Area",   "Value": round(result.bare_area),   "Unit": "sq m"},
        {"Section": "Land Cover", "Metric": "Water Area",       "Value": round(result.water_area),  "Unit": "sq m"},
        {"Section": "Land Cover", "Metric": "Agriculture Area", "Value": round(result.agri_area),   "Unit": "sq m"},
        {"Section": "Land Cover", "Metric": "ROI Total Area",   "Value": round(result.roi_area_m2), "Unit": "sq m"},
        # ── Interventions ────────────────────────────────────────────────────
        {"Section": "Interventions", "Metric": "Cool Roof Area","Value": round(result.roof_area),   "Unit": "sq m"},
        {"Section": "Interventions", "Metric": "Solar PV Area", "Value": round(result.solar_area),  "Unit": "sq m"},
        {"Section": "Interventions", "Metric": "Afforestation Area","Value":round(result.trees_area),"Unit":"sq m"},
        # ── Impact ───────────────────────────────────────────────────────────
        {"Section": "Impact", "Metric": "Annual CO₂ Offset",   "Value": round(result.co2_offset_kg),"Unit":"kg/yr"},
        {"Section": "Impact", "Metric": "Solar Capacity",      "Value": round(result.solar_kw),    "Unit": "kW"},
        {"Section": "Impact", "Metric": "Cool Roof Saving",    "Value": round(result.cool_roof_cooling_kw),"Unit":"kW"},
    ]
    # Append recommendations summary
    for i, r in enumerate(recs, 1):
        rows.append({
            "Section": "Recommendations",
            "Metric": f"Priority {r.priority}: {r.title}",
            "Value": r.impact,
            "Unit": f"Cost={r.cost_band} | Time={r.timeframe}"
        })

    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


# ── App layout ────────────────────────────────────────────────────────────────
def main() -> None:

    if not initialize_earth_engine():
        st.stop()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="border-bottom:1px solid rgba(56,189,248,0.2);padding-bottom:16px;margin-bottom:20px">
      <h1 style="font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;
                 background:linear-gradient(135deg,#38bdf8,#818cf8);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 margin:0;letter-spacing:-1px">
        🌡 Urban Heat Island  &nbsp;<span style="font-size:1rem;opacity:0.5;font-weight:400">AI · Research Platform · v5</span>
      </h1>
      <p style="color:#64748b;font-family:'Space Mono',monospace;font-size:0.75rem;margin:6px 0 0">
        MODIS P95 (analytics) · Landsat 30m percentile-rank zones · Sentinel-2 spectral · 5-class RF · 2-of-4 water vote
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────
    if "result" not in st.session_state:
        st.session_state["result"] = None

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Parameters")

        with st.expander("📡 Data Acquisition", expanded=True):
            cloud_thresh = st.slider("Max Cloud Cover (%)", 5, 60, 25,
                help="Lower = fewer images, higher radiometric quality")
            st.caption("Uses Landsat 8 + 9 merged for denser temporal coverage.")

        with st.expander("🧠 ML Classifier", expanded=True):
            texture_thresh = st.slider("GLCM Entropy Threshold", 1.0, 8.0, 3.5, step=0.5,
                help=(
                    "Separates smooth rooftops (< thresh → Solar/Cool Roof eligible) "
                    "from rough urban canopy (≥ thresh → Afforestation). "
                    "3–4 is optimal for most South Asian cities."
                ))
            st.caption("5-class RF: Water · Vegetation · Bare Soil · Urban · Agriculture")

        with st.expander("🌡️ Thermal Thresholds", expanded=True):
            t_high = st.number_input("Actionable Heat ≥ (°C)", value=38.0, step=0.5,
                help="Used for gauge display and recommendations. Zones use Landsat P60/P80 percentile-rank internally (aerosol-bias-immune).")
            t_crit = st.number_input("Critical Heat ≥ (°C)",   value=42.0, step=0.5,
                help="Used for gauge display and Cool Roof recommendation trigger. Must be above Actionable.")
            if t_crit <= t_high:
                st.warning("⚠️ Critical threshold must exceed Actionable.")
            st.caption("ℹ️ Zone delineation uses Landsat LST P60/P80 percentile-rank — aerosol-bias-immune. These °C values are used for gauge coloring and recommendations only.")

        with st.expander("📋 Zone Legend", expanded=False):
            st.markdown("""
| Symbol | Zone | Trigger |
|--------|------|---------|
| 🟠 | Solar PV | Built-up + LST ≥ T_high |
| 🔴 | Cool Roof | Built-up + LST ≥ T_crit |
| 🟣 | Afforestation | Rough urban + LST ≥ T_high |

**Land Cover Classes (RF):**
🔵 Water · 🟢 Vegetation · 🟡 Bare Soil · 🔴 Urban · 🟡 Agriculture
""")

        with st.expander("ℹ️ Methodology", expanded=False):
            st.markdown("""
**LST:** Landsat C2 Level-2 ST_B10 direct scale/offset  
&nbsp;&nbsp;&nbsp;&nbsp;`LST_C = DN × 0.00341802 + 149.0 − 273.15`  
&nbsp;&nbsp;&nbsp;&nbsp;USGS Radiative Transfer Equation, ±2°C accuracy  
**Cross-check:** MODIS Terra MOD11A1 P95 (1 km daily)  
**Emissivity:** NDVI threshold method (Sobrino 2004) — display only  
**Land Cover:** ESA WorldCover v2 labels + 5-class RF  
**Rooftop gate:** NDBI > 0.20 AND IBI > 0.10 (primary)  
&nbsp;&nbsp;&nbsp;&nbsp;OR RF class=3 AND NDBI > 0.15 AND IBI > 0.05  
**Feature space:** 14 spectral indices (IBI, BU, BSI…)  
**UHI Intensity:** Urban mean − Rural mean LST  
**WBGT:** Approximated from P95 LST (outdoor conditions)
""")

    # ── Search bar ────────────────────────────────────────────────────────────
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        city = st.text_input(
            "🔍 Region of Interest",
            value="Chennai",
            label_visibility="collapsed",
            placeholder="Enter city, district, or campus name…"
        )
    with col_btn:
        run_clicked = st.button("🚀 Analyse", use_container_width=True, type="primary")

    if run_clicked:
        if t_crit <= t_high:
            st.error("Fix thresholds before running.")
        else:
            result = run_pipeline(city, cloud_thresh, t_high, t_crit, texture_thresh)
            st.session_state["result"] = result

    result: Optional[PipelineResult] = st.session_state.get("result")

    # ── Results ───────────────────────────────────────────────────────────────
    if result:
        if result.errors:
            for err in result.errors:
                st.warning(err)

        # ── Map ───────────────────────────────────────────────────────────────
        st.markdown(f"### 🛰️ Geospatial View — {result.city}")
        st.markdown(
            f'<div class="info-bar">'
            f'📍 <code>{result.lat:.4f}°N, {result.lon:.4f}°E</code> &nbsp;·&nbsp; '
            f'5 km radius ROI ({result.roi_area_m2/1e6:.1f} km²) &nbsp;·&nbsp; '
            f'UHI Class: <b>{result.uhi_class}</b> &nbsp;·&nbsp; '
            f'Heat Stress: <b>{result.heat_stress_category}</b> (WBGT ≈ {result.wbgt_estimate:.1f}°C)'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Sanity warnings ────────────────────────────────────────────────────
        if result.sanity_warnings:
            with st.expander(f"⚠️ {len(result.sanity_warnings)} Data Quality Warning(s) — click to review", expanded=True):
                for w in result.sanity_warnings:
                    st.warning(w)
                st.caption(
                    "These warnings do not stop the analysis but indicate reduced confidence "
                    "in zone areas. Derived metrics (CO₂, kW) use capped values where applicable."
                )

        # ── MODIS cross-check banner ───────────────────────────────────────────
        if result.modis_lst_p95 > 0:
            delta = result.modis_lst_p95 - result.peak_lst
            st.markdown(
                f'<div class="info-bar" style="border-color:rgba(168,139,250,0.3)">'
                f'🛰️ <b>Thermal source comparison:</b> &nbsp;'
                f'MODIS P95 (analytics) = <code>{result.peak_lst:.1f}°C</code> &nbsp;|&nbsp; '
                f'Landsat C2 P95 (30m spatial) = <code>{result.modis_lst_p95:.1f}°C</code> &nbsp;|&nbsp; '
                f'Δ = <code>{delta:+.1f}°C</code> aerosol bias &nbsp;·&nbsp; '
                f'Zones use Landsat percentile-rank (P60/P80) — aerosol-bias-immune'
                f'</div>',
                unsafe_allow_html=True
            )
        components.html(result.map_html, height=580)

        # ── Analytics Dashboard ───────────────────────────────────────────────
        st.divider()
        st.markdown("### 📊 Thermal Analytics")

        # Row 1: Gauge | LST distribution | UHI bar
        col_g, col_lst, col_uhi = st.columns([1, 1.2, 1], gap="medium")

        with col_g:
            st.markdown("##### Peak LST Gauge")
            st.plotly_chart(_gauge(result, t_high, t_crit), use_container_width=True)

        with col_lst:
            st.markdown("##### LST Statistical Distribution")
            st.plotly_chart(_lst_distribution_chart(result), use_container_width=True)
            st.markdown(
                f'<div style="font-family:\'Space Mono\',monospace;font-size:0.75rem;'
                f'color:#94a3b8;line-height:1.8">'
                f'P05 &nbsp;= <b style="color:#22c55e">{result.lst_p5:.1f} °C</b><br>'
                f'P50 &nbsp;= <b style="color:#f59e0b">{result.lst_p50:.1f} °C</b><br>'
                f'P95 &nbsp;= <b style="color:#ef4444">{result.peak_lst:.1f} °C</b><br>'
                f'σ &nbsp;&nbsp;&nbsp;= <b style="color:#a78bfa">{result.lst_std:.1f} °C</b>'
                f'</div>',
                unsafe_allow_html=True
            )

        with col_uhi:
            st.markdown("##### UHI Intensity")
            st.plotly_chart(_uhi_timeline_gauge(result), use_container_width=True)
            st.markdown(
                f'<div style="font-family:\'Space Mono\',monospace;font-size:0.75rem;'
                f'color:#94a3b8;line-height:2">'
                f'Urban mean &nbsp;= <b style="color:#ef4444">{result.urban_mean_lst:.1f}°C</b><br>'
                f'Rural mean &nbsp;&nbsp;= <b style="color:#22c55e">{result.rural_mean_lst:.1f}°C</b><br>'
                f'UHI Δ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <b style="color:#f97316">+{result.uhi_intensity:.1f}°C</b><br>'
                f'Class &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <b>{result.uhi_class}</b>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Row 2: Land cover donut | Zone bar chart | Key metrics
        st.divider()
        st.markdown("### 🗺️ Land Cover & Intervention Zones")
        col_donut, col_bars, col_kpi = st.columns([1, 1.2, 1], gap="medium")

        with col_donut:
            st.markdown("##### 5-Class Land Cover (RF)")
            st.plotly_chart(_land_cover_donut(result), use_container_width=True)

        with col_bars:
            st.markdown("##### Intervention Zone Distribution")
            st.plotly_chart(_zone_bar_chart(result), use_container_width=True)

        with col_kpi:
            st.markdown("##### Key Impact Metrics")
            confidence = "✅ Validated" if result.zones_validated else "⚠️ Capped estimate"
            st.caption(f"Zone confidence: {confidence}")
            st.metric("🌳 CO₂ Sequestration",
                      f"{result.co2_offset_kg:,.0f} kg/yr",
                      delta="10 t CO₂/ha/yr · IPCC AR6 tropical rate")
            st.metric("⚡ Solar Capacity",
                      f"{result.solar_kw:,.0f} kW",
                      delta="18% eff. × 0.70 packing factor")
            st.metric("❄️ Cool Roof Savings",
                      f"{result.cool_roof_cooling_kw:,.0f} kW",
                      delta="40 W/m² peak · Akbari 2009")
            status_label = ("🔴 CRITICAL" if result.is_critical(t_crit)
                            else "🟠 ELEVATED" if result.peak_lst >= t_high
                            else "🟢 NORMAL")
            st.metric("Zone Status", status_label)
            if result.modis_lst_p95 > 0:
                st.metric("🛰️ Landsat C2 P95 (display only)",
                          f"{result.modis_lst_p95:.1f} °C",
                          delta=f"Δ {result.modis_lst_p95 - result.peak_lst:+.1f}°C vs MODIS")

        # ── Scientific Recommendations Engine ────────────────────────────────
        st.divider()
        st.markdown("### 🔬 Scientific Recommendations Engine")
        st.caption(
            "Evidence-based interventions ranked by urgency × impact. "
            "Each recommendation cites peer-reviewed literature and includes "
            "quantified impact estimates, cost bands, and implementation timelines."
        )

        recs = generate_recommendations(result, t_crit)

        if not recs:
            st.success("✅ No critical interventions identified for this region.")
        else:
            for rec in recs:
                priority_colours = {1: "#ef4444", 2: "#f97316", 3: "#eab308", 4: "#22c55e"}
                priority_labels  = {1: "CRITICAL", 2: "HIGH", 3: "MEDIUM", 4: "LOW"}
                border_color = priority_colours.get(rec.priority, "#64748b")
                priority_label = priority_labels.get(rec.priority, "INFO")

                with st.container():
                    st.markdown(
                        f'<div class="rec-card rec-priority-{rec.priority}" '
                        f'style="border-color:{border_color}">'
                        f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                        f'<h4 style="margin:0;font-family:Syne,sans-serif;font-size:1rem">{rec.title}</h4>'
                        f'<span style="background:rgba(0,0,0,0.3);border:1px solid {border_color}33;'
                        f'color:{border_color};padding:2px 8px;border-radius:4px;'
                        f'font-family:Space Mono,monospace;font-size:0.7rem">'
                        f'P{rec.priority} · {priority_label}</span></div>'
                        f'<p style="color:#cbd5e1;font-size:0.88rem;margin:8px 0 6px">{rec.description}</p>'
                        f'<div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:8px">'
                        f'<span style="font-family:Space Mono,monospace;font-size:0.72rem;color:#38bdf8">'
                        f'📐 {rec.area_m2:,.0f} m²</span>'
                        f'<span style="font-family:Space Mono,monospace;font-size:0.72rem;color:#a78bfa">'
                        f'📈 {rec.impact}</span>'
                        f'<span style="font-family:Space Mono,monospace;font-size:0.72rem;color:#94a3b8">'
                        f'💰 Cost: {rec.cost_band} &nbsp;|&nbsp; 🔧 {rec.difficulty} &nbsp;|&nbsp; ⏱ {rec.timeframe}</span>'
                        f'</div>'
                        f'<details style="margin-top:10px">'
                        f'<summary style="font-size:0.75rem;color:#64748b;cursor:pointer">📚 Evidence Base</summary>'
                        f'<p style="font-size:0.75rem;color:#94a3b8;margin:6px 0 0;'
                        f'font-family:Space Mono,monospace">{rec.evidence}</p>'
                        f'</details>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # ── Intervention Action Map (planner-focused output) ─────────────────
        # A clean, single-purpose map that answers "where exactly do we install
        # solar PV / cool roofs / trees within this ROI?".  Separate from the
        # analyst map above (which carries 8+ scientific overlays) so planners
        # can grab the action plan without toggling layers.
        st.divider()
        st.markdown("### 🎯 Intervention Action Map — Where to Act")
        st.caption(
            "A focused geospatial map showing the precise locations within the "
            "5 km ROI where each intervention should be deployed. "
            "🟧 Solar PV · 🟥 Cool-roof coating · 🟪 Afforestation. "
            "Toggle individual layers via the control on the right of the map."
        )
        if result.intervention_map_html:
            components.html(result.intervention_map_html, height=560)
        else:
            st.info("Intervention map could not be rendered — see warnings above.")

        # Three-column layout: HTML download | PNG download | quick stats
        col_iv1, col_iv2, col_iv3 = st.columns([1, 1, 1.3], gap="medium")
        with col_iv1:
            if result.intervention_map_html:
                st.download_button(
                    label="🗺️ Download Map (Interactive HTML)",
                    data=result.intervention_map_html.encode("utf-8"),
                    file_name=f"{result.city.replace(' ', '_')}_Intervention_Map.html",
                    mime="text/html",
                    use_container_width=True,
                    help=(
                        "Self-contained HTML file. Open in any browser — works "
                        "offline. Share with planners, embed in reports."
                    ),
                )
            else:
                st.button("🗺️ HTML unavailable", disabled=True,
                          use_container_width=True)
        with col_iv2:
            if result.intervention_map_png:
                st.download_button(
                    label="🖼️ Download Map (PNG · 1600 px)",
                    data=result.intervention_map_png,
                    file_name=f"{result.city.replace(' ', '_')}_Intervention_Map.png",
                    mime="image/png",
                    use_container_width=True,
                    help=(
                        "1600×1600 px PNG snapshot — drop straight into "
                        "PowerPoint, Word, or LaTeX reports."
                    ),
                )
            else:
                st.button("🖼️ PNG unavailable", disabled=True,
                          use_container_width=True,
                          help="EE thumbnail service did not return a PNG. "
                               "The HTML download still works.")
        with col_iv3:
            st.markdown(
                f'<div style="font-family:\'Space Mono\',monospace;'
                f'font-size:0.78rem;color:#cbd5e1;line-height:1.85;'
                f'padding:6px 12px;border-left:3px solid #38bdf8;'
                f'background:rgba(56,189,248,0.05)">'
                f'<span style="color:#FFA500;font-weight:700">●</span> '
                f'<b>{result.solar_area:,.0f} m²</b> solar PV<br>'
                f'<span style="color:#FF0000;font-weight:700">●</span> '
                f'<b>{result.roof_area:,.0f} m²</b> cool roof<br>'
                f'<span style="color:#800080;font-weight:700">●</span> '
                f'<b>{result.trees_area:,.0f} m²</b> afforestation'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── Export ────────────────────────────────────────────────────────────
        st.divider()
        col_exp1, col_exp2 = st.columns([2, 1])
        with col_exp1:
            st.markdown("### 📥 Export Full Report")
            st.caption(
                "Exports all thermal statistics, land-cover breakdown, "
                "intervention zones, impact metrics, and prioritised "
                "recommendations as a structured CSV."
            )
        with col_exp2:
            # Sanitise filename — strip anything that isn't alphanumeric or _
            import re as _re
            safe_city = _re.sub(r"[^A-Za-z0-9_]+", "_", result.city).strip("_") or "Region"
            st.download_button(
                label="⬇️ Download Full Report (CSV)",
                data=_export_csv(result, recs),
                file_name=f"{safe_city}_UHI_Report_v5.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )

        # ── Debug ─────────────────────────────────────────────────────────────
        with st.expander("🔬 Raw Debug Values", expanded=False):
            st.json({
                "city": result.city, "lat": result.lat, "lon": result.lon,
                "roi_area_km2": round(result.roi_area_m2 / 1e6, 2),
                "lst_landsat_swa": {
                    "p5": result.lst_p5, "p50": result.lst_p50,
                    "p95": result.peak_lst, "std": result.lst_std
                },
                "lst_modis_cross_check": {
                    "p95_celsius": result.modis_lst_p95,
                    "delta_vs_landsat": round(result.peak_lst - result.modis_lst_p95, 2)
                    if result.modis_lst_p95 > 0 else "unavailable"
                },
                "uhi": {
                    "intensity_c": result.uhi_intensity,
                    "urban_mean": result.urban_mean_lst,
                    "rural_mean": result.rural_mean_lst,
                    "class": result.uhi_class
                },
                "wbgt": {
                    "estimate_c": result.wbgt_estimate,
                    "category": result.heat_stress_category
                },
                "zones_m2_capped": {
                    "solar": result.solar_area,
                    "roof": result.roof_area,
                    "trees": result.trees_area,
                    "note": "Areas capped at 35% of ROI if spectral confusion detected"
                },
                "land_cover_m2_raw": {
                    "vegetation": result.veg_area,
                    "bare_soil": result.bare_area,
                    "water": result.water_area,
                    "agriculture": result.agri_area
                },
                "derived_metrics": {
                    "co2_offset_kg_yr": result.co2_offset_kg,
                    "solar_kw": result.solar_kw,
                    "cool_roof_kw": result.cool_roof_cooling_kw,
                    "assumptions": {
                        "solar_efficiency": "18% monocrystalline × 0.70 packing",
                        "co2_rate": "10 t CO2/ha/yr · IPCC AR6 tropical plantation",
                        "cool_roof": "40 W/m2 peak, annual avg 10 W/m2 · Akbari 2009"
                    }
                },
                "validation": {
                    "zones_validated": result.zones_validated,
                    "sanity_warnings": result.sanity_warnings,
                    "pipeline_errors": result.errors
                }
            })

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;
             border:1px dashed rgba(56,189,248,0.2);border-radius:12px;margin-top:20px">
          <div style="font-size:3rem">🌡️</div>
          <h3 style="font-family:Syne,sans-serif;color:#38bdf8;font-weight:800">
            Ready to Analyse
          </h3>
          <p style="color:#64748b;font-family:Space Mono,monospace;font-size:0.85rem">
            Enter a city name and click <b>Analyse</b> to run the full pipeline.<br>
            Results include LST, UHI Intensity, Land Cover, and Scientific Recommendations.
          </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()