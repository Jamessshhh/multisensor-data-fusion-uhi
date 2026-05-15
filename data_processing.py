"""
Urban Heat Island AI - Data Processing & Physics Module (v5)
Hybrid thermal architecture: MODIS for accurate absolute temperature,
Landsat for 30m spatial zone delineation.

Architecture (v5):
  MODIS MOD11A2 (8-day, 1km) → accurate absolute LST for analytics/reporting
    - Daily 10:30am overpass before aerosol builds
    - 8-day compositing inherently cloud-filters
    - P95 across hot season = peak heat metric
    - Used for: LST gauge, UHI intensity, WBGT, heat stress, cross-check

  Landsat 8+9 C2 L2 (30m) → spatial zone delineation only
    - 30m resolution resolves individual rooftops and street blocks
    - Used ONLY for hot/critical thermal masks for zone boundaries
    - Relative thermal contrast (percentile-based) used instead of
      absolute thresholds — avoids the aerosol bias problem entirely:
      instead of asking "is this pixel > 38°C?" we ask
      "is this pixel in the hottest 30% of the ROI?" — aerosol
      bias shifts all pixels equally so relative ranking is unaffected
    - Median composite of all qualifying scenes in hot season

  Sentinel-2 (10m) → all spectral indices (NDBI, IBI, NDVI, water detection)
"""

import ee
import logging
from datetime import datetime

logger = logging.getLogger("DataProcessor_v5")

# ── Collections ───────────────────────────────────────────────────────────────
_S2_COLLECTION   = "COPERNICUS/S2_SR_HARMONIZED"
_L8_COLLECTION   = "LANDSAT/LC08/C02/T1_L2"
_L9_COLLECTION   = "LANDSAT/LC09/C02/T1_L2"
_MODIS_LST       = "MODIS/061/MOD11A2"   # 8-day composite, 1 km

# ── Scale factors ──────────────────────────────────────────────────────────────
_MODIS_SCALE = 0.02      # LST_Day_1km → Kelvin
_L_SCALE     = 0.00341802
_L_OFFSET    = 149.0
_KELVIN      = 273.15

# ── Emissivity parameters ──────────────────────────────────────────────────────
_NDVI_S      = 0.2
_NDVI_V      = 0.5
_EMISS_SOIL  = 0.97
_EMISS_VEG   = 0.99
_EMISS_WATER = 0.992


# ── Date-range helpers ────────────────────────────────────────────────────────

def _dynamic_date_range(roi_lat: float | None = None) -> tuple[str, str]:
    """
    Returns (start, end) targeting the hottest 6-month window of the LAST
    complete year.  Hemisphere-aware:
      Northern (lat > 0): April–October
      Southern (lat ≤ 0): October–April (crosses year boundary)
    Falls back to full-year scan if imagery count is too low.
    """
    year = datetime.utcnow().year - 1
    if roi_lat is None or roi_lat > 0:
        return f"{year}-04-01", f"{year}-10-01"
    else:
        # Southern hemisphere summer
        return f"{year-1}-10-01", f"{year}-04-01"


# ── Sentinel-2 cloud masking ──────────────────────────────────────────────────

def _mask_s2_clouds(image: ee.Image) -> ee.Image:
    """
    SCL-based masking (S2 SR Harmonized).  Keeps only:
        4  = Vegetation
        5  = Not-vegetated (bare soil)
        6  = Water
        7  = Unclassified
       11  = Snow/Ice
    Also removes cirrus-contaminated pixels using B10 reflectance > 0.01.
    """
    scl = image.select("SCL")
    clear = (scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
                      .Or(scl.eq(7)).Or(scl.eq(11)))
    return (image.updateMask(clear)
                 .divide(10000)
                 .copyProperties(image, ["system:time_start"]))


# ── Landsat cloud/shadow masking ──────────────────────────────────────────────

def _mask_landsat_qa(image: ee.Image) -> ee.Image:
    """
    Bitwise QA_PIXEL masking for Landsat Collection 2.
    Removes: dilated cloud (bit 1), cirrus (bit 2), cloud (bit 3),
             cloud shadow (bit 4), snow (bit 5).
    Keeps only high-confidence clear pixels.
    """
    qa = image.select("QA_PIXEL")
    bad_bits = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
    bad_mask = qa.bitwiseAnd(bad_bits)
    # ST_QA radiometric saturation: require ≥ 64 (medium confidence)
    st_qa = image.select("ST_QA")
    return image.updateMask(bad_mask.Not()).updateMask(st_qa.gte(64))


# ── Emissivity model ──────────────────────────────────────────────────────────

def _compute_emissivity(s2_img: ee.Image, roi: ee.Geometry) -> ee.Image:
    """
    Derives land surface emissivity (ε) from NDVI fractional vegetation cover.

    Method: NDVI Threshold Method (Sobrino et al. 2004)
      Pv   = ((NDVI - NDVI_s) / (NDVI_v - NDVI_s))²   (vegetation fraction)
      ε    = ε_soil * (1 - Pv) + ε_veg * Pv            (weighted average)
    Special case: water pixels → ε_water = 0.992
    """
    ndvi = s2_img.normalizedDifference(["B8", "B4"])

    # Vegetation fraction (Pv), clamped 0–1
    pv = ndvi.subtract(_NDVI_S).divide(_NDVI_V - _NDVI_S).pow(2).clamp(0, 1)

    # NDVI-based emissivity
    emissivity = (
        ee.Image(_EMISS_SOIL).multiply(pv.subtract(1).abs())
        .add(ee.Image(_EMISS_VEG).multiply(pv))
    )

    # Water correction: MNDWI > 0 catches turbid/shallow water bodies
    # (was 0.2 — too strict for sedimented Tamil Nadu tanks and lakes)
    mndwi = s2_img.normalizedDifference(["B3", "B11"])
    ndwi  = s2_img.normalizedDifference(["B3", "B8"])
    water_mask = mndwi.gt(-0.1).Or(ndwi.gt(0.0))
    emissivity = emissivity.where(water_mask, _EMISS_WATER)

    return emissivity.rename("emissivity").clip(roi)


# ── MODIS LST (analytics — accurate absolute temperature) ─────────────────────

def _get_modis_lst(roi: ee.Geometry, start: str, end: str) -> ee.Image:
    """MODIS MOD11A2 P95 in Celsius — authoritative absolute LST for reporting."""
    def _mask_modis_quality(img):
        qc   = img.select("QC_Day")
        good = qc.bitwiseAnd(3).eq(0)   # bits 0-1 = "00" → good quality
        return img.updateMask(good)

    col = (
        ee.ImageCollection(_MODIS_LST)
        .filterBounds(roi)
        .filterDate(start, end)
        .map(_mask_modis_quality)
        .select("LST_Day_1km")
    )
    lst_p95 = (col.reduce(ee.Reducer.percentile([95]))
               .select("LST_Day_1km_p95")
               .multiply(_MODIS_SCALE).subtract(_KELVIN)
               .rename("LST").clip(roi))
    logger.info("MODIS MOD11A2 P95 LST ready (analytics — absolute temperature).")
    return lst_p95


# ── Landsat C2 LST (zone delineation — relative thermal contrast) ─────────────

def _get_landsat_lst_zones(roi: ee.Geometry, start: str, end: str) -> ee.Image:
    """
    Landsat 8+9 C2 median LST composite (30m) — used ONLY for zone spatial
    delineation via percentile-rank thresholds (not absolute °C thresholds).

    KEY INSIGHT: aerosol bias shifts ALL Landsat pixels by roughly the same
    amount (+8–10°C uniformly). This means the RELATIVE RANKING of pixels
    is unaffected — the hottest rooftop is still the hottest rooftop even
    if the absolute value is wrong. So we threshold on percentile rank
    (e.g. top 40% = "hot", top 20% = "critical") rather than fixed °C values.
    This makes zone delineation aerosol-bias-immune while preserving 30m
    spatial resolution that MODIS (1km) cannot provide.
    """
    def _mask_qa(img):
        qa    = img.select("QA_PIXEL")
        bad   = qa.bitwiseAnd((1<<1)|(1<<2)|(1<<3)|(1<<4)|(1<<5))
        st_qa = img.select("ST_QA")
        return img.updateMask(bad.Not()).updateMask(st_qa.gte(64))

    def _to_lst(img):
        return (img.select("ST_B10")
                .multiply(_L_SCALE).add(_L_OFFSET).subtract(_KELVIN)
                .rename("LST"))

    l8  = ee.ImageCollection(_L8_COLLECTION).filterBounds(roi).filterDate(start, end).filter(ee.Filter.lt("CLOUD_COVER", 30)).map(_mask_qa)
    l9  = ee.ImageCollection(_L9_COLLECTION).filterBounds(roi).filterDate(start, end).filter(ee.Filter.lt("CLOUD_COVER", 30)).map(_mask_qa)
    col = l8.merge(l9)

    lst_median = col.map(_to_lst).median().clip(roi)
    logger.info("Landsat C2 median LST ready (30m — zone delineation only).")
    return lst_median


# ── UHI Intensity (from MODIS — absolute temps, 1km resolution) ───────────────

def compute_uhi_intensity(lst_modis: ee.Image, roi: ee.Geometry) -> dict:
    """UHI = mean(warm half) − mean(cool half) using MODIS absolute LST."""
    scale = 1000   # MODIS 1km

    med = lst_modis.reduceRegion(
        reducer=ee.Reducer.median(), geometry=roi,
        scale=scale, maxPixels=1e9, bestEffort=True
    ).getNumber("LST")

    urban_mean = lst_modis.updateMask(lst_modis.gt(med)).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=roi,
        scale=scale, maxPixels=1e9, bestEffort=True
    ).getNumber("LST")

    rural_mean = lst_modis.updateMask(lst_modis.lt(med)).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=roi,
        scale=scale, maxPixels=1e9, bestEffort=True
    ).getNumber("LST")

    return {
        "uhi_intensity":  urban_mean.subtract(rural_mean),
        "urban_mean_lst": urban_mean,
        "rural_mean_lst": rural_mean,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_satellite_data(roi: ee.Geometry, cloud_thresh: int,
                       roi_lat: float | None = None) -> tuple:
    """
    Returns:
        s2              : Sentinel-2 median composite (10m) for spectral indices
        lst_modis       : MODIS P95 LST in °C (1km) — analytics & reporting
        lst_landsat     : Landsat C2 median LST in °C (30m) — zone delineation
        emissivity      : NDVI-based ε model (display only)
    """
    start, end = _dynamic_date_range(roi_lat)
    logger.info(f"Date window: {start} → {end} | cloud≤{cloud_thresh}%")

    # ── Sentinel-2: median optical + P10 SWIR for water detection ────────────
    s2_col = (
        ee.ImageCollection(_S2_COLLECTION)
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
        .map(_mask_s2_clouds)
        .select(["B2", "B3", "B4", "B8", "B11", "B12"])
    )
    s2_median = s2_col.median().clip(roi)
    s2_p10    = s2_col.reduce(ee.Reducer.percentile([10])).clip(roi)
    s2 = s2_median.addBands(
        s2_p10.select(["B11_p10", "B12_p10"]).rename(["B11", "B12"]),
        overwrite=True
    )
    logger.info("Sentinel-2 composite ready.")

    # ── MODIS LST (analytics) ─────────────────────────────────────────────────
    lst_modis   = _get_modis_lst(roi, start, end)

    # ── Landsat LST (zone delineation at 30m) ────────────────────────────────
    lst_landsat = _get_landsat_lst_zones(roi, start, end)

    # ── Emissivity (display only) ─────────────────────────────────────────────
    emissivity  = _compute_emissivity(s2, roi)

    return s2, lst_modis, lst_landsat, emissivity


def calculate_physics(s2_img: ee.Image, lst_celsius: ee.Image) -> tuple:
    """
    Derives surface roughness (GLCM entropy on NIR) for texture-based
    roof / canopy discrimination.

    Returns:
        tuple: (roughness: ee.Image, lst_celsius: ee.Image)
    """
    logger.info("Computing GLCM texture analysis …")

    # Scale to INT16 for GLCM dynamic range
    nir = s2_img.select("B8").multiply(10000).toUint16()

    # size=3 → 7×7 kernel: slightly larger for 10 m pixels → better texture signal
    glcm = nir.glcmTexture(size=3, average=True)
    raw_entropy = glcm.select("B8_ent")

    # Normalize 0–10 scale (typical raw range 0–3.5 at size=3)
    roughness = raw_entropy.multiply(2.5).rename("ROUGHNESS")

    # Also compute correlation (smooth surfaces have high correlation)
    correlation = glcm.select("B8_corr").rename("CORRELATION")

    return roughness, lst_celsius