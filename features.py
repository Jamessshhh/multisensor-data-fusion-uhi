"""
Urban Heat Island AI - Feature Engineering Module (v3.1)
This module contains functions for engineering spectral indices from Sentinel-2 imagery,
used as features for the land cover classification model.
"""

import ee

def _calculate_core_indices(s2_img: ee.Image) -> ee.Image:
    """Calculates core spectral indices: NDVI, NDBI, MNDWI, NDWI, UI."""
    ndvi = s2_img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = s2_img.normalizedDifference(["B11", "B8"]).rename("NDBI")
    mndwi = s2_img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndwi = s2_img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ui = s2_img.normalizedDifference(["B12", "B8"]).rename("UI")
    return ee.Image([ndvi, ndbi, mndwi, ndwi, ui])

def _calculate_soil_veg_indices(s2_img: ee.Image, bands: dict) -> ee.Image:
    """Calculates indices for soil and vegetation discrimination: BSI, SAVI, EVI."""
    b2, b4, b8, b11 = bands['B2'], bands['B4'], bands['B8'], bands['B11']
    
    bsi = (b11.add(b4).subtract(b8).subtract(b2)).divide(
           b11.add(b4).add(b8).add(b2)).rename("BSI")

    savi = b8.subtract(b4).multiply(1.5).divide(
           b8.add(b4).add(0.5)).rename("SAVI")

    evi = b8.subtract(b4).multiply(2.5).divide(
          b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1)
    ).rename("EVI")
    
    return ee.Image([bsi, savi, evi])

def _calculate_urban_indices(s2_img: ee.Image, bands: dict, core_indices: ee.Image) -> ee.Image:
    """Calculates urban-specific and composite indices: NBI, BU, RI, IBI, VgNIRBI."""
    b3, b4, b8, b11 = bands['B3'], bands['B4'], bands['B8'], bands['B11']
    ndvi = core_indices.select('NDVI')
    mndwi = core_indices.select('MNDWI')
    savi = _calculate_soil_veg_indices(s2_img, bands).select('SAVI') # Recalculate SAVI if needed or pass it in

    nbi = b4.multiply(b11).divide(b8.add(1e-6)).rename("NBI")
    bu = core_indices.select('NDBI').subtract(ndvi).rename("BU")
    ri = b4.subtract(b3).divide(b4.add(b3).add(1e-6)).rename("RI")
    
    # IBI requires SAVI and MNDWI
    ibi_numerator = s2_img.normalizedDifference(['B11', 'B8']) # NDBI
    ibi_denominator = (savi.add(mndwi)).divide(2)
    ibi = ibi_numerator.subtract(ibi_denominator).divide(ibi_numerator.add(ibi_denominator)).rename("IBI")

    # VgNIRBI is similar to NDWI but used for urban-veg boundary
    vgnirbi = s2_img.normalizedDifference(['B3', 'B8']).rename('VgNIRBI')
    
    # NDSI is identical to MNDWI in this band configuration
    ndsi = mndwi.rename('NDSI')

    return ee.Image([nbi, bu, ri, ibi, vgnirbi, ndsi])

def engineer_features(s2_img: ee.Image) -> ee.Image:
    """
    Computes 14 spectral indices that maximally separate the 5 target classes.
    This function now orchestrates calls to smaller, more specific feature calculation functions.
    """
    bands = {
        "B2": s2_img.select("B2"), "B3": s2_img.select("B3"), "B4": s2_img.select("B4"),
        "B8": s2_img.select("B8"), "B11": s2_img.select("B11"), "B12": s2_img.select("B12")
    }

    core_indices = _calculate_core_indices(s2_img)
    soil_veg_indices = _calculate_soil_veg_indices(s2_img, bands)
    urban_indices = _calculate_urban_indices(s2_img, bands, core_indices)

    # Combine all engineered features
    all_features = core_indices.addBands(soil_veg_indices).addBands(urban_indices)
    
    # Add raw bands as well
    raw_bands = s2_img.select(list(bands.keys()))
    
    return s2_img.addBands(all_features).addBands(raw_bands)
