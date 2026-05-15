"""
Standalone Map Exporter
Downloads clean, high-resolution PNGs of NDVI, NDBI, and LST for the IEEE Paper.
"""

import ee
import requests
from auth import initialize_earth_engine
from data_processing import get_satellite_data, calculate_physics

print("🛰️ Initializing Earth Engine...")
if not initialize_earth_engine():
    print("❌ Auth failed! Please check your credentials.")
    exit()

# Set coordinates for your primary ROI (Kattankulathur)
lat, lon = 12.823, 80.044
roi_geometry = ee.Geometry.Point([lon, lat]).buffer(3000)

print("📡 Fetching Sentinel-2 & Landsat-8 Data...")
s2_optical, l8_thermal = get_satellite_data(roi_geometry, cloud_thresh=30)
_, lst_celsius = calculate_physics(s2_optical, l8_thermal)

print("⚙️ Computing NDVI & NDBI Layers...")
ndvi = s2_optical.normalizedDifference(['B8', 'B4']).rename('NDVI')
ndbi = s2_optical.normalizedDifference(['B11', 'B8']).rename('NDBI')

# Define professional color palettes for the paper
vis_ndvi = {'min': 0, 'max': 0.8, 'palette': ['white', 'lightgreen', 'darkgreen']}
vis_ndbi = {'min': -0.2, 'max': 0.4, 'palette': ['white', 'silver', 'dimgray']}
vis_lst = {'min': 30.0, 'max': 45.0, 'palette': ['blue', 'cyan', 'yellow', 'red']} 

def download_image(ee_image, vis_params, filename):
    print(f"📥 Generating high-res {filename}...")
    # Get a direct download URL from Google Earth Engine servers
    url = ee_image.getThumbURL({
        'min': vis_params['min'],
        'max': vis_params['max'],
        'palette': vis_params['palette'],
        'region': roi_geometry,
        'dimensions': 1000, # 1000px wide for crisp IEEE printing
        'format': 'png'
    })
    
    # Download and save the image
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"✅ Saved {filename}!")
    else:
        print(f"❌ Failed to download {filename}")

# Execute the downloads
download_image(ndvi, vis_ndvi, 'ndvi.png')
download_image(ndbi, vis_ndbi, 'ndbi.png')
download_image(lst_celsius, vis_lst, 'lst.png')

print("🎉 All paper graphics downloaded successfully!")