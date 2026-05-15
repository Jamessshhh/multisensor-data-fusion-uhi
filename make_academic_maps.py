import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import numpy as np

# Coordinates from your Earth Engine script (Kattankulathur)
lat_center, lon_center = 12.823, 80.044
offset = 0.027
extent = [lon_center - offset, lon_center + offset, lat_center - offset, lat_center + offset]

def format_academic_map(input_filename, output_filename, title, cmap_name, min_val, max_val, label):
    print(f"🎨 Formatting {input_filename}...")
    
    # 1. Load the raw image
    img = mpimg.imread(input_filename)
    
    # 2. Setup the academic figure
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300) # High-resolution for IEEE
    
    # 3. Plot the image with the geographic extent
    im = ax.imshow(img, extent=extent)
    
    # 4. Add Grid Lines
    ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # 5. Format Axis Labels
    ax.set_xlabel("Longitude (°E)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Latitude (°N)", fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # 6. Add the Colorbar
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=10, fontweight='bold')
    
    # 7. Add a simple North Arrow (Top Left)
    ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.85),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor='black', width=2, headwidth=8),
                fontsize=12, fontweight='bold', ha='center', va='top')

    # 8. NEW: Add Location Name (Bottom Right)
    # Uses a black, semi-transparent background box so it is always readable
    ax.text(0.96, 0.04, "SRM IST, Kattankulathur", 
            transform=ax.transAxes, color='white', 
            fontsize=9, fontweight='bold', ha='right', va='bottom',
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.4'))

    # Save the academic version
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {output_filename}!")

# --- Execute for all three maps ---

# NDVI: Green palette, 0 to 0.8
format_academic_map('ndvi.png', 'ndvi_academic.png', 
                    'NDVI (Vegetation Index)', 'Greens', 0.0, 0.8, 'NDVI Value')

# NDBI: Grey palette, -0.2 to 0.4
format_academic_map('ndbi.png', 'ndbi_academic.png', 
                    'NDBI (Built-Up Index)', 'Greys', -0.2, 0.4, 'NDBI Value')

# LST: Thermal (Inferno/Jet) palette, 30°C to 45°C
format_academic_map('lst.png', 'lst_academic.png', 
                    'Land Surface Temperature (LST)', 'jet', 30.0, 45.0, 'Temperature (°C)')

print("🎉 All academic maps are ready with Location Labels!")