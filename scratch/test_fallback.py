import sys
import os
sys.path.append('src')
import O4_Config_Utils as CFG
import O4_DEM_Utils as DEM
import O4_UI_Utils as UI

UI.verbosity = 1

lat = 25
lon = 128
print(f"Testing tile {lat}, {lon}")

# Set custom_dem to the directory
CFG.custom_dem = "/Users/hiroshi/Developer/Ortho4XP-Mac/Elevation_data/JapanDEM1-hgt"

try:
    dem = DEM.DEM(lat, lon, source=CFG.custom_dem, info_only=True)
    print("Success!")
except Exception as e:
    print(f"Failed with error: {e}")
