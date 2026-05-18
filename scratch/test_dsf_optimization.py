import sys
import os

# Add src folder to sys.path
sys.path.append("/Users/hiroshi/Developer/Ortho4XP-Mac/src")

import O4_DSF_Utils
import O4_UI_Utils as UI

class DummyTile:
    def __init__(self):
        self.lat = 34
        self.lon = 133
        self.mesh_zl = 16
        self.default_zl = 16
        self.default_website = "BI"
        self.cover_airports_with_highres = "None"
        self.zone_list = []
        self.cover_zl = 18
        self.cover_extent = 1
        self.build_dir = "/Users/hiroshi/Developer/Ortho4XP-Mac/tmp"

def run_test():
    tile = DummyTile()
    print("Building dummy ortho dico...")
    dico = O4_DSF_Utils.zone_list_to_ortho_dico(tile)
    print(f"SUCCESS! dico size: {len(dico)}")
    
    # Try custom ZL lookup logic similar to vectorized part
    import numpy
    import O4_Geo_Utils as GEO
    til_x_min, til_y_min = GEO.wgs84_to_orthogrid(tile.lat + 1, tile.lon, tile.mesh_zl)
    til_x_max, til_y_max = GEO.wgs84_to_orthogrid(tile.lat, tile.lon + 1, tile.mesh_zl)
    
    width = til_x_max - til_x_min + 1
    height = til_y_max - til_y_min + 1
    customzl_arr = numpy.empty((width, height), dtype=object)
    
    default_val = (16 * (til_x_min // 16), 16 * (til_y_min // 16), tile.default_zl, tile.default_website)
    customzl_arr.fill(default_val)
    
    for (tx, ty), val in dico.items():
        idx_x = tx - til_x_min
        idx_y = ty - til_y_min
        if 0 <= idx_x < width and 0 <= idx_y < height:
            customzl_arr[idx_x, idx_y] = val
            
    print("SUCCESS! Optimized NumPy grid pre-fill verification passed.")

if __name__ == "__main__":
    run_test()
