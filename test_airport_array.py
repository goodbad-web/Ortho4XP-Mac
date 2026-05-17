import pickle
import numpy as np
import O4_Geo_Utils as GEO

class DummyTile:
    def __init__(self):
        self.lat = 24
        self.lon = 125
        self.cover_extent = 1.0
        self.cover_zl = 18

tile = DummyTile()
airport_array = np.zeros((4096, 4096), dtype=np.bool_)

f = open('Tiles/zOrtho4XP_+24+125/Data+24+125.apt', 'rb')
dico_airports = pickle.load(f)
f.close()

for airport in dico_airports:
    (xmin, ymin, xmax, ymax) = dico_airports[airport]["boundary"].bounds
    xmin -= 1000 * tile.cover_extent * GEO.m_to_lon(tile.lat)
    xmax += 1000 * tile.cover_extent * GEO.m_to_lon(tile.lat)
    ymax += 1000 * tile.cover_extent * GEO.m_to_lat
    ymin -= 1000 * tile.cover_extent * GEO.m_to_lat

    (til_x_left, til_y_top) = GEO.wgs84_to_orthogrid(ymax + tile.lat, xmin + tile.lon, tile.cover_zl)
    (ymax_deg, xmin_deg) = GEO.gtile_to_wgs84(til_x_left, til_y_top, tile.cover_zl)
    ymax_final = ymax_deg - tile.lat
    xmin_final = xmin_deg - tile.lon

    (til_x_left2, til_y_top2) = GEO.wgs84_to_orthogrid(ymin + tile.lat, xmax + tile.lon, tile.cover_zl)
    (ymin_deg, xmax_deg) = GEO.gtile_to_wgs84(til_x_left2 + 16, til_y_top2 + 16, tile.cover_zl)
    ymin_final = ymin_deg - tile.lat
    xmax_final = xmax_deg - tile.lon

    colmin = round(max(0, xmin_final) * 4095)
    colmax = round(min(1, xmax_final) * 4095)
    rowmax = round((1 - max(0, ymin_final)) * 4095)
    rowmin = round((1 - min(1, ymax_final)) * 4095)
    
    print(f"{airport}: colmin={colmin}, colmax={colmax}, rowmin={rowmin}, rowmax={rowmax}, width={colmax-colmin}, height={rowmax-rowmin}")
    airport_array[rowmin:rowmax+1, colmin:colmax+1] = 1

print("Total 1s in airport_array:", np.sum(airport_array))
