import os
import array
import numpy
from PIL import Image
import O4_File_Names as FNAMES
import O4_Geo_Utils as GEO

def set_depth_ratio(n, node_is_coast, node_bathy, tile):
    if node_is_coast[n]:
        return 0
    else:
        return max(
            min(10 * tile.ratio_bathy * node_bathy[n] / 255, 1),
            0.1)

from O4_Recut_Water import recut_water_tris
    



def compute_depth_ratio_bounds_from_masks(
        nbr_nodes, node_coords, node_types, tile):

    water_nodes = [n for n in range(nbr_nodes) if (node_types[n] & 4 != 0)]

    node_bathy = 255 * numpy.ones(nbr_nodes, dtype = numpy.uint8)

    # Key is texture attribute at mask_zl
    # Value is an array of vertex integers
    mask_to_nodes = {}

    for n in water_nodes:
        lon = node_coords[5 * n + 0]
        lat = node_coords[5 * n + 1]
        mask_attr = GEO.wgs84_to_orthogrid(lat, lon, tile.mask_zl)
        if (mask_attr) in mask_to_nodes:
            mask_to_nodes[mask_attr].append(n)
        else:
            mask_to_nodes[mask_attr] = array.array('i',(n,))

    for mask_attr in mask_to_nodes:
        mask_file = os.path.join(FNAMES.mask_dir(tile.lat, tile.lon),
                FNAMES.distance_mask(*mask_attr))
        if not os.path.isfile(mask_file):
            continue
        img = Image.open(mask_file)
        mask_val = numpy.array(img, dtype=numpy.uint8)
        mask_nodes = mask_to_nodes[mask_attr]
        for n in mask_nodes:
            lon = node_coords[5 * n + 0]
            lat = node_coords[5 * n + 1]
            (s, t) = GEO.st_coord(lat, lon, *mask_attr, tile.mask_zl, None)
            pixx = int(s * 4095)
            pixy = int((1-t) * 4095)
            node_bathy[n] = mask_val[pixy, pixx]

    return node_bathy
            








