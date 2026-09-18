import os
import pickle
import shutil
import io
import subprocess
from math import floor, ceil
import array
import numpy
from PIL import Image, ImageDraw
from collections import defaultdict
import struct
import hashlib
import time
import tempfile
import O4_File_Names as FNAMES
import O4_Geo_Utils as GEO
import O4_Mask_Utils as MASK
import O4_UI_Utils as UI
import O4_Overlay_Utils as OVL
import O4_Mesh_Utils as MESH
import O4_Bathymetry as BATHY
import O4_DSF_Budget as DSF_BUDGET
import O4_Imagery_Utils as IMG

quad_init_level = 3
quad_capacity_high = 50000
quad_capacity_low = 35000

# For Laminar test suite
use_test_texture = False


def _texture_contract_matches(tile, texture_attributes, has_alpha):
    """Check an existing DDS against the current output policy."""
    til_x_left, til_y_top, zoomlevel, provider_code = texture_attributes
    expected_dimensions = IMG.expected_texture_dimensions(
        tile, til_x_left, til_y_top, zoomlevel, provider_code
    )
    configured_format = getattr(tile, "dds_format", "BC3")
    expected_format = IMG.resolve_dds_format(configured_format, has_alpha)
    target_path = os.path.join(
        tile.build_dir,
        "textures",
        FNAMES.dds_file_name_from_attributes(*texture_attributes),
    )
    valid, _ = IMG.validate_dds_file(
        target_path,
        expected_format=expected_format,
        expected_dimensions=expected_dimensions,
        require_mipmaps=True,
    )
    return valid


def _masked_dds_requires_alpha(tile, mask_im):
    """Return the alpha contract produced by the current mask policy.

    Overlay classification controls the terrain type, not whether
    ``convert_texture`` imprints a mask into the DDS. Keep those decisions
    separate so XP11 + bathy masked textures are checked against their actual
    BC3 output. Explicit BC1 retains the historical mask promotion to BC3.
    """
    if not getattr(tile, "imprint_masks_to_dds", False):
        return False
    if mask_im is None or mask_im is False:
        return False

    configured_format = str(getattr(tile, "dds_format", "BC3")).strip().upper()
    if configured_format == "BC1":
        return True
    try:
        return mask_im.convert("L").getextrema()[0] < 255
    except Exception:
        # A mask that cannot be inspected must not allow a potentially
        # alpha-bearing DDS to be incorrectly reused.
        return True


def _texture_contract_has_alpha(tile, texture_attributes):
    """Return the alpha contract for any texture classification path.

    Land and inland-water triangles can share a texture with a masked water
    triangle.  ``convert_texture`` then imprints the same mask into the DDS,
    so their reuse check must inspect that mask instead of assuming BC1.
    """
    if not getattr(tile, "imprint_masks_to_dds", False):
        return False
    mask_im = MASK.needs_mask(tile, *texture_attributes)
    return _masked_dds_requires_alpha(tile, mask_im)


################################################################################
def float2qquad(x):
    if x >= 1:
        return "111111111111111111111111"
    return numpy.binary_repr(int(16777216 * x)).zfill(24)  # 2**24 == 16777216


def numpy_wgs84_to_orthogrid(lat, lon, zoomlevel):
    ratio_x = lon / 180
    ratio_y = numpy.log(numpy.tan((90 + lat) * numpy.pi / 360)) / numpy.pi
    mult = 2 ** (zoomlevel - 1)
    til_x = numpy.floor((ratio_x + 1) * mult).astype(numpy.int32)
    til_y = numpy.floor((1 - ratio_y) * mult).astype(numpy.int32)
    return til_x, til_y


def _mesh_orthogrid_bounds(tile):
    """Return tile bounds in the same unaligned grid as mesh triangles."""
    til_x_min, til_y_min = numpy_wgs84_to_orthogrid(
        tile.lat + 1, tile.lon, tile.mesh_zl
    )
    til_x_max, til_y_max = numpy_wgs84_to_orthogrid(
        tile.lat, tile.lon + 1, tile.mesh_zl
    )
    return (
        int(til_x_min),
        int(til_y_min),
        int(til_x_max),
        int(til_y_max),
    )


def _validate_mesh_orthogrid_indices(til_xs, til_ys, bounds):
    """Reject triangle indices outside the tile instead of hiding them."""
    til_x_min, til_y_min, til_x_max, til_y_max = bounds
    til_xs = numpy.asarray(til_xs)
    til_ys = numpy.asarray(til_ys)
    outside = (
        (til_xs < til_x_min)
        | (til_xs > til_x_max)
        | (til_ys < til_y_min)
        | (til_ys > til_y_max)
    )
    if not numpy.any(outside):
        return

    outside_xs = til_xs[outside]
    outside_ys = til_ys[outside]
    raise ValueError(
        "Triangle orthogrid indices outside tile bounds: "
        "{} points, x=[{}, {}], y=[{}, {}], "
        "expected x=[{}, {}], y=[{}, {}]".format(
            int(numpy.count_nonzero(outside)),
            int(outside_xs.min()),
            int(outside_xs.max()),
            int(outside_ys.min()),
            int(outside_ys.max()),
            til_x_min,
            til_x_max,
            til_y_min,
            til_y_max,
        )
    )


def numpy_st_coord(lat, lon, tex_x, tex_y, zoomlevel):
    ratio_x = lon / 180
    ratio_y = numpy.log(numpy.tan((90 + lat) * numpy.pi / 360)) / numpy.pi
    mult = 2.0 ** (zoomlevel - 5)
    s = (ratio_x + 1) * mult - (tex_x // 16)
    t = 1 - ((1 - ratio_y) * mult - tex_y // 16)
    s = numpy.clip(s, 0, 1)
    t = numpy.clip(t, 0, 1)
    return s, t


def _integer_point_codes(node_coords, tile_lon, tile_lat):
    """Return the 24-bit tile-relative coordinates used by DSF pools."""
    xs = numpy.asarray(node_coords[0::5], dtype=numpy.float64) - tile_lon
    ys = numpy.asarray(node_coords[1::5], dtype=numpy.float64) - tile_lat
    scale = 1 << 24

    def encode(values):
        scaled = numpy.trunc(values * scale).astype(numpy.int64)
        return numpy.where(values >= 1, scale - 1, scaled).astype(numpy.uint32)

    return encode(xs), encode(ys)


def _build_integer_point_pools(
    x_codes,
    y_codes,
    bucket_size,
    init_level=quad_init_level,
):
    """Build the DSF point partition without binary-string dictionaries.

    The initial cells and child order are deterministic.  A cell is split
    only when it exceeds the same capacity used by ``QuadTree``.
    """
    x_codes = numpy.asarray(x_codes, dtype=numpy.uint32)
    y_codes = numpy.asarray(y_codes, dtype=numpy.uint32)
    if x_codes.shape != y_codes.shape:
        raise ValueError("point coordinate arrays must have the same shape")
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")

    node_count = x_codes.size
    pool_id_by_node = numpy.empty(node_count, dtype=numpy.int32)
    pool_nodes = []
    pool_prefixes = []
    init_shift = 24 - init_level
    root_x = x_codes >> init_shift
    root_y = y_codes >> init_shift
    root_keys = sorted(set(zip(root_x.tolist(), root_y.tolist())))

    for root_prefix_x, root_prefix_y in root_keys:
        root_indices = numpy.flatnonzero(
            (root_x == root_prefix_x) & (root_y == root_prefix_y)
        ).astype(numpy.int32, copy=False)
        pending = [(init_level, root_prefix_x, root_prefix_y, root_indices)]
        while pending:
            level, prefix_x, prefix_y, indices = pending.pop()
            if indices.size <= bucket_size:
                pool_id = len(pool_nodes)
                ordered = numpy.sort(indices)
                pool_nodes.append(ordered)
                pool_prefixes.append((level, int(prefix_x), int(prefix_y)))
                pool_id_by_node[ordered] = pool_id
                continue

            if level >= 24:
                raise ValueError(
                    "cannot split a DSF point pool beyond the 24-bit coordinate"
                )
            child_level = level + 1
            child_shift = 24 - child_level
            child_x = ((x_codes[indices] >> child_shift) & 1).astype(numpy.uint32)
            child_y = ((y_codes[indices] >> child_shift) & 1).astype(numpy.uint32)
            children = []
            for bit_x, bit_y in ((0, 0), (0, 1), (1, 0), (1, 1)):
                child_indices = indices[(child_x == bit_x) & (child_y == bit_y)]
                if child_indices.size:
                    children.append(
                        (
                            child_level,
                            (int(prefix_x) << 1) | bit_x,
                            (int(prefix_y) << 1) | bit_y,
                            child_indices,
                        )
                    )
            pending.extend(reversed(children))

    return pool_id_by_node, tuple(pool_nodes), tuple(pool_prefixes)


def _pool_local_coordinates(codes, level):
    """Extract the same 16-bit slice as ``bits[level:level + 16]``."""
    codes = numpy.asarray(codes, dtype=numpy.uint32)
    width = min(16, max(0, 24 - level))
    if width == 0:
        return numpy.zeros(codes.shape, dtype=numpy.uint16)
    shift = max(0, 24 - (level + 16))
    mask = (1 << width) - 1
    return ((codes >> shift) & mask).astype(numpy.uint16)


def _precompute_triangle_uvs(node_coords, oriented_tri_nodes, tri_tex_attr):
    """Return rounded uint16 UV pairs for the oriented triangle vertices."""
    oriented_tri_nodes = numpy.asarray(oriented_tri_nodes, dtype=numpy.intp)
    attributes = tuple(tri_tex_attr)
    tex_x = numpy.asarray([attribute[0] for attribute in attributes])
    tex_y = numpy.asarray([attribute[1] for attribute in attributes])
    zoomlevel = numpy.asarray([attribute[2] for attribute in attributes])
    lons = numpy.asarray(node_coords[0::5])
    lats = numpy.asarray(node_coords[1::5])
    vertex_lons = lons[oriented_tri_nodes]
    vertex_lats = lats[oriented_tri_nodes]
    s, t = numpy_st_coord(
        vertex_lats,
        vertex_lons,
        tex_x[:, numpy.newaxis],
        tex_y[:, numpy.newaxis],
        zoomlevel[:, numpy.newaxis],
    )
    uv = numpy.empty(oriented_tri_nodes.shape + (2,), dtype=numpy.uint16)
    uv[..., 0] = numpy.rint(numpy.clip(s, 0, 1) * 65535).astype(numpy.uint16)
    uv[..., 1] = numpy.rint(numpy.clip(t, 0, 1) * 65535).astype(numpy.uint16)
    return uv


def _plan_texture_requirements(tile, tri_tex_attr, tri_types):
    """Compute texture and mask decisions once per texture attribute."""
    flags = {}
    for attributes, tri_type in zip(tri_tex_attr, tri_types):
        state = flags.setdefault(
            attributes,
            {"has_water": False, "has_non_water": False},
        )
        if int(tri_type) == 2:
            state["has_water"] = True
        else:
            state["has_non_water"] = True

    plans = {}
    for attributes, state in flags.items():
        needs_mask = state["has_water"] or getattr(
            tile, "imprint_masks_to_dds", False
        )
        mask_im = MASK.needs_mask(tile, *attributes) if needs_mask else False
        mask_present = bool(mask_im)
        mask_alpha = _masked_dds_requires_alpha(tile, mask_im)
        texture_file_name = FNAMES.dds_file_name_from_attributes(*attributes)
        needs_texture = state["has_non_water"] or mask_present
        rebuild = False
        if needs_texture:
            target_tex = os.path.join(tile.build_dir, "textures", texture_file_name)
            rebuild = not os.path.isfile(target_tex)
        if needs_texture and not rebuild:
            rebuild = not _texture_contract_matches(
                tile,
                attributes,
                has_alpha=mask_alpha,
            )
        if needs_texture and mask_present:
            target_mask = MASK.mask_name_for_texture(tile, *attributes)
            if os.path.isfile(target_mask) and os.path.isfile(target_tex):
                rebuild = rebuild or (
                    os.path.getmtime(target_tex) < os.path.getmtime(target_mask)
                )
        if state["has_water"]:
            mask_target = os.path.join(
                tile.build_dir,
                "textures",
                FNAMES.mask_file(*attributes),
            )
            if mask_present:
                if rebuild or not getattr(tile, "imprint_masks_to_dds", False):
                    mask_im.save(mask_target)
            else:
                try:
                    os.remove(mask_target)
                except OSError:
                    pass
        plans[attributes] = {
            "mask_present": mask_present,
            "mask_alpha": mask_alpha,
            "texture_file_name": texture_file_name,
            "rebuild": rebuild,
        }
    return plans


################################################################################

################################################################################
class QuadTree(dict):
    class Bucket(dict):
        def __init__(self):
            self["size"] = 0
            self["idx_nodes"] = set()

    def __init__(self, level, bucket_size):
        self.bucket_size = bucket_size
        if level == 0:
            self[("", "")] = self.Bucket()
        else:
            for i in range(2 ** level):
                for j in range(2 ** level):
                    key = (
                        numpy.binary_repr(i).zfill(level),
                        numpy.binary_repr(j).zfill(level),
                    )
                    self[key] = self.Bucket()
        self.nodes = {}
        self.levels = {}
        self.last_node = 0

    def split_bucket(self, key):
        level = len(key[0]) + 1
        self[(key[0] + "0", key[1] + "0")] = self.Bucket()
        self[(key[0] + "0", key[1] + "1")] = self.Bucket()
        self[(key[0] + "1", key[1] + "0")] = self.Bucket()
        self[(key[0] + "1", key[1] + "1")] = self.Bucket()
        for idx in self[key]["idx_nodes"]:
            new_key = (self.nodes[idx][0][:level], self.nodes[idx][1][:level])
            self[new_key]["idx_nodes"].add(idx)
            self[new_key]["size"] += 1
            self.levels[idx] += 1
        del self[key]

    def insert(self, bx, by, level):
        while True:
            key = (bx[:level], by[:level])
            if key in self:
                break
            level += 1
        if self[key]["size"] < self.bucket_size:
            self[key]["idx_nodes"].add(self.last_node)
            self[key]["size"] += 1
            self.nodes[self.last_node] = (bx, by)
            self.levels[self.last_node] = level
            self.last_node += 1
        else:
            self.split_bucket(key)
            self.insert(bx, by, level + 1)

    def clean(self):
        for key in list(self.keys()):
            if not self[key]["size"]:
                del self[key]

    def statistics(self):
        lengths = numpy.array([self[key]["size"] for key in self])
        depths = numpy.array([len(key[0]) for key in self])
        UI.vprint(2, "     Number of buckets:", len(lengths))
        UI.vprint(
            2,
            "     Average depth:",
            depths.mean(),
            ", Average bucket size:",
            lengths.mean(),
        )
        UI.vprint(2, "     Largest depth:", numpy.max(depths))


################################################################################

################################################################################
def zone_list_to_ortho_dico(tile):
    # tile.zone_list is a list of 3-uples of the form
    # ([(lat0,lat0), ... ,(latN,lonN)], zoomlevel, provider_code)
    # where higher lines have priority over lower ones.
    masks_im = Image.new("L", (4096, 4096), "black")
    masks_draw = ImageDraw.Draw(masks_im)
    airport_array = numpy.zeros((4096, 4096), dtype=numpy.bool_)
    airport_highres_texture_keys = set()
    if tile.cover_airports_with_highres in ("True", "ICAO"):
        UI.vprint(1, "-> Checking airport locations for upgraded zoomlevel.")
        try:
            f = open(FNAMES.apt_file(tile), "rb")
            dico_airports = pickle.load(f)
            f.close()
        except:
            UI.vprint(
                1,
                "   WARNING: File",
                FNAMES.apt_file(tile),
                "is missing (erased after Step 1?), cannot check airport info ",
                "for upgraded zoomlevel.",
            )
            dico_airports = {}
        if tile.cover_airports_with_highres == "ICAO":
            airports_list = [
                airport
                for airport in dico_airports
                if dico_airports[airport]["key_type"] == "icao"
            ]
        else:
            airports_list = dico_airports.keys()
        for airport in airports_list:
            (xmin, ymin, xmax, ymax) = dico_airports[airport]["boundary"].bounds
            # extension
            xmin -= 1000 * tile.cover_extent * GEO.m_to_lon(tile.lat)
            xmax += 1000 * tile.cover_extent * GEO.m_to_lon(tile.lat)
            ymax += 1000 * tile.cover_extent * GEO.m_to_lat
            ymin -= 1000 * tile.cover_extent * GEO.m_to_lat
            # round off to texture boundaries at tile.cover_zl zoomlevel
            (til_x_left, til_y_top) = GEO.wgs84_to_orthogrid(
                ymax + tile.lat, xmin + tile.lon, tile.cover_zl
            )
            (ymax, xmin) = GEO.gtile_to_wgs84(
                til_x_left, til_y_top, tile.cover_zl
            )
            ymax -= tile.lat
            xmin -= tile.lon
            (til_x_left2, til_y_top2) = GEO.wgs84_to_orthogrid(
                ymin + tile.lat, xmax + tile.lon, tile.cover_zl
            )
            (ymin, xmax) = GEO.gtile_to_wgs84(
                til_x_left2 + 16, til_y_top2 + 16, tile.cover_zl
            )
            ymin -= tile.lat
            xmax -= tile.lon
            xmin = max(0, xmin)
            xmax = min(1, xmax)
            ymin = max(0, ymin)
            ymax = min(1, ymax)
            # mark to airport_array
            colmin = round(xmin * 4095)
            colmax = round(xmax * 4095)
            rowmax = round((1 - ymin) * 4095)
            rowmin = round((1 - ymax) * 4095)
            airport_array[rowmin : rowmax + 1, colmin : colmax + 1] = 1
    dico_customzl = {}
    dico_tmp = {}
    til_x_min, til_y_min, til_x_max, til_y_max = _mesh_orthogrid_bounds(tile)
    i = 1
    base_zone = (
        [
            tile.lat,
            tile.lon,
            tile.lat,
            tile.lon + 1,
            tile.lat + 1,
            tile.lon + 1,
            tile.lat + 1,
            tile.lon,
            tile.lat,
            tile.lon,
        ],
        tile.default_zl,
        tile.default_website,
    )
    for region in [base_zone] + tile.zone_list[::-1]:
        dico_tmp[i] = (region[1], region[2])
        pol = [
            (round((x - tile.lon) * 4095), round((tile.lat + 1 - y) * 4095))
            for (x, y) in zip(region[0][1::2], region[0][::2])
        ]
        masks_draw.polygon(pol, fill=i)
        i += 1
    # Resolution of the mesh_zl grid in terms of the 4096x4096nd airport_array
    step_x = 4096 / (til_x_max - til_x_min + 1)
    step_y = 4096 / (til_y_max - til_y_min + 1)

    for til_x in range(til_x_min, til_x_max + 1):
        for til_y in range(til_y_min, til_y_max + 1):
            (latp, lonp) = GEO.gtile_to_wgs84(
                til_x + 0.5, til_y + 0.5, tile.mesh_zl
            )
            lonp = max(min(lonp, tile.lon + 1), tile.lon)
            latp = max(min(latp, tile.lat + 1), tile.lat)
            x = round((lonp - tile.lon) * 4095)
            y = round((tile.lat + 1 - latp) * 4095)
            (zoomlevel, provider_code) = dico_tmp[masks_im.getpixel((x, y))]
            
            # Check a block in airport_array instead of a single pixel
            x0 = int(max(0, x - step_x / 2))
            x1 = int(min(4095, x + step_x / 2))
            y0 = int(max(0, y - step_y / 2))
            y1 = int(min(4095, y + step_y / 2))
            
            airport_upgrade = bool(
                airport_array[y0 : y1 + 1, x0 : x1 + 1].any()
            )
            if airport_upgrade:
                zoomlevel = max(zoomlevel, tile.cover_zl)
                
            til_x_text = 16 * (
                int(til_x / 2 ** (tile.mesh_zl - zoomlevel)) // 16
            )
            til_y_text = 16 * (
                int(til_y / 2 ** (tile.mesh_zl - zoomlevel)) // 16
            )
            dico_customzl[(til_x, til_y)] = (
                til_x_text,
                til_y_text,
                zoomlevel,
                provider_code,
            )
            if airport_upgrade:
                airport_highres_texture_keys.add(
                    (til_x_text, til_y_text, zoomlevel, str(provider_code))
                )
    if tile.cover_airports_with_highres == "Existing":
        # what we find in the texture folder of the existing tile
        for f in sorted(os.listdir(os.path.join(tile.build_dir, "textures"))):
            if f[-4:] != ".dds":
                continue
            items = f.split("_")
            (til_y_text, til_x_text) = [int(x) for x in items[:2]]
            zoomlevel = int(items[-1][-6:-4])
            provider_code = "_".join(items[2:])[:-6]
            for til_x in range(
                til_x_text * 2 ** (tile.mesh_zl - zoomlevel),
                (til_x_text + 16) * 2 ** (tile.mesh_zl - zoomlevel),
            ):
                for til_y in range(
                    til_y_text * 2 ** (tile.mesh_zl - zoomlevel),
                    (til_y_text + 16) * 2 ** (tile.mesh_zl - zoomlevel),
                ):
                    if ((til_x, til_y) not in dico_customzl) or dico_customzl[
                        (til_x, til_y)
                    ][2] <= zoomlevel:
                        dico_customzl[(til_x, til_y)] = (
                            til_x_text,
                            til_y_text,
                            zoomlevel,
                            provider_code,
                        )
                        airport_highres_texture_keys.add(
                            (til_x_text, til_y_text, zoomlevel, str(provider_code))
                        )

    tile.airport_highres_texture_keys = frozenset(airport_highres_texture_keys)
    return dico_customzl
################################################################################

################################################################################
def create_terrain_file(
    tile,
    texture_file_name,
    til_x_left,
    til_y_top,
    zoomlevel,
    provider_code,
    tri_type,
    is_overlay,
):
    # Force land for high-res airport tiles to avoid misidentification as water


    if not os.path.exists(os.path.join(tile.build_dir, "terrain")):
        os.makedirs(os.path.join(tile.build_dir, "terrain"))

    suffix = "_water" if tri_type == 1 else "_sea" if tri_type == 2 else ""
    if is_overlay:
        suffix += "_overlay"
    ter_file_name = texture_file_name[:-4] + suffix + ".ter"

    if use_test_texture:
        texture_file_name = "test_texture.dds"

    with open(os.path.join(tile.build_dir, "terrain", ter_file_name), "w") as f:

        f.write("A\n800\nTERRAIN\n\n")

        [lat_med, lon_med] = GEO.gtile_to_wgs84(
            til_x_left + 8, til_y_top + 8, zoomlevel
        )
        texture_approx_size = int(
            GEO.webmercator_pixel_size(lat_med, zoomlevel) * 4096
        )
        f.write(
            "LOAD_CENTER "
            + "{:.5f}".format(lat_med)
            + " "
            + "{:.5f}".format(lon_med)
            + " "
            + str(texture_approx_size)
            + " 4096\n"
        )

        f.write("BASE_TEX_NOWRAP ../textures/" + texture_file_name + "\n")

        if int(zoomlevel) >= 18 and tri_type == 0:
            f.write("NO_ALPHA\n")
            return ter_file_name

        if tri_type in (1, 2) and (not is_overlay):  # XP12 water
            #pass
            f.write("WATER_COLOR_MASK\n")
        elif (tri_type == 1) or (
            (tri_type == 2) and (is_overlay == "ratio_water")
        ):  # constant transparency level
            f.write("BORDER_TEX ../textures/water_transition.png\n")
            if not os.path.exists(
                os.path.join(tile.build_dir, "textures", "water_transition.png")
            ):
                shutil.copy(
                    os.path.join(FNAMES.Utils_dir, "water_transition.png"),
                    os.path.join(tile.build_dir, "textures"),
                )
        elif (tri_type == 2) and (not tile.imprint_masks_to_dds):  
            # border_tex mask
            f.write(
                "LOAD_CENTER_BORDER "
                + "{:.5f}".format(lat_med)
                + " "
                + "{:.5f}".format(lon_med)
                + " "
                + str(texture_approx_size)
                + " "
                + str(4096 // 2 ** (zoomlevel - tile.mask_zl))
                + "\n"
            )
            f.write(
                "BORDER_TEX ../textures/"
                + FNAMES.mask_file(
                    til_x_left, til_y_top, zoomlevel, provider_code
                )
                + "\n"
            )

        # Hack/TODO
        # Should we use decals on ocean floor ? 
        #if (not tri_type) and (tile.use_decal_on_terrain):
        if (tri_type != 1) and (tile.use_decal_on_terrain):
            f.write("DECAL_LIB lib/g10/decals/maquify_2_green_key.dcl\n")

        if tri_type in (1, 2):
            f.write("WET\n")
        else:
            f.write("NO_ALPHA\n")

        if (tri_type in (1, 2)) or (not tile.terrain_casts_shadows):
            f.write("NO_SHADOW\n")

        return ter_file_name


################################################################################

################################################################################
def extract_elevation_and_bathymetry_data(lat, lon):
    UI.vprint(1, "     Extracting some rasters from X-Plane's Global Scenery")
    effective_config = getattr(UI, "active_effective_config", {})
    effective_overlay_src = effective_config.get(
        "custom_overlay_src", OVL.custom_overlay_src
    )
    global_scenery_dsf = FNAMES.resolve_global_scenery_dsf(
        effective_overlay_src, lat, lon
    )
    if global_scenery_dsf is None:
        scenery_candidates = FNAMES.global_scenery_dsf_candidates(
            effective_overlay_src, lat, lon
        )
        if not scenery_candidates:
            UI.exit_message_and_bottom_line(
                "   ERROR: Global Scenery DSF was not found below ",
                effective_overlay_src or "(empty path)",
                ". Expected Earth nav data/" + FNAMES.long_latlon(lat, lon) + ".dsf.",
            )
        else:
            UI.exit_message_and_bottom_line(
                "   ERROR: Multiple Global Scenery DSFs matched:",
                *scenery_candidates,
            )
        return None

    work_dir = None
    tmp_file = None
    archive_path = None
    try:
        os.makedirs(FNAMES.Tmp_dir, exist_ok=True)
        work_dir = tempfile.mkdtemp(
            prefix="." + FNAMES.short_latlon(lat, lon) + "-dsf-",
            dir=FNAMES.Tmp_dir,
        )
        tmp_file = os.path.join(
            work_dir, FNAMES.short_latlon(lat, lon) + ".dsf"
        )
        archive_path = tmp_file + ".7z"
    except OSError as error:
        UI.exit_message_and_bottom_line(
            "     ERROR: could not create Global Scenery DSF temporary workspace:",
            error,
        )
        return None
    UI.vprint(2, "     Making a copy of the Global Scenery DSF in tmp dir")
    try:
        try:
            shutil.copy(global_scenery_dsf, tmp_file)
        except Exception as error:
            UI.exit_message_and_bottom_line(
                "     ERROR: could not copy Global Scenery DSF:",
                error,
            )
            return None

        with open(tmp_file, "rb") as source:
            magic = source.read(2)
        if magic == b"7z":
            UI.vprint(2, "     The original DSF is a 7z archive, uncompressing...")
            os.replace(tmp_file, archive_path)
            unzip_result = subprocess.run(
                [OVL.unzip_cmd.strip(), "e", "-y", "-o" + work_dir, archive_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if unzip_result.stdout:
                for line in unzip_result.stdout.splitlines():
                    UI.vprint(2, "     " + line)
            if unzip_result.returncode != 0 or not os.path.isfile(tmp_file):
                UI.logprint(
                    "Global Scenery DSF extraction failed:",
                    "returncode=",
                    unzip_result.returncode,
                    "output_exists=",
                    os.path.isfile(tmp_file),
                    "output=",
                    repr((unzip_result.stdout or "")[-2000:]),
                )
                UI.exit_message_and_bottom_line(
                    "     ERROR: could not uncompress Global Scenery DSF."
                )
                return None

        file_len = os.path.getsize(tmp_file)
        if file_len < 28:
            raise ValueError("DSF file is shorter than its header and checksum")

        with open(tmp_file, "rb") as source:
            body = source.read(file_len - 16)
            checksum = source.read(16)
            if len(checksum) != 16 or hashlib.md5(body).digest() != checksum:
                raise ValueError("DSF checksum is invalid")
            source.seek(0)
            if source.read(8) != b"XPLNEDSF":
                raise ValueError("DSF file type cookie is invalid")
            source.read(4)  # format number
            atoms_len = file_len - 12 - 16  # header and MD5 checksum
            atoms_consumed = 0
            bDEMN = None
            bDEMS = b""

            while atoms_consumed < atoms_len:
                if atoms_len - atoms_consumed < 8:
                    raise ValueError("truncated DSF atom header")
                atom_hdr = source.read(4).decode("ascii")
                atom_len = struct.unpack("<I", source.read(4))[0]
                if atom_len < 8 or atom_len > atoms_len - atoms_consumed:
                    raise ValueError("invalid DSF atom length")
                atom_data = source.read(atom_len - 8)
                if len(atom_data) != atom_len - 8:
                    raise ValueError("truncated DSF atom")

                if atom_hdr == "SMED":
                    dem_parts = []
                    elevation_data = None
                    large_subatom_index = 0
                    consumed = 0
                    while consumed < len(atom_data):
                        if len(atom_data) - consumed < 8:
                            raise ValueError("truncated SMED sub-atom header")
                        sub_hdr = atom_data[consumed : consumed + 4]
                        sub_len = struct.unpack(
                            "<I", atom_data[consumed + 4 : consumed + 8]
                        )[0]
                        if sub_len < 8 or sub_len > len(atom_data) - consumed:
                            raise ValueError("invalid SMED sub-atom length")
                        sub_data = atom_data[consumed + 8 : consumed + sub_len]
                        if sub_len > 100:
                            large_subatom_index += 1
                            if large_subatom_index == 1:
                                elevation_data = sub_data
                            elif large_subatom_index == 2 and elevation_data is not None:
                                bathy = numpy.frombuffer(sub_data, dtype=numpy.int16)
                                safe = numpy.frombuffer(elevation_data, dtype=numpy.int16) - 2
                                if bathy.size != safe.size:
                                    raise ValueError("DEM and bathymetry raster sizes differ")
                                sub_data = bytes(numpy.minimum(bathy, safe))
                        dem_parts.append(sub_hdr + struct.pack("<I", sub_len) + sub_data)
                        consumed += sub_len
                    bDEMS = b"".join(dem_parts)
                elif atom_hdr == "NFED":
                    consumed = 0
                    while consumed < len(atom_data):
                        if len(atom_data) - consumed < 8:
                            raise ValueError("truncated NFED sub-atom header")
                        sub_hdr = atom_data[consumed : consumed + 4]
                        sub_len = struct.unpack(
                            "<I", atom_data[consumed + 4 : consumed + 8]
                        )[0]
                        if sub_len < 8 or sub_len > len(atom_data) - consumed:
                            raise ValueError("invalid NFED sub-atom length")
                        if sub_hdr == b"NMED":
                            bDEMN = atom_data[consumed + 8 : consumed + sub_len]
                        consumed += sub_len
                atoms_consumed += atom_len

        if bDEMN is None:
            raise ValueError("Global Scenery DSF has no NMED raster")
        return (bDEMN, bDEMS)
    except (OSError, UnicodeDecodeError, struct.error, ValueError) as error:
        UI.exit_message_and_bottom_line(
            "     ERROR: Could not extract Global Scenery elevation data:",
            error,
        )
        return None
    finally:
        if work_dir is not None:
            try:
                shutil.rmtree(work_dir)
            except OSError:
                pass


################################################################################

################################################################################
def build_dsf(tile, download_queue):

    dsf_tmp = os.path.join(
        tile.build_dir,
        "Earth nav data",
        FNAMES.long_latlon(tile.lat, tile.lon) + ".dsf.tmp",
    )
    result = 0
    try:
        result = _build_dsf(tile, download_queue)
        return result
    finally:
        if result != 1:
            try:
                os.remove(dsf_tmp)
            except OSError:
                pass


def _build_dsf(tile, download_queue):

    tile.last_dsf_metrics = None

    
    dico_customzl = zone_list_to_ortho_dico(tile)

    # 1 Read mesh file
    UI.vprint(1, "-> Reading mesh file")
    mesh_filename = FNAMES.mesh_file(tile.build_dir, tile.lat, tile.lon)
    (mesh_version, nbr_nodes, node_coords, nbr_tris, tri_idx, tri_types) \
            = MESH.read_mesh_file(mesh_filename)

    # 2 Remap tri_types in (0,1,2)
    has_water = 7 if (mesh_version >= 1.3) else 3
    tri_types &= has_water
    tri_types[tri_types > 1] = 2
    if tile.use_masks_for_inland:
        tri_types[tri_types == 1] = 2

    # 3 Recut water tris for XP12
    UI.vprint(1, "-> Adapting water triangles to XP12 requirements")
    (nbr_nodes, node_coords, node_types, node_is_coast, nbr_tris, tri_idx, 
        tri_types) = BATHY.recut_water_tris(node_coords, tri_idx, tri_types)

    # 4 Compute bathymetry depth ratio bounds based on masks
    UI.vprint(1, "-> Computing bathymetry depth ratio bounds based on distance masks")
    node_bathy = BATHY.compute_depth_ratio_bounds_from_masks(
                            nbr_nodes, node_coords, node_types, tile)
    
    UI.vprint(1, "-> Computing point pools and texture requirements")
    dsf_timings = {}
    phase_started = time.perf_counter()
    
    # 5.1 Vectorized triangle attributes
    lons = node_coords[0::5]
    lats = node_coords[1::5]
    tri_idx_reshaped = tri_idx[:3*nbr_tris].reshape((nbr_tris, 3))
    tri_lons = numpy.mean(lons[tri_idx_reshaped], axis=1)
    tri_lats = numpy.mean(lats[tri_idx_reshaped], axis=1)
    til_xs, til_ys = numpy_wgs84_to_orthogrid(tri_lats, tri_lons, tile.mesh_zl)

    # Keep the lookup grid in the same coordinate system as the triangle indices.
    til_x_min, til_y_min, til_x_max, til_y_max = _mesh_orthogrid_bounds(tile)
    _validate_mesh_orthogrid_indices(
        til_xs,
        til_ys,
        (til_x_min, til_y_min, til_x_max, til_y_max),
    )
    
    # Vectorized custom ZL lookup using 2D NumPy array for huge speedup
    width = til_x_max - til_x_min + 1
    height = til_y_max - til_y_min + 1
    customzl_arr = numpy.empty((width, height), dtype=object)
    
    # Pre-fill with a safe default value to prevent any missing key errors
    default_val = (16 * (til_x_min // 16), 16 * (til_y_min // 16), tile.default_zl, tile.default_website)
    customzl_arr.fill(default_val)
    
    for (tx, ty), val in dico_customzl.items():
        idx_x = tx - til_x_min
        idx_y = ty - til_y_min
        if 0 <= idx_x < width and 0 <= idx_y < height:
            customzl_arr[idx_x, idx_y] = val
            
    idx_xs = til_xs - til_x_min
    idx_ys = til_ys - til_y_min
    tri_tex_attr = customzl_arr[idx_xs, idx_ys].tolist()
    dsf_timings["triangle_attributes_ms"] = (
        time.perf_counter() - phase_started
    ) * 1000.0
    
    # 5 Compute point pools using integer coordinates.
    if (tile.use_masks_for_inland):
        quad_capacity = quad_capacity_low
    else:
        quad_capacity = quad_capacity_high
    phase_started = time.perf_counter()
    x_codes, y_codes = _integer_point_codes(node_coords, tile.lon, tile.lat)
    (
        idx_node_to_idx_pool,
        pool_nodes,
        pool_prefixes,
    ) = _build_integer_point_pools(
        x_codes,
        y_codes,
        quad_capacity,
        quad_init_level,
    )
    pool_sizes = numpy.asarray([nodes.size for nodes in pool_nodes], dtype=numpy.int32)
    pool_levels = numpy.asarray(
        [prefix[0] for prefix in pool_prefixes], dtype=numpy.int16
    )
    UI.vprint(2, "     Number of buckets:", len(pool_sizes))
    UI.vprint(
        2,
        "     Average depth:",
        pool_levels.mean() if pool_levels.size else 0,
        ", Average bucket size:",
        pool_sizes.mean() if pool_sizes.size else 0,
    )
    UI.vprint(
        2,
        "     Largest depth:",
        int(pool_levels.max()) if pool_levels.size else 0,
    )
    dsf_timings["point_pools_ms"] = (time.perf_counter() - phase_started) * 1000.0

    # 6 Compute pool params
    phase_started = time.perf_counter()
    pool_nbr = len(pool_nodes)
    pool_param = {}
    node_icoords = numpy.zeros(5 * nbr_nodes, dtype = numpy.uint16)
    for idx_pool, (plist_arr, prefix) in enumerate(zip(pool_nodes, pool_prefixes)):
        level, prefix_x, prefix_y = prefix
        idx_0 = 5 * plist_arr
        idx_1 = idx_0 + 1
        idx_2 = idx_0 + 2

        node_icoords[idx_0] = _pool_local_coordinates(x_codes[plist_arr], level)
        node_icoords[idx_1] = _pool_local_coordinates(y_codes[plist_arr], level)
        altitudes = node_coords[idx_2]
        altmin = floor(altitudes.min())
        altmax = ceil(altitudes.max())
        if altmax - altmin < 770:
            scale_z = 771  # 65535=771*85
            inv_stp = 85
        elif altmax - altmin < 1284:
            scale_z = 1285  # 65535=1285*51
            inv_stp = 51
        elif altmax - altmin < 4368:
            scale_z = 4369  # 65535=4369*15
            inv_stp = 15
        else:
            scale_z = 13107  # 65535=13107*5
            inv_stp = 5
        scal_x = scal_y = 2 ** (-level)
        node_icoords[idx_2] = numpy.round(
            (altitudes - altmin) * inv_stp
        )
        pool_param[idx_pool] = (
            scal_x,
            tile.lon + prefix_x * scal_x,
            scal_y,
            tile.lat + prefix_y * scal_y,
            scale_z,
            altmin,
            2,
            -1,
            2,
            -1,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
        )
    node_icoords[3::5] = numpy.round(
        (1 + tile.normal_map_strength * node_coords[3::5]) / 2 * 65535
    )
    node_icoords[4::5] = numpy.round(
        (1 - tile.normal_map_strength * node_coords[4::5]) / 2 * 65535
    )
    node_icoords = array.array("H", node_icoords)
    dsf_timings["pool_parameters_ms"] = (
        time.perf_counter() - phase_started
    ) * 1000.0

    phase_started = time.perf_counter()
    oriented_tri_nodes = tri_idx_reshaped[:, (0, 2, 1)]
    tri_pool_ids = idx_node_to_idx_pool[oriented_tri_nodes]
    triangle_uv = _precompute_triangle_uvs(
        node_coords,
        oriented_tri_nodes,
        tri_tex_attr,
    )
    texture_requirements = _plan_texture_requirements(
        tile,
        tri_tex_attr,
        tri_types,
    )
    dsf_timings["texture_plan_and_uv_ms"] = (
        time.perf_counter() - phase_started
    ) * 1000.0

    
    
    ##########################
    dico_terrains = {}
    overlay_terrains = set()
    treated_textures = set()
    dsf_pools = {}
    # we need more pools for textured nodes than for nodes : land, UV masked
    # water, and XP water
    dsf_pool_nbr = 3 * pool_nbr
    for idx_dsfpool in range(dsf_pool_nbr):
        dsf_pools[idx_dsfpool] = array.array("H")
    dsf_pool_length = numpy.zeros(dsf_pool_nbr, "int")
    if (tile.water_tech == "XP11 + bathy"):
        # Land with ortho
        dsf_pool_plane = 7 * numpy.ones(dsf_pool_nbr, "int")
        # Masked ortho (UV1 = ortho, UV2 = border_tex (if aplicable))
        dsf_pool_plane[pool_nbr : 2 * pool_nbr] = 9
        # Regular XP water
        dsf_pool_plane[2 * pool_nbr : 3 * pool_nbr] = 7
    elif (tile.water_tech == "XP12"):
        # Land with ortho
        dsf_pool_plane = 7 * numpy.ones(dsf_pool_nbr, "int")
        # Masked ortho : UV1 = fetch/depth UV2 = ortho)
        #           or : UV1 = ortho, V2 = border_tex
        #                for inland water with constant alpha
        dsf_pool_plane[pool_nbr : 2 * pool_nbr] = 9
        # Regular XP water
        dsf_pool_plane[2 * pool_nbr : 3 * pool_nbr] = 7
    textured_nodes = {}
    len_textured_nodes = 0
    textured_tris = {}
    total_cross_pool = 0
    ##########################

    bPROP = b""
    bTERT = b""
    bOBJT = b""
    bPOLY = b""
    bNETW = b""
    bDEMN = b""
    bGEOD = b""
    bDEMS = b""
    bCMDS = b""

    nbr_dsfpools_yet_in = 0
    dico_terrains = {"terrain_Water": 0}
    bTERT = bytes("terrain_Water\0", "ascii")
    textured_tris[0] = defaultdict(lambda: array.array("H"))

    # Next, we build DSF mesh points (these take into accound texture 
    # as well), point pools, etc.

    # Cache functions and variables locally to reduce dot resolution overhead
    _set_depth_ratio = BATHY.set_depth_ratio
    _progress_bar = UI.progress_bar
    _vprint = UI.vprint
    _water_tech = tile.water_tech
    _imprint_masks_to_dds = tile.imprint_masks_to_dds
    _normal_map_strength = tile.normal_map_strength

    step = nbr_tris // 100 + 1
    
    # Tri counter for progress_bars
    done = 0
    phase_started = time.perf_counter()

    
    # First potentially masked water tris
    for tri in range(nbr_tris):
        tri_type = tri_types[tri]
        if (tri_type != 2):
            continue
        (n1, n2, n3) = tri_idx[3 * tri: 3 * tri + 3]
        if done % step == 0:
            _progress_bar(1, int(done / step * 0.9))
            if UI.is_cancel_requested():
                _vprint(1, "DSF construction interrupted.")
                return 0
        done += 1
        texture_attributes = tri_tex_attr[tri]
        # The entries for the terrain and texture main dictionnaries
        terrain_attributes = (texture_attributes, tri_type)
        is_overlay = False
        

        # Do we need to build new terrain file(s) ?
        if terrain_attributes in dico_terrains:
            terrain_idx = dico_terrains[terrain_attributes]
            is_overlay = terrain_idx in overlay_terrains
        else:
            texture_plan = texture_requirements[texture_attributes]
            needs_new_terrain = texture_plan["mask_present"]
            if needs_new_terrain:
                _vprint(2, "      Use of an alpha mask.")
            if needs_new_terrain:
                terrain_idx = len(dico_terrains)
                textured_tris[terrain_idx] = defaultdict(
                    lambda: array.array("H")
                )
                dico_terrains[terrain_attributes] = terrain_idx
                
                # Is it an overlay terrain or the new XP 12 phys water type ?
                # XP11 style => overlay
                is_overlay = (_water_tech == "XP11 + bathy") 
                # No alpha channel in DDS => overlay
                is_overlay |= not _imprint_masks_to_dds
                
                if is_overlay:
                    overlay_terrains.add(terrain_idx)
                
                texture_file_name = texture_plan["texture_file_name"]
                # do we need to (re)build a texture ?
                if texture_attributes not in treated_textures:
                    rebuild = texture_plan["rebuild"]
                    if rebuild:
                        download_queue.put(texture_attributes)
                    else:
                        _vprint(
                            2,
                            "   Texture file "
                            + texture_file_name
                            + " already present.",
                        )
                    treated_textures.add(texture_attributes)
                terrain_file_name = create_terrain_file(
                    tile,
                    texture_file_name,
                    *texture_attributes,
                    tri_type,
                    is_overlay
                )
                bTERT += bytes("terrain/" + terrain_file_name + "\0", "ascii")
            else:
                terrain_idx = 0
        
        # We put the tri in the right terrain
        # First the ones associated to the dico_customzl
        if terrain_idx:
            tri_p = array.array("H")
            for vertex_pos, n in enumerate(oriented_tri_nodes[tri]):
                n = int(n)
                idx_pool = int(tri_pool_ids[tri, vertex_pos])
                node_hash = (
                    idx_pool,
                    *node_icoords[5 * n : 5 * n + 2],
                    terrain_idx,
                )
                if node_hash in textured_nodes:
                    (idx_dsfpool, pos_in_pool) = textured_nodes[node_hash]
                else:
                    uv_s, uv_t = triangle_uv[tri, vertex_pos]
                    # BEWARE : normal coordinates are pointing (EAST,SOUTH)
                    # in X-Plane, not (EAST,NORTH) ! (cfr DSF specs), so v -> -v
                    if is_overlay: 
                        idx_dsfpool = idx_pool + pool_nbr
                        # border_tex masks with original normal
                        dsf_pools[idx_dsfpool].extend(
                            node_icoords[5 * n : 5 * n + 5]
                        )
                        dsf_pools[idx_dsfpool].extend(
                            (
                                int(uv_s),
                                int(uv_t),
                                int(uv_s),
                                int(uv_t),
                            )
                        )
                    else:  # dtx5 dds with mask included
                        idx_dsfpool = idx_pool + pool_nbr
                        dsf_pools[idx_dsfpool].extend(
                            node_icoords[5 * n : 5 * n + 5]
                        )
                        # TODO (improve fetch values)
                        ratio_bathy = _set_depth_ratio(n, node_is_coast,
                                                            node_bathy, tile)
                        ratio_fetch = 1
                        if int(texture_attributes[2]) >= 18:
                            ratio_bathy = 0
                            ratio_fetch = 0
                        dsf_pools[idx_dsfpool].extend(
                            (int(65535 * ratio_fetch), int(65535 * ratio_bathy),
                             int(uv_s), int(uv_t))
                        )
                    len_textured_nodes += 1
                    pos_in_pool = dsf_pool_length[idx_dsfpool]
                    textured_nodes[node_hash] = (idx_dsfpool, pos_in_pool)
                    dsf_pool_length[idx_dsfpool] += 1
                tri_p.extend((idx_dsfpool, pos_in_pool))
            # some triangles could be reduced to nothing by the pool snapping,
            # we skip thme (possible killer to X-Plane's drapping of roads ?)
            if (
                tri_p[:2] == tri_p[2:4]
                or tri_p[2:4] == tri_p[4:]
                or tri_p[4:] == tri_p[:2]
            ):
                continue
            if tri_p[0] == tri_p[2] == tri_p[4]:
                textured_tris[terrain_idx][tri_p[0]].extend(
                    (tri_p[1], tri_p[3], tri_p[5])
                )
            else:
                total_cross_pool += 1
                textured_tris[terrain_idx]["cross-pool"].extend(tri_p)
        # X-Plane water
        if (not terrain_idx) or is_overlay: 
            tri_p = array.array("H")
            for vertex_pos, n in enumerate(oriented_tri_nodes[tri]):
                n = int(n)
                node_hash = (n, 0)
                if node_hash in textured_nodes:
                    (idx_dsfpool, pos_in_pool) = textured_nodes[node_hash]
                else:
                    idx_dsfpool = int(tri_pool_ids[tri, vertex_pos]) + 2 * pool_nbr
                    len_textured_nodes += 1
                    pos_in_pool = dsf_pool_length[idx_dsfpool]
                    textured_nodes[node_hash] = [idx_dsfpool, pos_in_pool]
                    # in some cases we might prefer to use normal shading for
                    # some sea triangles too (albedo continuity with elevation
                    # derived masks)
                    # dsf_pools[idx_dsfpool].extend(node_icoords[5*n:5*n+5])
                    dsf_pools[idx_dsfpool].extend(
                        node_icoords[5 * n : 5 * n + 3]
                    )
                    dsf_pools[idx_dsfpool].extend((32768, 32768))
                    ratio_bathy = _set_depth_ratio(n, node_is_coast,
                                                        node_bathy, tile)
                    # TODO improve bathy and fetch ratio variety
                    ratio_fetch = 1
                    dsf_pools[idx_dsfpool].extend((int(65535 * ratio_fetch),
                                                   int(65535 * ratio_bathy))) 
                    dsf_pool_length[idx_dsfpool] += 1
                tri_p.extend((idx_dsfpool, pos_in_pool))
            if tri_p[0] == tri_p[2] == tri_p[4]:
                textured_tris[0][tri_p[0]].extend(
                    (tri_p[1], tri_p[3], tri_p[5])
                )
            else:
                total_cross_pool += 1
                textured_tris[0]["cross-pool"].extend(tri_p)

    # Second land and inland water tris
    for tri in range(nbr_tris):
        tri_type = tri_types[tri]
        if (tri_type == 2):
            continue
        (n1, n2, n3) = tri_idx[3 * tri: 3 * tri + 3]
        
        if done % step == 0:
            _progress_bar(1, int(done / step * 0.9))
            if UI.is_cancel_requested():
                _vprint(1, "DSF construction interrupted.")
                return 0
        done += 1
        texture_attributes = tri_tex_attr[tri]
        # The entries for the terrain and texture main dictionnaries
        terrain_attributes = (texture_attributes, tri_type)
        is_overlay = False

        # Do we need to build new terrain file(s) ?
        if terrain_attributes in dico_terrains:
            terrain_idx = dico_terrains[terrain_attributes]
            is_overlay = terrain_idx in overlay_terrains
        else:
            terrain_idx = len(dico_terrains)
            textured_tris[terrain_idx] = defaultdict(lambda: array.array("H"))
            dico_terrains[terrain_attributes] = terrain_idx
            is_overlay = tri_type == 1
            if is_overlay:
                overlay_terrains.add(terrain_idx)
            texture_plan = texture_requirements[texture_attributes]
            texture_file_name = texture_plan["texture_file_name"]
            # do we need to download a new texture ?
            if texture_attributes not in treated_textures:
                rebuild = texture_plan["rebuild"]
                if (rebuild):
                    download_queue.put(texture_attributes)
                else:
                    _vprint(
                        2,
                        "   Texture file "
                        + texture_file_name
                        + " already present.",
                    )
                treated_textures.add(texture_attributes)
            terrain_file_name = create_terrain_file(
                tile,
                texture_file_name,
                *texture_attributes,
                tri_type,
                is_overlay
            )
            bTERT += bytes("terrain/" + terrain_file_name + "\0", "ascii")
        # We put the tri in the right terrain
        # First the ones associated to the dico_customzl
        tri_p = array.array("H")
        for vertex_pos, n in enumerate(oriented_tri_nodes[tri]):
            n = int(n)
            idx_pool = int(tri_pool_ids[tri, vertex_pos])
            node_hash = (
                idx_pool,
                *node_icoords[5 * n : 5 * n + 2],
                terrain_idx,
            )
            if node_hash in textured_nodes:
                (idx_dsfpool, pos_in_pool) = textured_nodes[node_hash]
            else:
                uv_s, uv_t = triangle_uv[tri, vertex_pos]
                # BEWARE : normal coordinates are pointing (EAST,SOUTH) in 
                # X-Plane, not (EAST,NORTH) ! (cfr DSF specs), so v -> -v
                if not tri_type:  # land
                    idx_dsfpool = idx_pool
                    dsf_pools[idx_dsfpool].extend(
                        node_icoords[5 * n : 5 * n + 5]
                    )
                    dsf_pools[idx_dsfpool].extend(
                        (int(uv_s), int(uv_t))
                    )
                else:  # inland water
                    idx_dsfpool = idx_pool + pool_nbr
                    # constant alpha overlay with flat shading
                    dsf_pools[idx_dsfpool].extend(
                        node_icoords[5 * n : 5 * n + 3]
                    )
                    dsf_pools[idx_dsfpool].extend(
                        (
                            32768,
                            32768,
                            int(uv_s),
                            int(uv_t),
                            0,
                            int(round(tile.ratio_water * 65535)),
                        )
                    )
                len_textured_nodes += 1
                pos_in_pool = dsf_pool_length[idx_dsfpool]
                textured_nodes[node_hash] = (idx_dsfpool, pos_in_pool)
                dsf_pool_length[idx_dsfpool] += 1
            tri_p.extend((idx_dsfpool, pos_in_pool))
        # some triangles could be reduced to nothing by the pool snapping,
        # we skip them (possible killer to X-Plane's drapping of roads ?)
        if (
            tri_p[:2] == tri_p[2:4]
            or tri_p[2:4] == tri_p[4:]
            or tri_p[4:] == tri_p[:2]
        ):
            continue
        if tri_p[0] == tri_p[2] == tri_p[4]:
            textured_tris[terrain_idx][tri_p[0]].extend(
                (tri_p[1], tri_p[3], tri_p[5])
            )
        else:
            total_cross_pool += 1
            textured_tris[terrain_idx]["cross-pool"].extend(tri_p)
        
        # XP water
        if is_overlay: 
            tri_p = array.array("H")
            for vertex_pos, n in enumerate(oriented_tri_nodes[tri]):
                n = int(n)
                node_hash = (n, 0)
                if node_hash in textured_nodes:
                    (idx_dsfpool, pos_in_pool) = textured_nodes[node_hash]
                else:
                    idx_dsfpool = int(tri_pool_ids[tri, vertex_pos]) + 2 * pool_nbr
                    len_textured_nodes += 1
                    pos_in_pool = dsf_pool_length[idx_dsfpool]
                    textured_nodes[node_hash] = [idx_dsfpool, pos_in_pool]
                    # in some cases we might prefer to use normal shading for
                    # some sea triangles too (albedo continuity with elevation
                    # derived masks)
                    # dsf_pools[idx_dsfpool].extend(node_icoords[5*n:5*n+5])
                    dsf_pools[idx_dsfpool].extend(
                        node_icoords[5 * n : 5 * n + 3]
                    )
                    dsf_pools[idx_dsfpool].extend((32768, 32768))
                    ratio_bathy = _set_depth_ratio(n, node_is_coast,
                                                        node_bathy, tile)
                    ratio_fetch = 1
                    dsf_pools[idx_dsfpool].extend((int(65535 * ratio_fetch),
                                                   int(65535 * ratio_bathy))) 
                    dsf_pool_length[idx_dsfpool] += 1
                tri_p.extend((idx_dsfpool, pos_in_pool))
            if tri_p[0] == tri_p[2] == tri_p[4]:
                textured_tris[0][tri_p[0]].extend(
                    (tri_p[1], tri_p[3], tri_p[5])
                )
            else:
                total_cross_pool += 1
                textured_tris[0]["cross-pool"].extend(tri_p)
    
    download_queue.put("quit")
    dsf_timings["triangle_processing_ms"] = (
        time.perf_counter() - phase_started
    ) * 1000.0

    dsf_metrics = DSF_BUDGET.summarize_dsf_pools(
        len_textured_nodes,
        dsf_pool_length,
        dsf_pool_plane,
        dsf_pools,
        pool_count=dsf_pool_nbr,
    )
    command_errors = DSF_BUDGET.validate_dsf_commands(
        textured_tris, dsf_pool_length
    )
    if command_errors:
        dsf_metrics["structurally_valid"] = False
        dsf_metrics["structural_errors"] = (
            *dsf_metrics["structural_errors"],
            *command_errors,
        )
    dsf_budget = DSF_BUDGET.normalize_budget(
        getattr(tile, "dsf_node_budget", DSF_BUDGET.DEFAULT_DSF_NODE_BUDGET)
    )
    dsf_metrics["budget"] = dsf_budget
    dsf_metrics["budget_exceeded"] = dsf_metrics["point_count"] > dsf_budget

    encoding_started = time.perf_counter()
    UI.vprint(1, "-> Encoding of the DSF file")
    UI.vprint(1, "     Final DSF point instances: " + str(dsf_metrics["point_count"]))
    UI.vprint(
        1,
        "     DSF point pools: {} active / {} total; largest pool: {} points".format(
            dsf_metrics["active_pool_count"],
            dsf_metrics["pool_count"],
            dsf_metrics["max_pool_points"],
        ),
    )
    if not dsf_metrics["structurally_valid"]:
        UI.vprint(0, "ERROR: DSF point-pool structural validation failed.")
        for reason in dsf_metrics["structural_errors"]:
            UI.vprint(0, "       " + reason)
        return 0
    UI.vprint(1, "     DSF point-pool structure: valid")
    if dsf_metrics["budget_exceeded"]:
        UI.vprint(
            0,
            "WARNING: Final DSF point instances ({:,}) exceed the advisory "
            "point-pool budget of {:,}.".format(
                dsf_metrics["point_count"], dsf_budget
            ),
        )
        UI.vprint(
            0,
            "         This is an advisory project threshold; full-pipeline "
            "auto-reduction may retry the build.",
        )
    UI.vprint(2, "     Final nbr of cross pool tris: " + str(total_cross_pool))

    # Now is time to write our DSF to disk, the exact binary format is 
    # described on the wiki
    dsf_file_name = os.path.join(
        tile.build_dir,
        "Earth nav data",
        FNAMES.long_latlon(tile.lat, tile.lon) + ".dsf",
    )
    
    # Note: present code should always choose the first branch.
    if bPROP == b"":
        bPROP = bytes(
            "sim/west\0"
            + str(tile.lon)
            + "\0"
            + "sim/east\0"
            + str(tile.lon + 1)
            + "\0"
            + "sim/south\0"
            + str(tile.lat)
            + "\0"
            + "sim/north\0"
            + str(tile.lat + 1)
            + "\0"
            + "sim/creation_agent\0"
            + "Ortho4XP\0",
            "ascii",
        )
    else:
        bPROP += b"sim/creation_agent\0Patched by Ortho4XP\0"

    # Transfer DEM and bathymetry raster from Global Scenery tiles before any
    # temporary DSF can replace the currently active tile.
    extracted_rasters = extract_elevation_and_bathymetry_data(tile.lat, tile.lon)
    if extracted_rasters is None:
        return 0
    (bDEMN, bDEMS) = extracted_rasters

    # Computation of intermediate and of total length
    size_of_head_atom = 16 + len(bPROP)
    size_of_prop_atom = 8 + len(bPROP)
    size_of_defn_atom = (
        48 + len(bTERT) + len(bOBJT) + len(bPOLY) + len(bNETW) + len(bDEMN)
    )
    size_of_geod_atom = 8 + len(bGEOD)
    size_of_dems_atom = 8 + len(bDEMS)
    for k in range(dsf_pool_nbr):
        if dsf_pool_length[k] > 0:
            size_of_geod_atom += 21 + dsf_pool_plane[k] * (
                9 + 2 * dsf_pool_length[k]
            )
    UI.vprint(
        2, "     Size of DEFN atom : " + str(size_of_defn_atom) + " bytes."
    )
    UI.vprint(
        2, "     Size of GEOD atom : " + str(size_of_geod_atom) + " bytes."
    )
    f = open(dsf_file_name + ".tmp", "wb")
    f.write(b"XPLNEDSF")
    f.write(struct.pack("<I", 1))

    # Head super-atom
    f.write(b"DAEH")
    f.write(struct.pack("<I", size_of_head_atom))
    f.write(b"PORP")
    f.write(struct.pack("<I", size_of_prop_atom))
    f.write(bPROP)

    # Definitions super-atom
    f.write(b"NFED")
    f.write(struct.pack("<I", size_of_defn_atom))
    f.write(b"TRET")
    f.write(struct.pack("<I", 8 + len(bTERT)))
    f.write(bTERT)
    f.write(b"TJBO")
    f.write(struct.pack("<I", 8 + len(bOBJT)))
    f.write(bOBJT)
    f.write(b"YLOP")
    f.write(struct.pack("<I", 8 + len(bPOLY)))
    f.write(bPOLY)
    f.write(b"WTEN")
    f.write(struct.pack("<I", 8 + len(bNETW)))
    f.write(bNETW)
    f.write(b"NMED")
    f.write(struct.pack("<I", 8 + len(bDEMN)))
    f.write(bDEMN)

    # Geodata super-atom
    f.write(b"DOEG")
    f.write(struct.pack("<I", size_of_geod_atom))
    f.write(bGEOD)
    for k in range(dsf_pool_nbr):
        if dsf_pool_length[k] == 0:
            continue
        f.write(b"LOOP")
        f.write(
            struct.pack(
                "<I",
                13
                + dsf_pool_plane[k]
                + 2 * dsf_pool_plane[k] * dsf_pool_length[k],
            )
        )
        f.write(struct.pack("<I", dsf_pool_length[k]))
        f.write(struct.pack("<B", dsf_pool_plane[k]))
        for l in range(dsf_pool_plane[k]):
            f.write(b"\x00")
            f.write(dsf_pools[k][l :: dsf_pool_plane[k]].tobytes())
    for k in range(dsf_pool_nbr):
        if dsf_pool_length[k] == 0:
            continue
        f.write(b"LACS")
        f.write(struct.pack("<I", 8 + 8 * dsf_pool_plane[k]))
        for l in range(2 * dsf_pool_plane[k]):
            f.write(struct.pack("<f", pool_param[k % pool_nbr][l]))

    UI.progress_bar(1, 95)
    if UI.is_cancel_requested():
        UI.vprint(1, "DSF construction interrupted.")
        f.close()
        return 0

    # Since we possibly skipped some pools, and since we possibly
    # get pools from elsewhere, we rebuild a dico
    # which tells the pool position in the dsf of a pool prior
    # to the stripping :

    dico_new_dsf_pool = {}
    new_idx_dsfpool = nbr_dsfpools_yet_in
    for k in range(dsf_pool_nbr):
        if dsf_pool_length[k] != 0:
            dico_new_dsf_pool[k] = new_idx_dsfpool
            new_idx_dsfpool += 1
    pool_lookup = dico_new_dsf_pool.__getitem__

    # Commands atom
    # we first compute its size :
    size_of_cmds_atom = 8 + len(bCMDS)
    for terrain_idx in sorted(textured_tris, key=DSF_BUDGET.stable_id_key):
        if len(textured_tris[terrain_idx]) == 0:
            continue
        size_of_cmds_atom += 3
        for idx_dsfpool in sorted(
            textured_tris[terrain_idx], key=DSF_BUDGET.stable_id_key
        ):
            if idx_dsfpool != "cross-pool":
                size_of_cmds_atom += 13 + 2 * (
                    len(textured_tris[terrain_idx][idx_dsfpool])
                    + ceil(len(textured_tris[terrain_idx][idx_dsfpool]) / 255)
                )
            else:
                size_of_cmds_atom += 13 + 2 * (
                    len(textured_tris[terrain_idx][idx_dsfpool])
                    + ceil(len(textured_tris[terrain_idx][idx_dsfpool]) / 510)
                )
    UI.vprint(
        2, "     Size of CMDS atom : " + str(size_of_cmds_atom) + " bytes."
    )
    f.write(b"SDMC")  # CMDS header
    f.write(struct.pack("<I", size_of_cmds_atom))  # CMDS length
    f.write(bCMDS)
    for terrain_idx in sorted(textured_tris, key=DSF_BUDGET.stable_id_key):
        if len(textured_tris[terrain_idx]) == 0:
            continue
        # print("terrain_idx = "+str(terrain_idx))
        f.write(struct.pack("<B", 4))  # SET DEFINITION 16
        f.write(struct.pack("<H", terrain_idx))  # TERRAIN INDEX
        flag = (
            1 if terrain_idx not in overlay_terrains else 2
        )  # physical or overlay
        lod = -1 if flag == 1 else tile.overlay_lod
        for idx_dsfpool in sorted(
            textured_tris[terrain_idx], key=DSF_BUDGET.stable_id_key
        ):
            if idx_dsfpool != "cross-pool":
                f.write(struct.pack("<B", 1))  # POOL SELECT
                f.write(
                    struct.pack("<H", dico_new_dsf_pool[idx_dsfpool])
                )  # POOL INDEX

                f.write(struct.pack("<B", 18))  # TERRAIN PATCH FLAGS AND LOD
                f.write(struct.pack("<B", flag))  # FLAG
                f.write(struct.pack("<f", 0))  # NEAR LOD
                f.write(struct.pack("<f", lod))  # FAR LOD

                blocks = len(textured_tris[terrain_idx][idx_dsfpool]) // 255
                data_array = textured_tris[terrain_idx][idx_dsfpool]
                for j in range(blocks):
                    f.write(struct.pack("<BB", 23, 255))
                    f.write(data_array[255 * j : 255 * (j + 1)].tobytes())
                remaining_tri_p = len(data_array) % 255
                if remaining_tri_p != 0:
                    f.write(struct.pack("<BB", 23, remaining_tri_p))
                    f.write(data_array[255 * blocks : ].tobytes())
            else:  # idx_dsfpool == 'cross-pool'
                pool_idx_init = textured_tris[terrain_idx][idx_dsfpool][0]
                f.write(struct.pack("<B", 1))  # POOL SELECT
                f.write(
                    struct.pack("<H", dico_new_dsf_pool[pool_idx_init])
                )  # POOL INDEX
                f.write(struct.pack("<B", 18))  # TERRAIN PATCH FLAGS AND LOD
                f.write(struct.pack("<B", flag))  # FLAG
                f.write(struct.pack("<f", 0))  # NEAR LOD
                f.write(struct.pack("<f", lod))  # FAR LOD

                blocks = len(textured_tris[terrain_idx][idx_dsfpool]) // 510
                data_array = textured_tris[terrain_idx][idx_dsfpool]
                for j in range(blocks):
                    f.write(struct.pack("<BB", 24, 255))  # PATCH TRIANGLE CROSS-POOL + COUNT
                    chunk = data_array[510 * j : 510 * (j + 1)]
                    remapped = array.array("H", chunk)
                    remapped[0::2] = array.array(
                        "H",
                        (pool_lookup(pool_idx) for pool_idx in chunk[0::2]),
                    )
                    f.write(remapped.tobytes())
                remaining_tri_p = (len(data_array) % 510) // 2
                if remaining_tri_p != 0:
                    f.write(struct.pack("<BB", 24, remaining_tri_p))  # PATCH TRIANGLE CROSS-POOL + COUNT
                    chunk = data_array[510 * blocks : ]
                    remapped = array.array("H", chunk)
                    remapped[0::2] = array.array(
                        "H",
                        (pool_lookup(pool_idx) for pool_idx in chunk[0::2]),
                    )
                    f.write(remapped.tobytes())

    # DEMS atom
    if bDEMS != b"":
        f.write(b"SMED")
        f.write(struct.pack("<I", 8 + len(bDEMS)))
        f.write(bDEMS)

    UI.progress_bar(1, 98)
    if UI.is_cancel_requested():
        UI.vprint(1, "DSF construction interrupted.")
        f.close()
        return 0

    f.close()

    f = open(dsf_file_name + ".tmp", "rb")
    data = f.read()
    m = hashlib.md5()
    m.update(data)
    md5sum = m.digest()
    f.close()
    f = open(dsf_file_name + ".tmp", "ab")
    f.write(md5sum)
    f.close()
    
    UI.progress_bar(1, 100)
    
    size_of_dsf = (
        28
        + size_of_head_atom
        + size_of_defn_atom
        + size_of_geod_atom
        + size_of_cmds_atom
        + size_of_dems_atom
    )
    UI.vprint(
        1,
        "     DSF file encoded, total size is :",
        size_of_dsf,
        "bytes",
        "(" + UI.human_print(size_of_dsf) + ")",
    )
    dsf_timings["encoding_ms"] = (time.perf_counter() - encoding_started) * 1000.0
    dsf_metrics["timings_ms"] = dsf_timings
    performance_metrics = getattr(tile, "_performance_metrics", None)
    if performance_metrics is not None:
        performance_metrics.set_value("dsf_timings_ms", dsf_timings)
    tile.last_dsf_metrics = dsf_metrics
    return 1


##############################################################################
