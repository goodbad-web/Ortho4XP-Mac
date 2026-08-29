import os
import sys
import time
from math import atan
import numpy
import cv2
from PIL import Image, ImageDraw, ImageFilter

# Add src to python path to import GEO etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import O4_Geo_Utils as GEO
import O4_Mask_Utils as MASK

# Original Pillow-based drawing function for validation
def build_water_pre_mask_original(til_x, til_y, mesh_list, dico_sea, dico_inland, sea_level, tile):
    (latm0, lonm0) = GEO.gtile_to_wgs84(til_x, til_y, tile.mask_zl)
    (px0, py0) = GEO.wgs84_to_pix(latm0, lonm0, tile.mask_zl)
    px0 -= 1024
    py0 -= 1024
    mask_im = Image.new("L", (4096 + 2 * 1024, 4096 + 2 * 1024), "black")
    mask_draw = ImageDraw.Draw(mask_im)
    for mesh_file_name in mesh_list:
        latlonstr = mesh_file_name.split(".mes")[-2][-7:]
        lathere = int(latlonstr[0:3])
        lonhere = int(latlonstr[3:7])
        (px1, py1) = GEO.wgs84_to_pix(lathere, lonhere, tile.mask_zl)
        (px2, py2) = GEO.wgs84_to_pix(lathere, lonhere + 1, tile.mask_zl)
        (px3, py3) = GEO.wgs84_to_pix(
            lathere + 1, lonhere + 1, tile.mask_zl
        )
        (px4, py4) = GEO.wgs84_to_pix(lathere + 1, lonhere, tile.mask_zl)
        px1 -= px0
        px2 -= px0
        px3 -= px0
        px4 -= px0
        py1 -= py0
        py2 -= py0
        py3 -= py0
        py4 -= py0
        mask_draw.polygon(
            [(px1, py1), (px2, py2), (px3, py3), (px4, py4)], fill="white"
        )
    if (til_x, til_y) in dico_inland:
        for (lat1, lon1, lat2, lon2, lat3, lon3) in dico_inland[(til_x, til_y)]:
            (px1, py1) = GEO.wgs84_to_pix(lat1, lon1, tile.mask_zl)
            (px2, py2) = GEO.wgs84_to_pix(lat2, lon2, tile.mask_zl)
            (px3, py3) = GEO.wgs84_to_pix(lat3, lon3, tile.mask_zl)
            px1 -= px0
            px2 -= px0
            px3 -= px0
            py1 -= py0
            py2 -= py0
            py3 -= py0
            mask_draw.polygon([(px1, py1), (px2, py2), (px3, py3)], fill=sea_level)
    if (til_x, til_y) in dico_sea:
        for (lat1, lon1, lat2, lon2, lat3, lon3) in dico_sea[(til_x, til_y)]:
            (px1, py1) = GEO.wgs84_to_pix(lat1, lon1, tile.mask_zl)
            (px2, py2) = GEO.wgs84_to_pix(lat2, lon2, tile.mask_zl)
            (px3, py3) = GEO.wgs84_to_pix(lat3, lon3, tile.mask_zl)
            px1 -= px0
            px2 -= px0
            px3 -= px0
            py1 -= py0
            py2 -= py0
            py3 -= py0
            mask_draw.polygon([(px1, py1), (px2, py2), (px3, py3)], fill="black")
    img_array = numpy.array(mask_im, dtype=numpy.uint8)
    return img_array

# Original Pillow-based blur functions
def blur_mask_sand_original(img_array, tile, sea_level, blur_width):
    b_img_array = numpy.array(img_array)
    kernel = numpy.array(range(1, 2 * blur_width))
    kernel[blur_width:] = range(blur_width - 1, 0, -1)
    kernel = kernel / blur_width ** 2
    for i in range(0, len(b_img_array)):
        b_img_array[i] = numpy.convolve(b_img_array[i], kernel, "same")
    b_img_array = b_img_array.transpose()
    for i in range(0, len(b_img_array)):
        b_img_array[i] = numpy.convolve(b_img_array[i], kernel, "same")
    b_img_array = b_img_array.transpose()
    b_img_array = 2 * numpy.minimum(b_img_array, 127)
    b_img_array = numpy.array(b_img_array, dtype=numpy.uint8)
    return b_img_array

def blur_mask_rocks_original(img_array, tile, sea_level, blur_width):
    b_img_array = (
        numpy.array(
            Image.fromarray(img_array)
            .convert("L")
            .filter(ImageFilter.GaussianBlur(blur_width / 1.7)),
            dtype=numpy.uint8,
        )
        > 0
    ).astype(numpy.uint8) * 255
    # blur it
    b_img_array = numpy.array(
        Image.fromarray(b_img_array)
        .convert("L")
        .filter(ImageFilter.GaussianBlur(blur_width)),
        dtype=numpy.uint8,
    )
    # nonlinear transform
    gamma = 2.5
    b_img_array = (
        (
            (
                numpy.tan(
                    (b_img_array.astype(numpy.float32) - 127.5)
                    / 128
                    * atan(3)
                )
                - numpy.tan(-127.5 / 128 * atan(3))
            )
            * 254
            / (2 * numpy.tan(127.5 / 128 * atan(3)))
        )
        ** gamma
        / (255 ** (gamma - 1))
    ).astype(numpy.uint8)
    # still some slight smoothing at the shore
    b_img_array = numpy.maximum(
        b_img_array,
        numpy.array(
            Image.fromarray(img_array)
            .convert("L")
            .filter(ImageFilter.GaussianBlur(2 ** (tile.mask_zl - 14))),
            dtype=numpy.uint8,
        ),
    )
    return b_img_array

def blur_mask_3steps_original(img_array, tile, sea_level, blur_width):
    def transition_profile(ratio, ttype):
        if ttype == "spline":
            return 3 * ratio ** 2 - 2 * ratio ** 3
        elif ttype == "linear":
            return ratio
        elif ttype == "parabolic":
            return 2 * ratio - ratio ** 2
            
    transin = blur_width[0]
    midzone = blur_width[1]
    transout = blur_width[2]
    shore_level = 255
    b_img_array = b_mask_array = numpy.array(img_array)
    
    stepsin = int(transin / 3)
    for i in range(stepsin):
        value = shore_level + transition_profile(
            (i + 1) / stepsin, "parabolic"
        ) * (sea_level - shore_level)
        b_mask_array = (
            numpy.array(
                Image.fromarray(b_mask_array)
                .convert("L")
                .filter(ImageFilter.GaussianBlur(1)),
                dtype=numpy.uint8,
            )
            > 0
        ).astype(numpy.uint8) * 255
        b_img_array[(b_img_array == 0) * (b_mask_array != 0)] = value
        
    sea_b_radius = midzone / 3
    sea_b_radius_buffered = (midzone + transout) / 3
    b_mask_array = (
        numpy.array(
            Image.fromarray(b_mask_array)
            .convert("L")
            .filter(ImageFilter.GaussianBlur(sea_b_radius_buffered)),
            dtype=numpy.uint8,
        )
        > 0
    ).astype(numpy.uint8) * 255
    b_mask_array = (
        numpy.array(
            Image.fromarray(b_mask_array)
            .convert("L")
            .filter(
                ImageFilter.GaussianBlur(
                    sea_b_radius_buffered - sea_b_radius
                )
            ),
            dtype=numpy.uint8,
        )
        == 255
    ).astype(numpy.uint8) * 255
    b_img_array[(b_img_array == 0) * (b_mask_array != 0)] = sea_level
    
    stepsout = int(transout / 3)
    for i in range(stepsout):
        value = sea_level * (
            1 - transition_profile((i + 1) / stepsout, "linear")
        )
        b_mask_array = (
            numpy.array(
                Image.fromarray(b_mask_array)
                .convert("L")
                .filter(ImageFilter.GaussianBlur(1)),
                dtype=numpy.uint8,
            )
            > 0
        ).astype(numpy.uint8) * 255
        b_img_array[(b_img_array == 0) * (b_mask_array != 0)] = value
        
    b_img_array = numpy.array(
        Image.fromarray(b_img_array)
        .convert("L")
        .filter(ImageFilter.GaussianBlur(2)),
        dtype=numpy.uint8,
    )
    return b_img_array

# Mock tile class
class MockTile:
    def __init__(self):
        self.lat = 34
        self.lon = 135
        self.mask_zl = 14
        self.masking_mode = "sand"
        self.masks_width = 100

def test_equivalence():
    tile = MockTile()
    tile.use_gpu_for_masks = False # Initialize CPU mode
    til_x, til_y = 13328, 6496  # Mock values
    mesh_list = ["+34+135.mes"]
    
    # Generate 1000 mock triangles to represent real water scenery density
    numpy.random.seed(42)
    lat_center, lon_center = tile.lat + 0.5, tile.lon + 0.5
    dico_sea = {(til_x, til_y): []}
    dico_inland = {(til_x, til_y): []}
    
    for _ in range(500):
        # random small triangles
        l1, lo1 = lat_center + numpy.random.uniform(-0.01, 0.01), lon_center + numpy.random.uniform(-0.01, 0.01)
        l2, lo2 = l1 + 0.001, lo1
        l3, lo3 = l1, lo1 + 0.001
        dico_sea[(til_x, til_y)].append((l1, lo1, l2, lo2, l3, lo3))
        
        l1, lo1 = lat_center + numpy.random.uniform(-0.01, 0.01), lon_center + numpy.random.uniform(-0.01, 0.01)
        l2, lo2 = l1 + 0.001, lo1
        l3, lo3 = l1, lo1 + 0.001
        dico_inland[(til_x, til_y)].append((l1, lo1, l2, lo2, l3, lo3))
        
    sea_level = 150
    
    print("--- 1. Testing build_water_pre_mask Equivalency ---")
    t0 = time.time()
    arr_orig = build_water_pre_mask_original(til_x, til_y, mesh_list, dico_sea, dico_inland, sea_level, tile)
    t_orig = time.time() - t0
    
    t0 = time.time()
    arr_opt = MASK.build_water_pre_mask(til_x, til_y, mesh_list, dico_sea, dico_inland, sea_level, tile)
    t_opt = time.time() - t0
    
    diff = numpy.abs(arr_orig.astype(int) - arr_opt.astype(int))
    pixel_match = (diff == 0).sum() / diff.size * 100
    print(f"Pre-mask generation - Original: {t_orig:.4f}s | Optimized: {t_opt:.4f}s | Speedup: {t_orig / t_opt:.2f}x")
    print(f"Pre-mask pixel match rate: {pixel_match:.4f}%")
    assert pixel_match > 99.0, "Pre-mask verification failed!"

    print("\n--- 2. Testing blur_mask (Sand Mode) CPU vs GPU ---")
    blur_width = 30
    tile.masking_mode = "sand"
    tile.masks_width = blur_width * GEO.webmercator_pixel_size(tile.lat + 0.5, tile.mask_zl)
    
    t0 = time.time()
    blur_sand_orig = blur_mask_sand_original(arr_orig, tile, sea_level, blur_width)
    t_sand_orig = time.time() - t0
    
    # Test CPU
    tile.use_gpu_for_masks = False
    t0 = time.time()
    blur_sand_cpu = MASK.blur_mask(arr_orig, tile, sea_level)
    t_sand_cpu = time.time() - t0
    
    # Test GPU
    tile.use_gpu_for_masks = True
    t0 = time.time()
    blur_sand_gpu = MASK.blur_mask(arr_orig, tile, sea_level)
    t_sand_gpu = time.time() - t0
    
    # Compare CPU vs GPU equivalence
    diff_gpu = numpy.abs(blur_sand_cpu.astype(int) - blur_sand_gpu.astype(int))
    gpu_pixel_match = (diff_gpu == 0).sum() / diff_gpu.size * 100
    print(f"Sand Mode - CPU: {t_sand_cpu:.4f}s | GPU: {t_sand_gpu:.4f}s | Speedup (vs Orig): {t_sand_orig / t_sand_gpu:.2f}x")
    print(f"CPU vs GPU Match Rate: {gpu_pixel_match:.4f}%")
    assert gpu_pixel_match == 100.0, "GPU Sand mode produced mathematically different output!"
    
    diff_sand = numpy.abs(blur_sand_orig.astype(int) - blur_sand_cpu.astype(int))
    sand_match = (diff_sand == 0).sum() / diff_sand.size * 100
    assert sand_match > 99.9, "Sand mode verification failed!"

    print("\n--- 3. Testing blur_mask (Rocks Mode) CPU vs GPU ---")
    blur_width = 30
    tile.masking_mode = "rocks"
    tile.masks_width = blur_width * (2 * GEO.webmercator_pixel_size(tile.lat + 0.5, tile.mask_zl))
    
    t0 = time.time()
    blur_rocks_orig = blur_mask_rocks_original(arr_orig, tile, sea_level, blur_width)
    t_rocks_orig = time.time() - t0
    
    # Test CPU
    tile.use_gpu_for_masks = False
    t0 = time.time()
    blur_rocks_cpu = MASK.blur_mask(arr_orig, tile, sea_level)
    t_rocks_cpu = time.time() - t0
    
    # Test GPU
    tile.use_gpu_for_masks = True
    t0 = time.time()
    blur_rocks_gpu = MASK.blur_mask(arr_orig, tile, sea_level)
    t_rocks_gpu = time.time() - t0
    
    # Compare CPU vs GPU equivalence
    diff_gpu = numpy.abs(blur_rocks_cpu.astype(int) - blur_rocks_gpu.astype(int))
    gpu_pixel_match = (diff_gpu == 0).sum() / diff_gpu.size * 100
    print(f"Rocks Mode - CPU: {t_rocks_cpu:.4f}s | GPU: {t_rocks_gpu:.4f}s | Speedup (vs Orig): {t_rocks_orig / t_rocks_gpu:.2f}x")
    print(f"CPU vs GPU Match Rate: {gpu_pixel_match:.4f}%")
    assert gpu_pixel_match == 100.0, "GPU Rocks mode produced mathematically different output!"
    
    diff_rocks = numpy.abs(blur_rocks_orig.astype(int) - blur_rocks_cpu.astype(int))
    rocks_match = (diff_rocks <= 2).sum() / diff_rocks.size * 100
    assert rocks_match > 99.0, "Rocks mode verification failed!"

    print("\n--- 4. Testing blur_mask (3-Steps Mode) CPU vs GPU ---")
    # Using 3-step radius values
    blur_width_3s = [15, 30, 45]
    tile.masking_mode = "3steps"
    tile.masks_width = [L * GEO.webmercator_pixel_size(tile.lat + 0.5, tile.mask_zl) for L in blur_width_3s]
    
    t0 = time.time()
    blur_3s_orig = blur_mask_3steps_original(arr_orig, tile, sea_level, blur_width_3s)
    t_3s_orig = time.time() - t0
    
    # Test CPU
    tile.use_gpu_for_masks = False
    t0 = time.time()
    blur_3s_cpu = MASK.blur_mask(arr_orig, tile, sea_level)
    t_3s_cpu = time.time() - t0
    
    # Test GPU
    tile.use_gpu_for_masks = True
    t0 = time.time()
    blur_3s_gpu = MASK.blur_mask(arr_orig, tile, sea_level)
    t_3s_gpu = time.time() - t0
    
    # Compare CPU vs GPU equivalence
    diff_gpu = numpy.abs(blur_3s_cpu.astype(int) - blur_3s_gpu.astype(int))
    gpu_pixel_match = (diff_gpu == 0).sum() / diff_gpu.size * 100
    print(f"3-Steps Mode - CPU: {t_3s_cpu:.4f}s | GPU: {t_3s_gpu:.4f}s | Speedup (vs Orig): {t_3s_orig / t_3s_gpu:.2f}x")
    print(f"CPU vs GPU Match Rate: {gpu_pixel_match:.4f}%")
    assert gpu_pixel_match == 100.0, "GPU 3-Steps mode produced mathematically different output!"
    
    diff_3s = numpy.abs(blur_3s_orig.astype(int) - blur_3s_cpu.astype(int))
    match_3s = (diff_3s <= 3).sum() / diff_3s.size * 100
    assert match_3s > 99.0, "3-steps mode verification failed!"

if __name__ == "__main__":
    try:
        test_equivalence()
        print("\nALL MASKING MODES TESTED AND PASSED SUCCESSFULLY!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
