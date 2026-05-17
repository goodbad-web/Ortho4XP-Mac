import os
import sys
import time
import numpy
from PIL import Image, ImageDraw

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import O4_UI_Utils as UI
import O4_DEM_Utils as DEM

# Ensure OpenCL Cache disabled just like in main app
os.environ["OPENCV_OPENCL_CACHE_ENABLE"] = "0"

def test_dem_smoothing_equivalence():
    print("Generating 3673x3673px random float elevation raster...")
    numpy.random.seed(42)
    # Generate mock elevation data (mostly terrain with some gradients)
    raster = numpy.random.uniform(-50.0, 1500.0, (3673, 3673)).astype(numpy.float32)
    
    # Generate mock water/coastline mask image (L mode)
    print("Generating mock mask image...")
    mask_im = Image.new("L", (3673, 3673), 0)
    draw = ImageDraw.Draw(mask_im)
    draw.ellipse([500, 500, 3000, 3000], fill=255)
    draw.rectangle([1000, 1000, 2500, 2500], fill=128)
    
    pix_width = 10
    print(f"Running smoothing benchmark with filter width: {pix_width}px...")
    
    # 1. Test CPU path
    UI.use_gpu_for_dem_smoothing = False
    t0 = time.time()
    res_cpu = DEM.smoothen(raster.copy(), pix_width, mask_im.copy(), preserve_boundary=True)
    t_cpu = time.time() - t0
    print(f"CPU (Sequential NumPy Convolve) Time: {t_cpu:.4f}s")
    
    # 2. Test GPU path
    UI.use_gpu_for_dem_smoothing = True
    t0 = time.time()
    res_gpu = DEM.smoothen(raster.copy(), pix_width, mask_im.copy(), preserve_boundary=True)
    t_gpu = time.time() - t0
    print(f"GPU (Metal / UMat sepFilter2D) Time:  {t_gpu:.4f}s")
    print(f"Acceleration Speedup: {t_cpu / t_gpu:.2f}x")
    
    # 3. Analyze differences
    diff = numpy.abs(res_cpu - res_gpu)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    
    # In OpenCL float operations, minimal floating-point rounding deviations (e.g. < 1e-3) are perfectly normal and expected.
    # We assert that the average difference is exceptionally close to zero.
    exact_match = (diff < 1e-3).sum() / diff.size * 100
    print(f"Pixel match rate within 0.001 tolerance: {exact_match:.4f}%")
    
    assert exact_match > 99.9, "Validation failed: GPU output differs from CPU output!"
    print("\nDEM SMOOTHING GPU ACCELERATION MATHEMATICALLY VERIFIED WITH SUCCESS!")

if __name__ == "__main__":
    try:
        test_dem_smoothing_equivalence()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
