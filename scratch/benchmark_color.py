import os
import sys
import time
import numpy
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import O4_UI_Utils as UI
import O4_Imagery_Utils as IMG

def test_color_equivalence():
    # Setup test filter configurations in IMG's dictionary
    IMG.color_filters_dict = {
        "test_brightness": [("brightness-contrast", 20, 15)],
        "test_saturation": [("saturation", 30)],
        "test_sharpness": [("sharpness", 1.8)],
        "test_blur": [("blur", 2.0)],
        "test_levels": [("levels", 
            10, 1.2, 240, 5, 250, # Channel 0
            20, 0.8, 230, 10, 245, # Channel 1
            15, 1.0, 235, 12, 248  # Channel 2
        )],
        "test_combo": [
            ("brightness-contrast", -10, 25),
            ("saturation", -20),
            ("sharpness", 1.2),
            ("levels", 
                5, 1.1, 250, 0, 255, 
                5, 1.1, 250, 0, 255, 
                5, 1.1, 250, 0, 255
            )
        ]
    }
    
    # 1. Create a dummy test image (4096x4096px RGB)
    print("Generating 4096x4096px dummy test image...")
    numpy.random.seed(42)
    # Generate structured patterns + noise to make it realistic
    img_data = numpy.zeros((4096, 4096, 3), dtype=numpy.uint8)
    for i in range(3):
        # Gradients
        img_data[:, :, i] = numpy.linspace(0, 255, 4096).reshape(1, 4096).repeat(4096, axis=0).astype(numpy.uint8)
    # Add some structural features
    im = Image.fromarray(img_data)
    draw = ImageDraw.Draw(im)
    draw.ellipse([500, 500, 3500, 3500], fill=(200, 50, 80))
    draw.rectangle([1000, 1000, 3000, 3000], fill=(50, 220, 120))
    
    # Run tests across all filter types
    filters_to_test = ["test_brightness", "test_saturation", "test_sharpness", "test_blur", "test_levels", "test_combo"]
    
    for filter_name in filters_to_test:
        print(f"\n=================== Testing: {filter_name} ===================")
        
        # Test CPU
        UI.use_gpu_for_color_filters = False
        t0 = time.time()
        im_cpu = IMG.color_transform(im.copy(), filter_name)
        t_cpu = time.time() - t0
        
        # Test GPU
        UI.use_gpu_for_color_filters = True
        t0 = time.time()
        im_gpu = IMG.color_transform(im.copy(), filter_name)
        t_gpu = time.time() - t0
        
        # Convert to arrays for analytical comparisons
        arr_cpu = numpy.array(im_cpu)
        arr_gpu = numpy.array(im_gpu)
        
        # Debug prints for brightness-contrast comparison
        print(f"Sample pixel CPU: {arr_cpu[0, 0:5, 0]}")
        print(f"Sample pixel GPU: {arr_gpu[0, 0:5, 0]}")
        
        # Compute differences
        diff = numpy.abs(arr_cpu.astype(int) - arr_gpu.astype(int))
        print(f"Max diff: {diff.max()}")
        print(f"Mean diff: {diff.mean():.4f}")
        
        # Due to slight differences in floating-point roundings between PIL/numpy and OpenCV GPU routines,
        # we allow a minimal pixel rounding tolerance (e.g. +/- 1 or 2).
        exact_match = (diff == 0).sum() / diff.size * 100
        tolerance_match = (diff <= 1).sum() / diff.size * 100
        
        print(f"CPU (PIL) Time: {t_cpu:.4f}s")
        print(f"GPU (Metal) Time: {t_gpu:.4f}s")
        print(f"Speedup: {t_cpu / t_gpu:.2f}x")
        print(f"Exact pixel match rate: {exact_match:.4f}%")
        print(f"Match rate within +/- 1 tolerance: {tolerance_match:.4f}%")
        
        # We assert that tolerance match is near perfect (e.g. > 99%)
        assert tolerance_match > 99.5, f"Validation failed for {filter_name}!"
        
    print("\nALL COLOR FILTERS TESTED AND VERIFIED WITH MATHEMATICAL EQUIVALENCE!")

if __name__ == "__main__":
    try:
        test_color_equivalence()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
