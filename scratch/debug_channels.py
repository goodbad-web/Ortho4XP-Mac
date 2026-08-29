import os
import sys
import numpy
from PIL import Image, ImageDraw

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import O4_UI_Utils as UI
import O4_Imagery_Utils as IMG

IMG.color_filters_dict = {
    "test_brightness": [("brightness-contrast", 20, 15)],
}

# Create a test image with distinct RGB colors
im = Image.new("RGB", (100, 100))
draw = ImageDraw.Draw(im)
draw.rectangle([0, 0, 100, 100], fill=(200, 50, 80)) # Distinct colors

# CPU Path with explicit try-except to catch silent errors
UI.use_gpu_for_color_filters = False
try:
    # Explicitly run the PIL path steps to see if they fail
    color_code = "test_brightness"
    color_filter = IMG.color_filters_dict[color_code][0]
    (brightness, contrast) = color_filter[1:3]
    from math import tan, pi
    # Try running the exact PIL point operation
    test_im = im.copy()
    test_im = test_im.point(
        lambda i: 128
        + tan(pi / 4 * (1 + contrast / 128))
        * (brightness + (255 - brightness) / 255 * i - 128)
    )
    arr_cpu = numpy.array(test_im)
    print("PIL point operation succeeded without error!")
except Exception as ex:
    print(f"PIL point operation failed with exception: {ex}")
    import traceback
    traceback.print_exc()
    arr_cpu = numpy.array(im.copy())

# GPU Path
UI.use_gpu_for_color_filters = True
im_gpu = IMG.color_transform(im.copy(), "test_brightness")
arr_gpu = numpy.array(im_gpu)

print(f"Original pixel: {numpy.array(im)[50, 50]}")
print(f"CPU pixel:      {arr_cpu[50, 50]}")
print(f"GPU pixel:      {arr_gpu[50, 50]}")
print(f"Diff:           {arr_cpu[50, 50].astype(int) - arr_gpu[50, 50].astype(int)}")
