import cv2
import numpy
from PIL import Image

# 1. Create a simple 3-channel RGB image
img_np = numpy.zeros((1, 5, 3), dtype=numpy.uint8)
img_np[0, :, 0] = [10, 50, 100, 150, 200] # R channel
img_np[0, :, 1] = [10, 50, 100, 150, 200] # G channel
img_np[0, :, 2] = [10, 50, 100, 150, 200] # B channel

print("Original array:")
print(img_np)

# 2. Compute a dummy LUT (e.g. out = in + 5)
lut_1d = numpy.clip(numpy.arange(256) + 5, 0, 255).astype(numpy.uint8)

# Test 1: Using 1D LUT (256,)
try:
    res_1d = cv2.LUT(img_np, lut_1d)
    print("\nResult with 1D LUT (256,):")
    print(res_1d)
except Exception as e:
    print(f"\n1D LUT error: {e}")

# Test 2: Using 2D LUT (256, 1)
try:
    lut_2d = lut_1d.reshape((256, 1))
    res_2d = cv2.LUT(img_np, lut_2d)
    print("\nResult with 2D LUT (256, 1):")
    print(res_2d)
except Exception as e:
    print(f"\n2D LUT error: {e}")
