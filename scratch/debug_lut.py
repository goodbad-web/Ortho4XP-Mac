from math import tan, pi
import numpy
from PIL import Image

brightness = 20
contrast = 15

# 1. PIL LUT
im = Image.new("L", (256, 1))
im.putdata(list(range(256)))
im_out = im.point(
    lambda i: 128
    + tan(pi / 4 * (1 + contrast / 128))
    * (brightness + (255 - brightness) / 255 * i - 128)
)
pil_lut = list(im_out.getdata())

# 2. NumPy LUT
numpy_lut = numpy.arange(256, dtype=numpy.float32)
numpy_lut = 128 + tan(pi / 4 * (1 + contrast / 128)) * (brightness + (255 - brightness) / 255 * numpy_lut - 128)
numpy_lut = numpy.clip(numpy_lut, 0, 255).astype(numpy.uint8)

# Print comparison
print("Index | PIL | NumPy | Diff")
for idx in [0, 10, 50, 100, 150, 200, 255]:
    p_val = pil_lut[idx]
    n_val = numpy_lut[idx]
    print(f"{idx:5d} | {p_val:3d} | {n_val:5d} | {p_val - n_val:4d}")
