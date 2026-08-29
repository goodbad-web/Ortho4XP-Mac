import time
import numpy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# 1. Original functions
def float2qquad_orig(x):
    if x >= 1:
        return "111111111111111111111111"
    return numpy.binary_repr(int(16777216 * x)).zfill(24)

# 2. Optimized sequential function
def float2qquad_opt(x):
    if x >= 1:
        return "111111111111111111111111"
    return f"{int(16777216 * x):024b}"

def float2qquad_opt_batch(arr):
    # Vectorized computation but formatted sequentially in python
    val = (16777216 * arr).astype(numpy.int32)
    val = numpy.clip(val, 0, 16777216 - 1)
    return [f"{v:024b}" for v in val]

# Mock QuadTree class for insert simulation
class MockQuadTree:
    def __init__(self):
        self.nodes = []
    def insert(self, bx, by, level):
        self.nodes.append((bx, by))

# 3. Test execution
if __name__ == "__main__":
    print("Generating 3,000,000 mock coordinates (0.0 to 1.0)...")
    n = 3000000
    # Simulate nodes in range [tile_lon, tile_lon + 1.0] and [tile_lat, tile_lat + 1.0]
    node_coords = numpy.zeros(5 * n)
    node_coords[0::5] = 133.0 + numpy.random.rand(n)
    node_coords[1::5] = 34.0 + numpy.random.rand(n)
    
    tile_lon = 133.0
    tile_lat = 34.0
    quad_init_level = 3
    
    # Test Original Loop
    print("Running original loop sequentially...")
    t0 = time.time()
    q_orig = MockQuadTree()
    # Simulating original loop
    for i in range(n):
        q_orig.insert(
            float2qquad_orig(node_coords[5 * i + 0] - tile_lon),
            float2qquad_orig(node_coords[5 * i + 1] - tile_lat),
            quad_init_level
        )
    t_orig = time.time() - t0
    print(f"Original loop completed in: {t_orig:.4f} seconds.")
    
    # Test Optimized Loop
    print("Running optimized batch loop...")
    t1 = time.time()
    q_opt = MockQuadTree()
    
    # 1. NumPy batch slice and transform
    xs = node_coords[0::5] - tile_lon
    ys = node_coords[1::5] - tile_lat
    
    # 2. Optimized batch float2qquad
    bx_list = float2qquad_opt_batch(xs)
    by_list = float2qquad_opt_batch(ys)
    
    # 3. Cache insert method locally to bypass name lookup inside loop
    insert_method = q_opt.insert
    
    # 4. Standard list iteration
    for i in range(n):
        insert_method(bx_list[i], by_list[i], quad_init_level)
        
    t_opt = time.time() - t1
    print(f"Optimized loop completed in: {t_opt:.4f} seconds.")
    
    # Verification
    print("Verifying correctness...")
    if q_orig.nodes == q_opt.nodes:
        print("Success: 100% correct match!")
        print(f"Loop Speedup: {t_orig / t_opt:.2f}x faster!")
    else:
        print("Error: Mismatch!")
