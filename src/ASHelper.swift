import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Vision
import CoreImage
import Metal
import MetalKit
import MetalFX
import Darwin

struct DDSHeader {
    var magic: UInt32 = 0x20534444; var size: UInt32 = 124; var flags: UInt32 = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000 
    var height: UInt32; var width: UInt32; var pitchOrLinearSize: UInt32; var depth: UInt32 = 0; var mipmapCount: UInt32
    var res1: (UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,UInt32) = (0,0,0,0,0,0,0,0,0,0,0)
    var pfSize: UInt32 = 32; var pfFlags: UInt32 = 0x4; var fourCC: UInt32
    var pfRGBBitCount: UInt32 = 0; var pfRBitMask: UInt32 = 0; var pfGBitMask: UInt32 = 0; var pfBBitMask: UInt32 = 0; var pfABitMask: UInt32 = 0
    var caps: UInt32 = 0x1000 | 0x400000 | 0x8; var caps2: UInt32 = 0; var caps3: UInt32 = 0; var caps4: UInt32 = 0; var reserved2: UInt32 = 0
    func toData() -> Data { var d = Data(); var t = self; withUnsafeBytes(of: &t) { d.append(contentsOf: $0) }; return d }
}

struct DDSHeaderDX10 {
    var dxgiFormat: UInt32; var resourceDimension: UInt32 = 3; var miscFlag: UInt32 = 0; var arraySize: UInt32 = 1; var miscFlags2: UInt32 = 0
    func toData() -> Data { var d = Data(); var t = self; withUnsafeBytes(of: &t) { d.append(contentsOf: $0) }; return d }
}

func checkedMultiply(_ values: Int..., label: String) throws -> Int {
    var result = 1
    for value in values {
        let multiplication = result.multipliedReportingOverflow(by: value)
        guard value >= 0, !multiplication.overflow else {
            throw NSError(domain: "ASHelper", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "integer_overflow_\(label)"
            ])
        }
        result = multiplication.partialValue
    }
    return result
}

let stderrLock = NSLock()

func reportError(_ message: String) {
    stderrLock.lock()
    defer { stderrLock.unlock() }
    if let data = (message + "\n").data(using: .utf8) {
        FileHandle.standardError.write(data)
    }
}

func fail(_ message: String) -> Never {
    reportError(message)
    exit(1)
}

final class BatchFailureState {
    private let lock = NSLock()
    private var failures: [String] = []

    func recordFailure(index: Int, input: String, mask: String, output: String) {
        lock.lock()
        failures.append(
            "task=\(index) input='\(input)' mask='\(mask)' output='\(output)'"
        )
        lock.unlock()
    }

    func hasFailure() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return !failures.isEmpty
    }

    func reportFailures() {
        lock.lock()
        let currentFailures = failures
        lock.unlock()
        for failure in currentFailures {
            reportError("ASHelper: GPU batch failed (\(failure))")
        }
    }
}

struct MetalFXDirectDDSColor: Codable {
    let r: Double
    let g: Double
    let b: Double
    let contrast: Double
    let brightness: Double
    let saturation: Double
}

struct MetalFXDirectDDSItem: Codable {
    let input: String
    let mask: String
    let output: String
    let format: String
    let color: MetalFXDirectDDSColor
}

struct MetalFXDirectDDSRequest: Codable {
    let version: Int
    let items: [MetalFXDirectDDSItem]
}

struct TensorOpsDirectDDSRequest: Codable {
    let version: Int
    let pack: String
    let items: [MetalFXDirectDDSItem]
}

func residentMemoryMB() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
        MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size
    )
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
            task_info(
                mach_task_self_,
                task_flavor_t(MACH_TASK_BASIC_INFO),
                rebound,
                &count
            )
        }
    }
    guard result == KERN_SUCCESS else { return 0 }
    return UInt64(info.resident_size) / (1024 * 1024)
}

let metalSource = """
#include <metal_stdlib>
using namespace metal;
struct ColorBlockResult {
    ushort c0;
    ushort c1;
    uint indices;
    float error;
};

struct AlphaBlockResult {
    uchar a0;
    uchar a1;
    ulong indices;
    float error;
};

float perceptualError(float3 a, float3 b) {
    float3 d = a - b;
    return d.r * d.r * 0.2126 + d.g * d.g * 0.7152 + d.b * d.b * 0.0722;
}

ushort pack565(float3 color) {
    color = clamp(color, 0.0, 1.0);
    return ((ushort)(color.r * 31.0) << 11)
        | ((ushort)(color.g * 63.0) << 5)
        | (ushort)(color.b * 31.0);
}

float3 unpack565(ushort value) {
    return float3(
        float((value >> 11) & 31) / 31.0,
        float((value >> 5) & 63) / 63.0,
        float(value & 31) / 31.0
    );
}

void buildColorPalette(ushort c0, ushort c1, thread float3 palette[4]) {
    float3 p0 = unpack565(c0);
    float3 p1 = unpack565(c1);
    palette[0] = p0;
    palette[1] = p1;
    palette[2] = (2.0 * p0 + p1) / 3.0;
    palette[3] = (p0 + 2.0 * p1) / 3.0;
}

ColorBlockResult evaluateColor(
    thread float3 pixels[16],
    float3 initial0,
    float3 initial1
) {
    float3 endpoint0 = initial0;
    float3 endpoint1 = initial1;

    for (uint iteration = 0; iteration < 8; iteration++) {
        ushort c0 = pack565(endpoint0);
        ushort c1 = pack565(endpoint1);
        if (c0 <= c1) {
            if (c0 < c1) {
                ushort swapped = c0;
                c0 = c1;
                c1 = swapped;
            } else if (c0 < 65535) c0 += 1;
            else if (c1 > 0) c1 -= 1;
        }
        float3 palette[4];
        buildColorPalette(c0, c1, palette);
        uint indices = 0;
        float aa = 0.0; float ab = 0.0; float bb = 0.0;
        float3 ad = float3(0.0); float3 bd = float3(0.0);
        for (uint i = 0; i < 16; i++) {
            uint index = 0;
            float best = perceptualError(pixels[i], palette[0]);
            for (uint candidate = 1; candidate < 4; candidate++) {
                float error = perceptualError(pixels[i], palette[candidate]);
                if (error < best) { best = error; index = candidate; }
            }
            indices |= index << (i * 2);
            float a = 1.0 - float(index) / 3.0;
            float b = float(index) / 3.0;
            aa += a * a; ab += a * b; bb += b * b;
            ad += pixels[i] * a; bd += pixels[i] * b;
        }
        float determinant = aa * bb - ab * ab;
        if (abs(determinant) > 0.000001) {
            endpoint0 = clamp((ad * bb - bd * ab) / determinant, 0.0, 1.0);
            endpoint1 = clamp((bd * aa - ad * ab) / determinant, 0.0, 1.0);
        }
    }

    ushort finalC0 = pack565(endpoint0);
    ushort finalC1 = pack565(endpoint1);
    if (finalC0 <= finalC1) {
        if (finalC0 < finalC1) {
            ushort swapped = finalC0;
            finalC0 = finalC1;
            finalC1 = swapped;
        } else if (finalC0 < 65535) finalC0 += 1;
        else if (finalC1 > 0) finalC1 -= 1;
    }
    float3 palette[4];
    buildColorPalette(finalC0, finalC1, palette);
    uint finalIndices = 0;
    float error = 0.0;
    for (uint i = 0; i < 16; i++) {
        uint index = 0;
        float best = perceptualError(pixels[i], palette[0]);
        for (uint candidate = 1; candidate < 4; candidate++) {
            float candidateError = perceptualError(pixels[i], palette[candidate]);
            if (candidateError < best) {
                best = candidateError;
                index = candidate;
            }
        }
        finalIndices |= index << (i * 2);
        error += best;
    }
    return ColorBlockResult{finalC0, finalC1, finalIndices, error};
}

void buildAlphaPalette(uchar a0, uchar a1, thread float palette[8]) {
    palette[0] = float(a0);
    palette[1] = float(a1);
    if (a0 > a1) {
        for (uint i = 1; i <= 6; i++) {
            palette[i + 1] = (float(7 - i) * float(a0) + float(i) * float(a1)) / 7.0;
        }
    } else {
        for (uint i = 1; i <= 4; i++) {
            palette[i + 1] = (float(5 - i) * float(a0) + float(i) * float(a1)) / 5.0;
        }
        palette[6] = 0.0;
        palette[7] = 255.0;
    }
}

AlphaBlockResult evaluateAlpha(thread float values[16], uchar a0, uchar a1) {
    float palette[8];
    buildAlphaPalette(a0, a1, palette);
    ulong indices = 0;
    float error = 0.0;
    for (uint i = 0; i < 16; i++) {
        uint index = 0;
        float best = abs(values[i] - palette[0]);
        for (uint candidate = 1; candidate < 8; candidate++) {
            float distance = abs(values[i] - palette[candidate]);
            if (distance < best) { best = distance; index = candidate; }
        }
        indices |= (ulong(index) << (i * 3));
        error += best * best;
    }
    return AlphaBlockResult{a0, a1, indices, error};
}

bool isBetterAlpha(AlphaBlockResult candidate, AlphaBlockResult current) {
    return candidate.error < current.error
        || (candidate.error == current.error && candidate.a0 < current.a0)
        || (candidate.error == current.error && candidate.a0 == current.a0 && candidate.a1 < current.a1);
}

AlphaBlockResult optimizeAlpha(thread float values[16]) {
    uchar candidates[18];
    uint candidateCount = 0;
    for (uint i = 0; i < 16; i++) {
        uchar value = (uchar)clamp(values[i], 0.0, 255.0);
        bool exists = false;
        for (uint j = 0; j < candidateCount; j++) if (candidates[j] == value) exists = true;
        if (!exists && candidateCount < 18) candidates[candidateCount++] = value;
    }
    uint sourceCandidateCount = candidateCount;
    if (sourceCandidateCount == 1) {
        uchar value = candidates[0];
        return evaluateAlpha(values, 0, value == 0 ? 255 : value);
    }
    if (sourceCandidateCount == 2) {
        uchar low = min(candidates[0], candidates[1]);
        uchar high = max(candidates[0], candidates[1]);
        AlphaBlockResult best = evaluateAlpha(values, low, high);
        if (high == 255 && low != 0) {
            AlphaBlockResult alternate = evaluateAlpha(values, 0, low);
            if (isBetterAlpha(alternate, best)) best = alternate;
        }
        return best;
    }
    for (uint extra = 0; extra < 2; extra++) {
        uchar value = extra == 0 ? 0 : 255;
        bool exists = false;
        for (uint j = 0; j < candidateCount; j++) if (candidates[j] == value) exists = true;
        if (!exists && candidateCount < 18) candidates[candidateCount++] = value;
    }

    AlphaBlockResult best = AlphaBlockResult{0, 1, 0, INFINITY};
    for (uint i = 0; i < candidateCount; i++) {
        for (uint j = 0; j < candidateCount; j++) {
            if (candidates[i] == candidates[j]) continue;
            AlphaBlockResult current = evaluateAlpha(values, candidates[i], candidates[j]);
            if (isBetterAlpha(current, best)) {
                best = current;
            }
        }
    }
    return best;
}

kernel void compressTexture(
    texture2d<float, access::read> input [[texture(0)]],
    device uchar *output [[buffer(0)]],
    constant uint &formatMode [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 pos = gid * 4;
    if (pos.x >= input.get_width() || pos.y >= input.get_height()) return;
    thread float3 pixels[16];
    thread float alpha[16];
    for (uint i = 0; i < 16; i++) {
        uint2 readPos = min(pos + uint2(i % 4, i / 4), uint2(input.get_width() - 1, input.get_height() - 1));
        float4 value = input.read(readPos);
        pixels[i] = value.rgb;
        alpha[i] = value.a * 255.0;
    }

    uint blocksPerRow = (input.get_width() + 3) / 4;
    uint offset = (gid.y * blocksPerRow + gid.x) * (formatMode == 0 ? 8 : 16);
    if (formatMode >= 1) {
        AlphaBlockResult alphaBlock = optimizeAlpha(alpha);
        output[offset] = alphaBlock.a0;
        output[offset + 1] = alphaBlock.a1;
        for (uint i = 0; i < 6; i++) output[offset + 2 + i] = (uchar)((alphaBlock.indices >> (i * 8)) & 0xFF);
        offset += 8;
    }

    float3 mean = float3(0.0);
    for (uint i = 0; i < 16; i++) mean += pixels[i];
    mean /= 16.0;
    float3 axes[4] = {
        float3(1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, 0.0, 1.0),
        normalize(float3(0.2126, 0.7152, 0.0722))
    };
    ColorBlockResult best = ColorBlockResult{0, 1, 0, INFINITY};
    for (uint axisIndex = 0; axisIndex < 4; axisIndex++) {
        float3 axis = axes[axisIndex];
        float minProjection = INFINITY;
        float maxProjection = -INFINITY;
        float3 minPixel = pixels[0];
        float3 maxPixel = pixels[0];
        for (uint i = 0; i < 16; i++) {
            float projection = dot(pixels[i], axis);
            if (projection < minProjection) { minProjection = projection; minPixel = pixels[i]; }
            if (projection > maxProjection) { maxProjection = projection; maxPixel = pixels[i]; }
        }
        ColorBlockResult current = evaluateColor(pixels, maxPixel, minPixel);
        if (current.error < best.error
            || (current.error == best.error && current.c0 < best.c0)
            || (current.error == best.error && current.c0 == best.c0 && current.c1 < best.c1)) {
            best = current;
        }
    }
    output[offset] = best.c0 & 0xFF;
    output[offset + 1] = best.c0 >> 8;
    output[offset + 2] = best.c1 & 0xFF;
    output[offset + 3] = best.c1 >> 8;
    output[offset + 4] = best.indices & 0xFF;
    output[offset + 5] = (best.indices >> 8) & 0xFF;
    output[offset + 6] = (best.indices >> 16) & 0xFF;
    output[offset + 7] = (best.indices >> 24) & 0xFF;
}
"""

// FP8SR is deliberately kept as a small, fixed graph.  The source is
// compiled only on macOS 27+ and is never loaded by the normal DDS path.
// Activations and accumulators remain Float16; only the model weights are
// MetalFloat8E4M3 tensors.
let fp8TensorOpsSource = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

struct FP8SRIm2ColParams {
    uint width;
    uint height;
    uint sourceChannels;
    uint sourceStride;
    uint targetK;
    uint kernelSize;
    ulong baseIndex;
};

struct FP8SRPostParams {
    uint pixelCount;
    uint channels;
    float scale;
};

struct FP8SRMatmulParams {
    uint m;
    uint n;
    uint k;
};

struct FP8SRPixelParams {
    uint width;
    uint height;
    uint pixelCount;
    uint channels;
};

kernel void fp8sr_im2col_image(
    texture2d<float, access::read> source [[texture(0)]],
    device half *destination [[buffer(1)]],
    constant FP8SRIm2ColParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    ulong pixelCount = ulong(params.width) * ulong(params.height);
    ulong total = pixelCount * ulong(params.targetK);
    ulong logicalIndex = params.baseIndex + ulong(gid);
    if (logicalIndex >= total) return;
    uint pixel = uint(logicalIndex / ulong(params.targetK));
    uint feature = uint(logicalIndex % ulong(params.targetK));
    if (feature >= params.kernelSize * params.kernelSize * params.sourceChannels) {
        destination[logicalIndex] = half(0.0h);
        return;
    }
    uint inputChannel = feature % params.sourceChannels;
    uint kernelIndex = feature / params.sourceChannels;
    int x = int(pixel % params.width) + int(kernelIndex % params.kernelSize) - int(params.kernelSize / 2);
    int y = int(pixel / params.width) + int(kernelIndex / params.kernelSize) - int(params.kernelSize / 2);
    x = clamp(x, 0, int(params.width) - 1);
    y = clamp(y, 0, int(params.height) - 1);
    float4 value = source.read(uint2(x, y));
    destination[logicalIndex] = half(inputChannel == 0 ? value.r : (inputChannel == 1 ? value.g : value.b));
}

kernel void fp8sr_im2col_features(
    device const half *source [[buffer(0)]],
    device half *destination [[buffer(1)]],
    constant FP8SRIm2ColParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    ulong pixelCount = ulong(params.width) * ulong(params.height);
    ulong total = pixelCount * ulong(params.targetK);
    ulong logicalIndex = params.baseIndex + ulong(gid);
    if (logicalIndex >= total) return;
    uint pixel = uint(logicalIndex / ulong(params.targetK));
    uint feature = uint(logicalIndex % ulong(params.targetK));
    if (feature >= params.kernelSize * params.kernelSize * params.sourceChannels) {
        destination[logicalIndex] = half(0.0h);
        return;
    }
    uint inputChannel = feature % params.sourceChannels;
    uint kernelIndex = feature / params.sourceChannels;
    int x = int(pixel % params.width) + int(kernelIndex % params.kernelSize) - int(params.kernelSize / 2);
    int y = int(pixel / params.width) + int(kernelIndex / params.kernelSize) - int(params.kernelSize / 2);
    x = clamp(x, 0, int(params.width) - 1);
    y = clamp(y, 0, int(params.height) - 1);
    uint sourcePixel = uint(y) * params.width + uint(x);
    // TensorOps output tensors use [channels, pixels] extents, where the
    // innermost dimension is the channel.  Keep the intermediate buffer in
    // that physical pixel-major layout for the next im2col pass.
    destination[logicalIndex] = source[sourcePixel * params.sourceStride + inputChannel];
}

kernel void fp8sr_matmul(
    device half *activationBuffer [[buffer(0)]],
    device uchar *weightBuffer [[buffer(1)]],
    device half *outputBuffer [[buffer(2)]],
    constant FP8SRMatmulParams &params [[buffer(3)]],
    uint2 threadgroupID [[threadgroup_position_in_grid]]) {
    constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
        64, 32, 32, false, false, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto activation = tensor<device half, dextents<int, 2>, tensor_inline>(
        activationBuffer,
        dextents<int, 2>{int(params.k), int(params.m)},
        array<int, 2>{1, int(params.k)});
    auto weights = tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline>(
        weightBuffer,
        dextents<int, 2>{int(params.n), int(params.k)},
        array<int, 2>{1, 128});
    auto output = tensor<device half, dextents<int, 2>, tensor_inline>(
        outputBuffer,
        dextents<int, 2>{int(params.n), int(params.m)},
        array<int, 2>{1, int(params.n)});
    for (uint k = 0; k < params.k; k += 32) {
        auto activationChunk = activation.slice(k, threadgroupID.y * 64);
        auto weightChunk = weights.slice(threadgroupID.x * 32, k);
        auto outputTile = output.slice(threadgroupID.x * 32, threadgroupID.y * 64);
        operation.run(activationChunk, weightChunk, outputTile);
    }
}

kernel void fp8sr_matmul_fp16(
    device half *activationBuffer [[buffer(0)]],
    device uchar *weightBuffer [[buffer(1)]],
    device half *outputBuffer [[buffer(2)]],
    constant FP8SRMatmulParams &params [[buffer(3)]],
    uint2 threadgroupID [[threadgroup_position_in_grid]]) {
    constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
        64, 32, 32, false, false, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto activation = tensor<device half, dextents<int, 2>, tensor_inline>(
        activationBuffer,
        dextents<int, 2>{int(params.k), int(params.m)},
        array<int, 2>{1, int(params.k)});
    auto weights = tensor<device half, dextents<int, 2>, tensor_inline>(
        reinterpret_cast<device half *>(weightBuffer),
        dextents<int, 2>{int(params.n), int(params.k)},
        array<int, 2>{1, 64});
    auto output = tensor<device half, dextents<int, 2>, tensor_inline>(
        outputBuffer,
        dextents<int, 2>{int(params.n), int(params.m)},
        array<int, 2>{1, int(params.n)});
    for (uint k = 0; k < params.k; k += 32) {
        auto activationChunk = activation.slice(k, threadgroupID.y * 64);
        auto weightChunk = weights.slice(threadgroupID.x * 32, k);
        auto outputTile = output.slice(threadgroupID.x * 32, threadgroupID.y * 64);
        operation.run(activationChunk, weightChunk, outputTile);
    }
}

kernel void fp8sr_matmul_fp4(
    device half *activationBuffer [[buffer(0)]],
    device uchar *weightBuffer [[buffer(1)]],
    device half *outputBuffer [[buffer(2)]],
    constant FP8SRMatmulParams &params [[buffer(3)]],
    uint2 threadgroupID [[threadgroup_position_in_grid]]) {
    constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
        64, 32, 32, false, false, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto activation = tensor<device half, dextents<int, 2>, tensor_inline>(
        activationBuffer,
        dextents<int, 2>{int(params.k), int(params.m)},
        array<int, 2>{1, int(params.k)});
    auto weights = tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline>(
        weightBuffer,
        dextents<int, 2>{int(params.n), int(params.k)},
        array<int, 2>{1, 256});
    auto output = tensor<device half, dextents<int, 2>, tensor_inline>(
        outputBuffer,
        dextents<int, 2>{int(params.n), int(params.m)},
        array<int, 2>{1, int(params.n)});
    for (uint k = 0; k < params.k; k += 32) {
        auto activationChunk = activation.slice(k, threadgroupID.y * 64);
        auto weightChunk = weights.slice(threadgroupID.x * 32, k);
        auto outputTile = output.slice(threadgroupID.x * 32, threadgroupID.y * 64);
        operation.run(activationChunk, weightChunk, outputTile);
    }
}

kernel void fp8sr_matmul_int2(
    device half *activationBuffer [[buffer(0)]],
    device uchar *weightBuffer [[buffer(1)]],
    device half *outputBuffer [[buffer(2)]],
    constant FP8SRMatmulParams &params [[buffer(3)]],
    uint2 threadgroupID [[threadgroup_position_in_grid]]) {
    constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
        64, 32, 32, false, false, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto activation = tensor<device half, dextents<int, 2>, tensor_inline>(
        activationBuffer,
        dextents<int, 2>{int(params.k), int(params.m)},
        array<int, 2>{1, int(params.k)});
    auto weights = tensor<device int2b_format, dextents<int, 2>, tensor_inline>(
        weightBuffer,
        dextents<int, 2>{int(params.n), int(params.k)},
        array<int, 2>{1, 512});
    auto output = tensor<device half, dextents<int, 2>, tensor_inline>(
        outputBuffer,
        dextents<int, 2>{int(params.n), int(params.m)},
        array<int, 2>{1, int(params.n)});
    for (uint k = 0; k < params.k; k += 32) {
        auto activationChunk = activation.slice(k, threadgroupID.y * 64);
        auto weightChunk = weights.slice(threadgroupID.x * 32, k);
        auto outputTile = output.slice(threadgroupID.x * 32, threadgroupID.y * 64);
        operation.run(activationChunk, weightChunk, outputTile);
    }
}

kernel void fp8sr_postprocess(
    device half *values [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    constant FP8SRPostParams &params [[buffer(2)]],
    device atomic_uint *error [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = params.pixelCount * params.channels;
    if (gid >= total) return;
    uint channel = gid % params.channels;
    float value = float(values[gid]) * params.scale + float(bias[channel]);
    if (!isfinite(value)) {
        atomic_store_explicit(error, 1u, memory_order_relaxed);
        value = 0.0f;
    }
    values[gid] = half(max(value, 0.0f));
}

kernel void fp8sr_pixel_shuffle(
    device const half *values [[buffer(0)]],
    device uchar *destination [[buffer(1)]],
    constant FP8SRPixelParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.pixelCount) return;
    uint x = gid % params.width;
    uint y = gid / params.width;
    for (uint dy = 0; dy < 2; ++dy) {
        for (uint dx = 0; dx < 2; ++dx) {
            uint outX = x * 2 + dx;
            uint outY = y * 2 + dy;
            uint outOffset = (outY * params.width * 2 + outX) * 4;
            uint subpixel = (dy * 2 + dx) * 3;
            for (uint channel = 0; channel < 3; ++channel) {
                float value = clamp(float(values[gid * params.channels + subpixel + channel]), 0.0f, 1.0f);
                destination[outOffset + channel] = uchar(round(value * 255.0f));
            }
            destination[outOffset + 3] = 255;
        }
    }
}
"""

class MetalCompressor {
    static let shared: MetalCompressor? = MetalCompressor()
    
    let dev: MTLDevice
    let pipe: MTLComputePipelineState
    let q: MTLCommandQueue
    let loader: MTKTextureLoader
    
    private init?() {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let lib = try? dev.makeLibrary(source: metalSource, options: nil),
              let fn = lib.makeFunction(name: "compressTexture"),
              let pipe = try? dev.makeComputePipelineState(function: fn),
              let q = dev.makeCommandQueue() else { return nil }
        self.dev = dev
        self.pipe = pipe
        self.q = q
        self.loader = MTKTextureLoader(device: dev)
    }
}

struct CachedMask {
    let image: CIImage
    let hasExplicitAlpha: Bool
}

// Memory Cache for mask images to eliminate repetitive disk I/O
let maskCacheLock = NSLock()
var maskCache: [String: CachedMask] = [:]

func imageHasExplicitAlpha(at url: URL) -> Bool? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        return nil
    }

    switch image.alphaInfo {
    case .first, .last, .premultipliedFirst, .premultipliedLast, .alphaOnly:
        return true
    case .none, .noneSkipFirst, .noneSkipLast:
        return false
    @unknown default:
        return nil
    }
}

@discardableResult
func writeDDS(_ data: Data, to outputPath: String) -> Bool {
    do {
        try data.write(
            to: URL(fileURLWithPath: outputPath),
            options: .atomic
        )
        return true
    } catch {
        reportError(
            "ASHelper: Failed to write DDS '\(outputPath)': "
            + error.localizedDescription
        )
        return false
    }
}

func getRawRGBA(cgImage: CGImage) -> [UInt8] {
    let w = cgImage.width; let h = cgImage.height
    var raw = [UInt8](repeating: 0, count: w * h * 4)
    let ctx = CGContext(data: &raw, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    ctx?.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h)); return raw
}

func unpremultiplyRGBA(_ raw: inout [UInt8]) {
    guard raw.count >= 4 else { return }
    for offset in stride(from: 0, to: raw.count - 3, by: 4) {
        let alpha = Int(raw[offset + 3])
        guard alpha > 0, alpha < 255 else {
            if alpha == 0 {
                raw[offset] = 0
                raw[offset + 1] = 0
                raw[offset + 2] = 0
            }
            continue
        }
        raw[offset] = UInt8(min(255, (Int(raw[offset]) * 255 + alpha / 2) / alpha))
        raw[offset + 1] = UInt8(min(255, (Int(raw[offset + 1]) * 255 + alpha / 2) / alpha))
        raw[offset + 2] = UInt8(min(255, (Int(raw[offset + 2]) * 255 + alpha / 2) / alpha))
    }
}

func premultiplyRGBA(_ raw: inout [UInt8]) {
    guard raw.count >= 4 else { return }
    for offset in stride(from: 0, to: raw.count - 3, by: 4) {
        let alpha = Int(raw[offset + 3])
        guard alpha < 255 else { continue }
        raw[offset] = UInt8((Int(raw[offset]) * alpha + 127) / 255)
        raw[offset + 1] = UInt8((Int(raw[offset + 1]) * alpha + 127) / 255)
        raw[offset + 2] = UInt8((Int(raw[offset + 2]) * alpha + 127) / 255)
    }
}

// MetalFX consumes an opaque RGB image in this path.  Keep a separate
// unassociated-alpha conversion for RGBA inputs so transparent RGB values are
// not darkened by Core Graphics premultiplication before they reach MetalFX.
// CGContext does not accept CGImageAlphaInfo.last with a DeviceRGB color
// space. Render into a valid premultiplied buffer, then restore straight RGB.
func getRawRGBAUnassociated(cgImage: CGImage) -> [UInt8]? {
    let width = cgImage.width
    let height = cgImage.height
    guard width > 0, height > 0,
          let byteCount = try? checkedMultiply(width, height, 4, label: "rgba_bytes") else {
        return nil
    }
    var raw = [UInt8](repeating: 0, count: byteCount)
    guard let context = CGContext(
        data: &raw,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    unpremultiplyRGBA(&raw)
    return raw
}

func cgImageFromRGBA(_ input: [UInt8], width: Int, height: Int) -> CGImage? {
    var raw = input
    guard let context = CGContext(
        data: &raw,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }
    return context.makeImage()
}

func cgImageFromRGBAUnassociated(_ input: [UInt8], width: Int, height: Int) -> CGImage? {
    guard width > 0, height > 0,
          let byteCount = try? checkedMultiply(width, height, 4, label: "rgba_bytes"),
          input.count == byteCount else { return nil }
    var premultiplied = input
    premultiplyRGBA(&premultiplied)
    return cgImageFromRGBA(premultiplied, width: width, height: height)
}

func writePNG(_ raw: [UInt8], width: Int, height: Int, outputPath: String) -> Bool {
    guard width > 0, height > 0 else { return false }
    let pixelProduct = width.multipliedReportingOverflow(by: height)
    guard !pixelProduct.overflow else { return false }
    let byteProduct = pixelProduct.partialValue.multipliedReportingOverflow(by: 4)
    guard !byteProduct.overflow,
          raw.count == byteProduct.partialValue,
          let image = cgImageFromRGBA(raw, width: width, height: height) else {
        return false
    }
    let destinationURL = URL(fileURLWithPath: outputPath)
    do {
        try FileManager.default.createDirectory(
            at: destinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
    } catch {
        return false
    }
    guard let destination = CGImageDestinationCreateWithURL(
        destinationURL as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
    ) else { return false }
    CGImageDestinationAddImage(destination, image, nil)
    return CGImageDestinationFinalize(destination)
}

func writePNGUnassociated(_ raw: [UInt8], width: Int, height: Int, outputPath: String) -> Bool {
    guard width > 0, height > 0,
          let byteCount = try? checkedMultiply(width, height, 4, label: "rgba_bytes"),
          raw.count == byteCount,
          let image = cgImageFromRGBAUnassociated(raw, width: width, height: height) else {
        return false
    }
    let destinationURL = URL(fileURLWithPath: outputPath)
    do {
        try FileManager.default.createDirectory(
            at: destinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
    } catch {
        return false
    }
    guard let destination = CGImageDestinationCreateWithURL(
        destinationURL as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
    ) else { return false }
    CGImageDestinationAddImage(destination, image, nil)
    return CGImageDestinationFinalize(destination)
}

func cpuPreprocessedImage(
    sourceImage: CGImage,
    maskPath: String,
    r: Double,
    g: Double,
    b: Double,
    contrast: Double,
    brightness: Double,
    saturation: Double
) -> CGImage? {
    let width = sourceImage.width
    let height = sourceImage.height
    var output = getRawRGBA(cgImage: sourceImage)
    var maskRaw: [UInt8]? = nil
    var maskWidth = 0
    var maskHeight = 0
    var maskHasExplicitAlpha = false

    if !maskPath.isEmpty && maskPath != "none" {
        let maskURL = URL(fileURLWithPath: maskPath)
        guard let maskSource = CGImageSourceCreateWithURL(maskURL as CFURL, nil),
              let maskImage = CGImageSourceCreateImageAtIndex(maskSource, 0, nil),
              let hasExplicitAlpha = imageHasExplicitAlpha(at: maskURL) else {
            return nil
        }
        maskRaw = getRawRGBA(cgImage: maskImage)
        maskWidth = maskImage.width
        maskHeight = maskImage.height
        maskHasExplicitAlpha = hasExplicitAlpha
    }

    func clampedUnit(_ value: Double) -> Double {
        min(1.0, max(0.0, value))
    }

    for y in 0..<height {
        for x in 0..<width {
            let offset = (y * width + x) * 4
            let alphaBeforeMask = Double(output[offset + 3]) / 255.0
            var alpha = alphaBeforeMask
            if let maskRaw, maskWidth > 0, maskHeight > 0 {
                let maskX = min(maskWidth - 1, x * maskWidth / width)
                let maskY = min(maskHeight - 1, y * maskHeight / height)
                let maskOffset = (maskY * maskWidth + maskX) * 4
                let maskValue: Double
                if maskHasExplicitAlpha {
                    maskValue = Double(maskRaw[maskOffset + 3]) / 255.0
                } else {
                    maskValue = (
                        0.2126 * Double(maskRaw[maskOffset])
                        + 0.7152 * Double(maskRaw[maskOffset + 1])
                        + 0.0722 * Double(maskRaw[maskOffset + 2])
                    ) / 255.0
                }
                alpha *= clampedUnit(maskValue)
            }

            // getRawRGBA returns premultiplied RGB. Unpremultiply before
            // applying color controls, then restore premultiplication.
            let unpremultiply = alphaBeforeMask > 0.000001
                ? 1.0 / alphaBeforeMask
                : 0.0
            var red = Double(output[offset]) / 255.0 * unpremultiply
            var green = Double(output[offset + 1]) / 255.0 * unpremultiply
            var blue = Double(output[offset + 2]) / 255.0 * unpremultiply

            red *= r
            green *= g
            blue *= b
            if contrast != 1.0 {
                red = (red - 0.5) * contrast + 0.5
                green = (green - 0.5) * contrast + 0.5
                blue = (blue - 0.5) * contrast + 0.5
            }
            if brightness != 0.0 {
                red += brightness
                green += brightness
                blue += brightness
            }
            if saturation != 1.0 {
                let luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                red = luminance + (red - luminance) * saturation
                green = luminance + (green - luminance) * saturation
                blue = luminance + (blue - luminance) * saturation
            }

            let premultiply = clampedUnit(alpha)
            output[offset] = UInt8(clampedUnit(red) * premultiply * 255.0)
            output[offset + 1] = UInt8(clampedUnit(green) * premultiply * 255.0)
            output[offset + 2] = UInt8(clampedUnit(blue) * premultiply * 255.0)
            output[offset + 3] = UInt8(premultiply * 255.0)
        }
    }
    return cgImageFromRGBA(output, width: width, height: height)
}

struct CPUColorBlockResult {
    let c0: UInt16
    let c1: UInt16
    let indices: UInt32
    let error: Double
}

struct CPUAlphaBlockResult {
    let a0: UInt8
    let a1: UInt8
    let indices: UInt64
    let error: Double
}

func clampUnit(_ value: Double) -> Double {
    min(1.0, max(0.0, value))
}

func normalizeVector(_ value: SIMD3<Double>) -> SIMD3<Double> {
    let length = sqrt(value.x * value.x + value.y * value.y + value.z * value.z)
    return length > 0.0000001 ? value / length : SIMD3<Double>(1.0, 0.0, 0.0)
}

func pack565(_ color: SIMD3<Double>) -> UInt16 {
    let r = UInt16(clampUnit(color.x) * 31.0)
    let g = UInt16(clampUnit(color.y) * 63.0)
    let b = UInt16(clampUnit(color.z) * 31.0)
    return (r << 11) | (g << 5) | b
}

func canonical565(_ first: UInt16, _ second: UInt16) -> (UInt16, UInt16) {
    if first > second { return (first, second) }
    if first < second { return (second, first) }
    if first < UInt16.max { return (first + 1, second) }
    if second > 0 { return (first, second - 1) }
    return (UInt16.max, 0)
}

func unpack565(_ value: UInt16) -> SIMD3<Double> {
    SIMD3<Double>(
        Double((value >> 11) & 31) / 31.0,
        Double((value >> 5) & 63) / 63.0,
        Double(value & 31) / 31.0
    )
}

func colorPalette565(_ c0: UInt16, _ c1: UInt16) -> [SIMD3<Double>] {
    let p0 = unpack565(c0)
    let p1 = unpack565(c1)
    return [
        p0,
        p1,
        (p0 * 2.0 + p1) / 3.0,
        (p0 + p1 * 2.0) / 3.0,
    ]
}

func perceptualColorError(_ lhs: SIMD3<Double>, _ rhs: SIMD3<Double>) -> Double {
    let delta = lhs - rhs
    return delta.x * delta.x * 0.2126
        + delta.y * delta.y * 0.7152
        + delta.z * delta.z * 0.0722
}

func assignColorIndices(
    _ pixels: [SIMD3<Double>],
    _ palette: [SIMD3<Double>]
) -> (UInt32, Double) {
    var indices: UInt32 = 0
    var error = 0.0
    for (pixelIndex, pixel) in pixels.enumerated() {
        var bestIndex: UInt32 = 0
        var bestError = perceptualColorError(pixel, palette[0])
        for candidate in 1..<palette.count {
            let candidateError = perceptualColorError(pixel, palette[candidate])
            if candidateError < bestError {
                bestError = candidateError
                bestIndex = UInt32(candidate)
            }
        }
        indices |= bestIndex << UInt32(pixelIndex * 2)
        error += bestError
    }
    return (indices, error)
}

func refitColorEndpoints(
    _ pixels: [SIMD3<Double>],
    _ indices: UInt32
) -> (SIMD3<Double>, SIMD3<Double>)? {
    var aa = 0.0
    var ab = 0.0
    var bb = 0.0
    var ad = SIMD3<Double>(repeating: 0.0)
    var bd = SIMD3<Double>(repeating: 0.0)
    for (index, pixel) in pixels.enumerated() {
        let paletteIndex = Int((indices >> UInt32(index * 2)) & 3)
        let a = 1.0 - Double(paletteIndex) / 3.0
        let b = Double(paletteIndex) / 3.0
        aa += a * a
        ab += a * b
        bb += b * b
        ad += pixel * a
        bd += pixel * b
    }
    let determinant = aa * bb - ab * ab
    guard abs(determinant) > 0.000001 else { return nil }
    let endpoint0 = (ad * bb - bd * ab) / determinant
    let endpoint1 = (bd * aa - ad * ab) / determinant
    return (
        SIMD3<Double>(
            clampUnit(endpoint0.x),
            clampUnit(endpoint0.y),
            clampUnit(endpoint0.z)
        ),
        SIMD3<Double>(
            clampUnit(endpoint1.x),
            clampUnit(endpoint1.y),
            clampUnit(endpoint1.z)
        )
    )
}

func encodeColorBlock(_ pixels: [SIMD3<Double>]) -> CPUColorBlockResult {
    let axes = [
        SIMD3<Double>(1.0, 0.0, 0.0),
        SIMD3<Double>(0.0, 1.0, 0.0),
        SIMD3<Double>(0.0, 0.0, 1.0),
        normalizeVector(SIMD3<Double>(0.2126, 0.7152, 0.0722)),
    ]
    var best = CPUColorBlockResult(c0: 0, c1: 1, indices: 0, error: Double.infinity)
    for axis in axes {
        var lowProjection = Double.infinity
        var highProjection = -Double.infinity
        var lowPixel = pixels[0]
        var highPixel = pixels[0]
        for pixel in pixels {
            let projection = pixel.x * axis.x + pixel.y * axis.y + pixel.z * axis.z
            if projection < lowProjection { lowProjection = projection; lowPixel = pixel }
            if projection > highProjection { highProjection = projection; highPixel = pixel }
        }
        var endpoint0 = highPixel
        var endpoint1 = lowPixel
        for _ in 0..<8 {
            let packed = canonical565(pack565(endpoint0), pack565(endpoint1))
            let assigned = assignColorIndices(pixels, colorPalette565(packed.0, packed.1))
            if let refit = refitColorEndpoints(pixels, assigned.0) {
                endpoint0 = refit.0
                endpoint1 = refit.1
            }
        }
        let packed = canonical565(pack565(endpoint0), pack565(endpoint1))
        let assigned = assignColorIndices(pixels, colorPalette565(packed.0, packed.1))
        let result = CPUColorBlockResult(
            c0: packed.0,
            c1: packed.1,
            indices: assigned.0,
            error: assigned.1
        )
        if result.error < best.error
            || (result.error == best.error && result.c0 < best.c0)
            || (result.error == best.error && result.c0 == best.c0 && result.c1 < best.c1) {
            best = result
        }
    }
    return best
}

func alphaPalette(_ a0: UInt8, _ a1: UInt8) -> [Double] {
    let first = Double(a0)
    let second = Double(a1)
    if a0 > a1 {
        return [
            first, second,
            (6.0 * first + second) / 7.0,
            (5.0 * first + 2.0 * second) / 7.0,
            (4.0 * first + 3.0 * second) / 7.0,
            (3.0 * first + 4.0 * second) / 7.0,
            (2.0 * first + 5.0 * second) / 7.0,
            (first + 6.0 * second) / 7.0,
        ]
    }
    return [
        first, second,
        (4.0 * first + second) / 5.0,
        (3.0 * first + 2.0 * second) / 5.0,
        (2.0 * first + 3.0 * second) / 5.0,
        (first + 4.0 * second) / 5.0,
        0.0, 255.0,
    ]
}

func evaluateAlphaBlock(_ values: [Double], _ a0: UInt8, _ a1: UInt8) -> CPUAlphaBlockResult {
    let palette = alphaPalette(a0, a1)
    var indices: UInt64 = 0
    var error = 0.0
    for (pixelIndex, value) in values.enumerated() {
        var bestIndex: UInt64 = 0
        var bestError = abs(value - palette[0])
        for candidate in 1..<palette.count {
            let candidateError = abs(value - palette[candidate])
            if candidateError < bestError {
                bestError = candidateError
                bestIndex = UInt64(candidate)
            }
        }
        indices |= bestIndex << UInt64(pixelIndex * 3)
        error += bestError * bestError
    }
    return CPUAlphaBlockResult(a0: a0, a1: a1, indices: indices, error: error)
}

func isBetterAlphaBlock(_ candidate: CPUAlphaBlockResult, than current: CPUAlphaBlockResult) -> Bool {
    candidate.error < current.error
        || (candidate.error == current.error && candidate.a0 < current.a0)
        || (
            candidate.error == current.error
            && candidate.a0 == current.a0
            && candidate.a1 < current.a1
        )
}

func encodeAlphaBlock(_ values: [Double]) -> CPUAlphaBlockResult {
    var sourceCandidates: [UInt8] = []
    for value in values {
        let clamped = UInt8(min(255.0, max(0.0, value)))
        if !sourceCandidates.contains(clamped) {
            sourceCandidates.append(clamped)
        }
    }

    // Most terrain masks contain large constant or binary 4x4 regions.  Those
    // cases can be represented exactly by their endpoint values and avoid an
    // otherwise quadratic search over the 18-value candidate set.
    if sourceCandidates.count == 1 {
        let value = sourceCandidates[0]
        return evaluateAlphaBlock(values, 0, value == 0 ? 255 : value)
    }
    if sourceCandidates.count == 2 {
        let low = min(sourceCandidates[0], sourceCandidates[1])
        let high = max(sourceCandidates[0], sourceCandidates[1])
        var best = evaluateAlphaBlock(values, low, high)
        if high == 255 && low != 0 {
            let alternate = evaluateAlphaBlock(values, 0, low)
            if isBetterAlphaBlock(alternate, than: best) {
                best = alternate
            }
        }
        return best
    }

    // Keep the same first-seen candidate order as the Metal implementation.
    // The final tie-break is deterministic, but matching enumeration order
    // also keeps floating-point ties consistent across GPU and CPU paths.
    var candidates = sourceCandidates
    for value in [UInt8(0), UInt8(255)] where !candidates.contains(value) {
        candidates.append(value)
    }
    var best = CPUAlphaBlockResult(a0: 0, a1: 1, indices: 0, error: Double.infinity)
    for first in candidates {
        for second in candidates where first != second {
            let result = evaluateAlphaBlock(values, first, second)
            if isBetterAlphaBlock(result, than: best) {
                best = result
            }
        }
    }
    return best
}

func encodeRawBCImage(_ raw: [UInt8], width: Int, height: Int, mode: UInt32) -> Data {
    let blocksWide = (width + 3) / 4
    let blocksHigh = (height + 3) / 4
    var output = Data()
    for blockY in 0..<blocksHigh {
        for blockX in 0..<blocksWide {
            var colors: [SIMD3<Double>] = []
            var alphas: [Double] = []
            for pixelIndex in 0..<16 {
                let x = min(width - 1, blockX * 4 + pixelIndex % 4)
                let y = min(height - 1, blockY * 4 + pixelIndex / 4)
                let offset = (y * width + x) * 4
                colors.append(SIMD3<Double>(
                    Double(raw[offset]) / 255.0,
                    Double(raw[offset + 1]) / 255.0,
                    Double(raw[offset + 2]) / 255.0
                ))
                alphas.append(Double(raw[offset + 3]))
            }
            if mode >= 1 {
                let alpha = encodeAlphaBlock(alphas)
                var block = [UInt8](repeating: 0, count: 8)
                block[0] = alpha.a0
                block[1] = alpha.a1
                for byteIndex in 0..<6 {
                    block[byteIndex + 2] = UInt8((alpha.indices >> UInt64(byteIndex * 8)) & 0xff)
                }
                output.append(contentsOf: block)
            }
            let color = encodeColorBlock(colors)
            var block = [UInt8](repeating: 0, count: 8)
            block[0] = UInt8(color.c0 & 0xff)
            block[1] = UInt8(color.c0 >> 8)
            block[2] = UInt8(color.c1 & 0xff)
            block[3] = UInt8(color.c1 >> 8)
            block[4] = UInt8(color.indices & 0xff)
            block[5] = UInt8((color.indices >> 8) & 0xff)
            block[6] = UInt8((color.indices >> 16) & 0xff)
            block[7] = UInt8((color.indices >> 24) & 0xff)
            output.append(contentsOf: block)
        }
    }
    return output
}

func resizeCGImage(_ image: CGImage, width: Int, height: Int) -> CGImage? {
    guard let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }
    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage()
}

func compressImageWithCPUMipmaps(_ image: CGImage, mode: UInt32) -> Data? {
    var current = image
    var output = Data()
    while true {
        output.append(encodeRawBCImage(
            getRawRGBA(cgImage: current),
            width: current.width,
            height: current.height,
            mode: mode
        ))
        if current.width == 1 && current.height == 1 { break }
        let nextWidth = max(1, current.width / 2)
        let nextHeight = max(1, current.height / 2)
        guard let next = resizeCGImage(current, width: nextWidth, height: nextHeight) else {
            return nil
        }
        current = next
    }
    return output
}

func compressWithPreprocessedCIImage(finalCI: CIImage, mode: UInt32, useGPU: Bool) -> Data? {
    guard let comp = MetalCompressor.shared else { return nil }
    let dev = comp.dev
    let pipe = comp.pipe
    let q = comp.q
    
    let bounds = finalCI.extent
    let w = Int(bounds.width)
    let h = Int(bounds.height)
    
    // 1. Create an empty MTLTexture in VRAM (with mipmapped = true)
    let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: true)
    desc.usage = [.shaderRead, .shaderWrite, .pixelFormatView]
    guard let mtlTexture = dev.makeTexture(descriptor: desc) else { return nil }
    let expectedMipCount = Int(floor(log2(Double(max(w, h))))) + 1
    guard mtlTexture.mipmapLevelCount == expectedMipCount else {
        reportError(
            "ASHelper: Metal texture mip count mismatch (expected \(expectedMipCount), got \(mtlTexture.mipmapLevelCount))."
        )
        return nil
    }
    
    // 2. Render preprocessed CIImage directly into the MTLTexture level 0 [100% Zero-Copy]
    // [Y-Flip Fix] CIImage has bottom-left origin, while MTLTexture has top-left origin.
    // We must vertically flip the image so it compiles right-side up in X-Plane.
    let flippedCI = finalCI
        .transformed(by: CGAffineTransform(scaleX: 1, y: -1))
        .transformed(by: CGAffineTransform(translationX: 0, y: bounds.height))
    
    let ctx = CIContext(options: [.useSoftwareRenderer: false])
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    ctx.render(flippedCI, to: mtlTexture, commandBuffer: nil, bounds: bounds, colorSpace: colorSpace)
    
    // 3. Generate lower mipmap levels directly inside VRAM using GPU Blit Encoder
    guard let cmb = q.makeCommandBuffer() else { return nil }
    guard let mipEncoder = cmb.makeBlitCommandEncoder() else {
        reportError("ASHelper: Failed to create the Metal mipmap blit encoder.")
        return nil
    }
    mipEncoder.generateMipmaps(for: mtlTexture)
    mipEncoder.endEncoding()
    
    // 4. Run Metal compute kernels to compress each mipmap level
    var buffers: [MTLBuffer] = []
    for level in 0..<mtlTexture.mipmapLevelCount {
        let lW = max(1, w >> level); let lH = max(1, h >> level); let bW = (lW + 3) / 4; let bH = (lH + 3) / 4
        let sz = bW * bH * (mode == 0 ? 8 : 16)
        guard let buf = dev.makeBuffer(length: sz, options: .storageModeShared),
              buf.length == sz else { return nil }
        buffers.append(buf)
        
        guard let enc = cmb.makeComputeCommandEncoder() else { return nil }
        var m = mode
        enc.setComputePipelineState(pipe)
        guard let view = mtlTexture.makeTextureView(pixelFormat: mtlTexture.pixelFormat, textureType: mtlTexture.textureType, levels: level..<level+1, slices: 0..<1) else { return nil }
        enc.setTexture(view, index: 0)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBytes(&m, length: 4, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: (bW + 15) / 16, height: (bH + 15) / 16, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
    
    cmb.commit()
    cmb.waitUntilCompleted()
    guard cmb.status == .completed else {
        reportError(
            "ASHelper: Metal preprocessing command buffer failed with status \(cmb.status.rawValue): "
            + (cmb.error?.localizedDescription ?? "unknown error")
        )
        return nil
    }
    if let error = cmb.error {
        reportError("ASHelper: Metal preprocessing command buffer error: \(error.localizedDescription)")
        return nil
    }

    var outData = Data()
    for buf in buffers {
        outData.append(Data(bytes: buf.contents(), count: buf.length))
    }
    let expectedSize = (0..<mtlTexture.mipmapLevelCount).reduce(0) { total, level in
        let lW = max(1, w >> level)
        let lH = max(1, h >> level)
        return total + ((lW + 3) / 4) * ((lH + 3) / 4) * (mode == 0 ? 8 : 16)
    }
    guard outData.count == expectedSize else {
        reportError(
            "ASHelper: Metal compression payload size mismatch (expected \(expectedSize), got \(outData.count))."
        )
        return nil
    }
    return outData
}

func appendCPUCompressedDDS(
    finalCI: CIImage,
    bounds: CGRect,
    mode: UInt32,
    out: inout Data,
    sourceImage: CGImage? = nil
) -> Bool {
    // Batch conversion can be deliberately forced onto the CPU when Metal is
    // unavailable or when the GPU path fails.  A GPU-only CIContext would
    // make that fallback fail before the CPU BC encoder is reached.
    let ctx = CIContext(options: nil)
    let cgImage: CGImage?
    if let sourceImage {
        cgImage = sourceImage
    } else {
        cgImage = ctx.createCGImage(finalCI, from: bounds)
        if cgImage == nil {
            reportError(
                "ASHelper: CPU Core Image rendering failed for extent "
                + "\(bounds.origin.x),\(bounds.origin.y) \(bounds.width)x\(bounds.height)."
            )
        }
    }
    guard let cgImage else { return false }
    guard let compressed = compressImageWithCPUMipmaps(cgImage, mode: mode) else {
        return false
    }
    out.append(compressed)
    return true
}

func compressWithMipmaps(cgImage: CGImage, mode: UInt32) -> Data? {
    guard let comp = MetalCompressor.shared else { return nil }
    let dev = comp.dev
    let pipe = comp.pipe
    let q = comp.q
    let loader = comp.loader
    
    let options: [MTKTextureLoader.Option: Any] = [
        .SRGB: false,
        .generateMipmaps: true,
        .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue | MTLTextureUsage.shaderWrite.rawValue | MTLTextureUsage.pixelFormatView.rawValue)
    ]
    
    guard let tex = try? loader.newTexture(cgImage: cgImage, options: options) else { return nil }
    let w = tex.width; let h = tex.height
    let expectedMipCount = Int(floor(log2(Double(max(w, h))))) + 1
    guard tex.mipmapLevelCount == expectedMipCount else {
        reportError(
            "ASHelper: Metal texture mip count mismatch (expected \(expectedMipCount), got \(tex.mipmapLevelCount))."
        )
        return nil
    }
    
    guard let cmb = q.makeCommandBuffer() else { return nil }
    var buffers: [MTLBuffer] = []
    
    for level in 0..<tex.mipmapLevelCount {
        let lW = max(1, w >> level); let lH = max(1, h >> level); let bW = (lW + 3) / 4; let bH = (lH + 3) / 4
        let sz = bW * bH * (mode == 0 ? 8 : 16)
        guard let buf = dev.makeBuffer(length: sz, options: .storageModeShared),
              buf.length == sz else { return nil }
        buffers.append(buf)
        
        guard let enc = cmb.makeComputeCommandEncoder() else { return nil }
        var m = mode
        enc.setComputePipelineState(pipe)
        guard let view = tex.makeTextureView(pixelFormat: tex.pixelFormat, textureType: tex.textureType, levels: level..<level+1, slices: 0..<1) else { return nil }
        enc.setTexture(view, index: 0)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBytes(&m, length: 4, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: (bW + 15) / 16, height: (bH + 15) / 16, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
    
    cmb.commit()
    cmb.waitUntilCompleted()
    guard cmb.status == .completed else {
        reportError(
            "ASHelper: Metal mipmap command buffer failed with status \(cmb.status.rawValue): "
            + (cmb.error?.localizedDescription ?? "unknown error")
        )
        return nil
    }
    if let error = cmb.error {
        reportError("ASHelper: Metal mipmap command buffer error: \(error.localizedDescription)")
        return nil
    }
    
    var outData = Data()
    for buf in buffers {
        outData.append(Data(bytes: buf.contents(), count: buf.length))
    }
    let expectedSize = (0..<tex.mipmapLevelCount).reduce(0) { total, level in
        let lW = max(1, w >> level)
        let lH = max(1, h >> level)
        return total + ((lW + 3) / 4) * ((lH + 3) / 4) * (mode == 0 ? 8 : 16)
    }
    guard outData.count == expectedSize else {
        reportError(
            "ASHelper: Metal mipmap payload size mismatch (expected \(expectedSize), got \(outData.count))."
        )
        return nil
    }
    return outData
}

func convertCGImageWithPreprocess(
    sourceImage: CGImage,
    sourceLabel: String,
    maskPath: String,
    r: Double,
    g: Double,
    b: Double,
    contrast: Double,
    brightness: Double,
    saturation: Double,
    outputPath: String,
    format: String,
    useGPU: Bool
) -> Bool {
    guard format == "BC1" || format == "BC3" else {
        reportError("ASHelper does not support \(format) output. Use nvcompress instead.")
        return false
    }
    // Construct CIImage from the decoded CGImage.  This keeps the CPU batch
    // fallback independent of Image I/O provider-backed CIImage rendering.
    let srcCI = CIImage(cgImage: sourceImage)
    var finalCI = srcCI
    
    // 1. Blend mask if present (via in-memory Mask Cache)
    if maskPath != "none" && maskPath != "" {
        let maskURL = URL(fileURLWithPath: maskPath)
        var cachedMask: CachedMask? = nil
        maskCacheLock.lock()
        if let cached = maskCache[maskPath] {
            cachedMask = cached
            maskCacheLock.unlock()
        } else {
            maskCacheLock.unlock()
            if FileManager.default.fileExists(atPath: maskPath),
               let maskSource = CGImageSourceCreateWithURL(maskURL as CFURL, nil),
               let maskCGImage = CGImageSourceCreateImageAtIndex(maskSource, 0, nil),
               let hasExplicitAlpha = imageHasExplicitAlpha(at: maskURL) {
                let loaded = CIImage(cgImage: maskCGImage)
                let entry = CachedMask(image: loaded, hasExplicitAlpha: hasExplicitAlpha)
                maskCacheLock.lock()
                maskCache[maskPath] = entry
                cachedMask = entry
                maskCacheLock.unlock()
            }
        }

        guard let cachedMask else {
            reportError("ASHelper: Failed to load mask image '\(maskPath)'.")
            return false
        }

        let mCI = cachedMask.image
        guard mCI.extent.width > 0, mCI.extent.height > 0 else {
            reportError("ASHelper: Mask image '\(maskPath)' has an invalid extent.")
            return false
        }

        // Scale the mask to match the high-resolution source image extent,
        // preventing CIBlendWithAlphaMask from downscaling finalCI to the mask's low resolution.
        let scaleX = srcCI.extent.width / mCI.extent.width
        let scaleY = srcCI.extent.height / mCI.extent.height
        var resizedMask = mCI
        if scaleX != 1.0 || scaleY != 1.0 {
            guard let scaleFilter = CIFilter(name: "CILanczosScaleTransform") else {
                reportError("ASHelper: Failed to create the mask scaling filter.")
                return false
            }
            scaleFilter.setValue(mCI, forKey: kCIInputImageKey)
            scaleFilter.setValue(scaleX, forKey: kCIInputScaleKey)
            scaleFilter.setValue(scaleY / scaleX, forKey: "inputAspectRatio")
            guard let scaled = scaleFilter.outputImage else {
                reportError("ASHelper: Failed to scale mask image '\(maskPath)'.")
                return false
            }
            let originReset = scaled.transformed(by: CGAffineTransform(translationX: -scaled.extent.origin.x, y: -scaled.extent.origin.y))
            resizedMask = originReset.cropped(to: srcCI.extent)
        }

        let maskForBlend: CIImage
        if cachedMask.hasExplicitAlpha {
            // RGBA (and alpha-only) masks already carry the control alpha.
            maskForBlend = resizedMask
        } else {
            // L/grayscale masks carry their control value as luminance, not alpha.
            guard let maskToAlpha = CIFilter(name: "CIMaskToAlpha") else {
                reportError("ASHelper: Failed to create the grayscale mask conversion filter.")
                return false
            }
            maskToAlpha.setValue(resizedMask, forKey: kCIInputImageKey)
            guard let convertedMask = maskToAlpha.outputImage else {
                reportError("ASHelper: Failed to convert grayscale mask '\(maskPath)' to alpha.")
                return false
            }
            maskForBlend = convertedMask
        }

        guard let blendFilter = CIFilter(name: "CIBlendWithAlphaMask") else {
            reportError("ASHelper: Failed to create the alpha mask blend filter.")
            return false
        }
        blendFilter.setValue(srcCI, forKey: kCIInputImageKey)
        blendFilter.setValue(maskForBlend, forKey: kCIInputMaskImageKey)
        guard let blended = blendFilter.outputImage else {
            reportError("ASHelper: Failed to apply mask image '\(maskPath)'.")
            return false
        }
        finalCI = blended
    }
    
    // 2. Color Balance (RGB Multiply)
    if r != 1.0 || g != 1.0 || b != 1.0 {
        let matrixFilter = CIFilter(name: "CIColorMatrix")!
        matrixFilter.setValue(finalCI, forKey: kCIInputImageKey)
        matrixFilter.setValue(CIVector(x: CGFloat(r), y: 0, z: 0, w: 0), forKey: "inputRVector")
        matrixFilter.setValue(CIVector(x: 0, y: CGFloat(g), z: 0, w: 0), forKey: "inputGVector")
        matrixFilter.setValue(CIVector(x: 0, y: 0, z: CGFloat(b), w: 0), forKey: "inputBVector")
        matrixFilter.setValue(CIVector(x: 0, y: 0, z: 0, w: 1), forKey: "inputAVector")
        if let matrixed = matrixFilter.outputImage {
            finalCI = matrixed
        }
    }
    
    // 3. Color Controls (Brightness, Contrast, Saturation)
    if contrast != 1.0 || brightness != 0.0 || saturation != 1.0 {
        let controlsFilter = CIFilter(name: "CIColorControls")!
        controlsFilter.setValue(finalCI, forKey: kCIInputImageKey)
        controlsFilter.setValue(CGFloat(contrast), forKey: kCIInputContrastKey)
        controlsFilter.setValue(CGFloat(brightness), forKey: kCIInputBrightnessKey)
        controlsFilter.setValue(CGFloat(saturation), forKey: kCIInputSaturationKey)
        if let controlled = controlsFilter.outputImage {
            finalCI = controlled
        }
    }
    
    // 4. Zero-Copy rendering and direct VRAM compression
    let bounds = finalCI.extent
    let w = Int(bounds.width)
    let h = Int(bounds.height)
    let isBC7 = (format == "BC7")
    let isBC3 = (format == "BC3")
    let formatCode: UInt32 = isBC7 ? 2 : (isBC3 ? 1 : 0)
    
    let mipCount = UInt32(floor(log2(Double(max(w, h)))) + 1)
    let sz = UInt32(((w + 3) / 4) * ((h + 3) / 4) * (formatCode == 0 ? 8 : 16))
    let hdr = DDSHeader(height: UInt32(h), width: UInt32(w), pitchOrLinearSize: sz, mipmapCount: mipCount, fourCC: isBC7 ? 0x30315844 : (isBC3 ? 0x35545844 : 0x31545844))
    var out = hdr.toData()
    if isBC7 { out.append(DDSHeaderDX10(dxgiFormat: 98).toData()) }
    var compressed = false
    if useGPU, let gData = compressWithPreprocessedCIImage(finalCI: finalCI, mode: formatCode, useGPU: useGPU) {
        out.append(gData)
        compressed = true
    }
    if !compressed {
        let canUseDecodedSourceDirectly =
            (maskPath == "none" || maskPath.isEmpty)
            && r == 1.0 && g == 1.0 && b == 1.0
            && contrast == 1.0 && brightness == 0.0 && saturation == 1.0
        let cpuImage = canUseDecodedSourceDirectly
            ? sourceImage
            : cpuPreprocessedImage(
                sourceImage: sourceImage,
                maskPath: maskPath,
                r: r,
                g: g,
                b: b,
                contrast: contrast,
                brightness: brightness,
                saturation: saturation
            )
        if !appendCPUCompressedDDS(
            finalCI: finalCI,
            bounds: bounds,
            mode: formatCode,
            out: &out,
            sourceImage: cpuImage
        ) {
            reportError("ASHelper: Failed to compress image '\(sourceLabel)'.")
            return false
        }
    }

    return writeDDS(out, to: outputPath)
}

func convertWithPreprocess(
    jpegPath: String,
    maskPath: String,
    r: Double,
    g: Double,
    b: Double,
    contrast: Double,
    brightness: Double,
    saturation: Double,
    outputPath: String,
    format: String,
    useGPU: Bool
) -> Bool {
    let jpegURL = URL(fileURLWithPath: jpegPath)
    guard let source = CGImageSourceCreateWithURL(jpegURL as CFURL, nil),
          let sourceImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        reportError("ASHelper: Failed to load source image '\(jpegPath)'.")
        return false
    }
    return convertCGImageWithPreprocess(
        sourceImage: sourceImage,
        sourceLabel: jpegPath,
        maskPath: maskPath,
        r: r,
        g: g,
        b: b,
        contrast: contrast,
        brightness: brightness,
        saturation: saturation,
        outputPath: outputPath,
        format: format,
        useGPU: useGPU
    )
}

func lanczosImage(inputPath: String) -> CGImage? {
    let url = URL(fileURLWithPath: inputPath)
    guard let ci = CIImage(contentsOf: url), let f = CIFilter(name: "CILanczosScaleTransform") else {
        return nil
    }
    f.setValue(ci, forKey: kCIInputImageKey); f.setValue(2.0, forKey: kCIInputScaleKey)
    guard let out = f.outputImage else { return nil }
    return CIContext(options: nil).createCGImage(out, from: out.extent)
}

func lanczosUpscale(inputPath: String, outputPath: String) -> Bool {
    guard let cg = lanczosImage(inputPath: inputPath) else {
        reportError("ASHelper: Failed to render Lanczos-upscaled image '\(inputPath)'.")
        return false
    }
    guard let dest = CGImageDestinationCreateWithURL(
        URL(fileURLWithPath: outputPath) as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
    ) else {
        reportError("ASHelper: Failed to create Lanczos-upscaled image output '\(outputPath)'.")
        return false
    }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else {
        reportError("ASHelper: Failed to write Lanczos-upscaled image '\(outputPath)'.")
        return false
    }
    return true
}

@available(macOS 13.0, *)
private enum MetalFXSpatialError: Error, CustomStringConvertible {
    case invalidInput
    case unsupported
    case execution(String)
    case output

    var description: String {
        switch self {
        case .invalidInput: return "invalid_input"
        case .unsupported: return "unsupported"
        case .execution(let message): return message
        case .output: return "output_write"
        }
    }
}

@available(macOS 13.0, *)
private struct MetalFXRawResult {
    let raw: [UInt8]
    let width: Int
    let height: Int
    let alphaMode: String
    let readbackMs: Double
}

@available(macOS 13.0, *)
private struct MetalFXScaleResult {
    let raw: [UInt8]
    let readbackMs: Double
}

@available(macOS 13.0, *)
private final class MetalFXSpatialRuntime {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let ciContext: CIContext

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalFXSpatialError.execution("metal_device")
        }
        guard MTLFXSpatialScalerDescriptor.supportsDevice(device) else {
            throw MetalFXSpatialError.unsupported
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalFXSpatialError.execution("command_queue")
        }
        self.device = device
        self.commandQueue = commandQueue
        self.ciContext = CIContext(mtlDevice: device, options: nil)
    }

    private func scaleOpaque(
        raw: [UInt8],
        width: Int,
        height: Int
    ) throws -> MetalFXScaleResult {
        guard width > 0, height > 0,
              let inputBytes = try? checkedMultiply(width, height, 4, label: "metalfx_input"),
              raw.count == inputBytes,
              let outputWidth = try? checkedMultiply(width, 2, label: "metalfx_output_width"),
              let outputHeight = try? checkedMultiply(height, 2, label: "metalfx_output_height"),
              let outputBytes = try? checkedMultiply(outputWidth, outputHeight, 4, label: "metalfx_output") else {
            throw MetalFXSpatialError.invalidInput
        }

        let descriptor = MTLFXSpatialScalerDescriptor()
        descriptor.colorTextureFormat = .rgba8Unorm
        descriptor.outputTextureFormat = .rgba8Unorm
        descriptor.inputWidth = width
        descriptor.inputHeight = height
        descriptor.outputWidth = outputWidth
        descriptor.outputHeight = outputHeight
        descriptor.colorProcessingMode = .perceptual
        guard let scaler = descriptor.makeSpatialScaler(device: device) else {
            throw MetalFXSpatialError.execution("scaler")
        }

        let inputDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        inputDescriptor.storageMode = .shared
        inputDescriptor.usage = scaler.colorTextureUsage
        guard let inputTexture = device.makeTexture(descriptor: inputDescriptor) else {
            throw MetalFXSpatialError.execution("input_texture")
        }
        raw.withUnsafeBytes { bytes in
            inputTexture.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: bytes.baseAddress!,
                bytesPerRow: width * 4
            )
        }

        let outputDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: outputWidth,
            height: outputHeight,
            mipmapped: false
        )
        outputDescriptor.storageMode = .private
        outputDescriptor.usage = scaler.outputTextureUsage
        guard let outputTexture = device.makeTexture(descriptor: outputDescriptor),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let readback = device.makeBuffer(length: outputBytes, options: .storageModeShared) else {
            throw MetalFXSpatialError.execution("output_resources")
        }

        scaler.colorTexture = inputTexture
        scaler.inputContentWidth = width
        scaler.inputContentHeight = height
        scaler.outputTexture = outputTexture
        scaler.encode(commandBuffer: commandBuffer)
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalFXSpatialError.execution("readback_encoder")
        }
        blit.copy(
            from: outputTexture,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOriginMake(0, 0, 0),
            sourceSize: MTLSizeMake(outputWidth, outputHeight, 1),
            to: readback,
            destinationOffset: 0,
            destinationBytesPerRow: outputWidth * 4,
            destinationBytesPerImage: outputBytes
        )
        blit.endEncoding()
        let readbackStarted = CFAbsoluteTimeGetCurrent()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw MetalFXSpatialError.execution(
                "command_buffer_" + (commandBuffer.error?.localizedDescription ?? "failed")
            )
        }
        let output = Array(UnsafeBufferPointer(
            start: readback.contents().assumingMemoryBound(to: UInt8.self),
            count: outputBytes
        ))
        return MetalFXScaleResult(
            raw: output,
            readbackMs: (CFAbsoluteTimeGetCurrent() - readbackStarted) * 1000.0
        )
    }

    private func scaleAlphaBicubic(
        _ alpha: [UInt8],
        width: Int,
        height: Int
    ) throws -> [UInt8] {
        guard let alphaBytes = try? checkedMultiply(width, height, label: "alpha_input"),
              alpha.count == alphaBytes else {
            throw MetalFXSpatialError.invalidInput
        }
        var sourceBytes = alpha
        guard let sourceContext = CGContext(
            data: &sourceBytes,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ), let sourceImage = sourceContext.makeImage() else {
            throw MetalFXSpatialError.execution("alpha_source")
        }
        let input = CIImage(cgImage: sourceImage)
        guard let filter = CIFilter(name: "CIBicubicScaleTransform") else {
            throw MetalFXSpatialError.execution("bicubic_filter")
        }
        filter.setValue(input, forKey: kCIInputImageKey)
        filter.setValue(2.0, forKey: kCIInputScaleKey)
        filter.setValue(1.0, forKey: kCIInputAspectRatioKey)
        guard let output = filter.outputImage,
              let outputWidth = try? checkedMultiply(width, 2, label: "alpha_output_width"),
              let outputHeight = try? checkedMultiply(height, 2, label: "alpha_output_height"),
              let rendered = ciContext.createCGImage(
                  output,
                  from: CGRect(x: 0, y: 0, width: outputWidth, height: outputHeight)
              ) else {
            throw MetalFXSpatialError.execution("alpha_render")
        }
        var result = [UInt8](repeating: 0, count: outputWidth * outputHeight)
        guard let outputContext = CGContext(
            data: &result,
            width: outputWidth,
            height: outputHeight,
            bitsPerComponent: 8,
            bytesPerRow: outputWidth,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            throw MetalFXSpatialError.execution("alpha_output")
        }
        outputContext.draw(rendered, in: CGRect(x: 0, y: 0, width: outputWidth, height: outputHeight))
        return result
    }

    func upscale(image: CGImage) throws -> (raw: [UInt8], alphaMode: String, readbackMs: Double) {
        let width = image.width
        let height = image.height
        guard let source = getRawRGBAUnassociated(cgImage: image),
              let pixelCount = try? checkedMultiply(width, height, label: "metalfx_pixels"),
              source.count == pixelCount * 4 else {
            throw MetalFXSpatialError.invalidInput
        }
        let hasAlpha = stride(from: 3, to: source.count, by: 4).contains { source[$0] < 255 }
        if !hasAlpha {
            let scaled = try scaleOpaque(raw: source, width: width, height: height)
            return (scaled.raw, "opaque", scaled.readbackMs)
        }

        var opaque = source
        for offset in stride(from: 3, to: opaque.count, by: 4) {
            opaque[offset] = 255
        }
        let rgbOutput = try scaleOpaque(raw: opaque, width: width, height: height)
        let alphaInput = stride(from: 3, to: source.count, by: 4).map { source[$0] }
        let alphaOutput = try scaleAlphaBicubic(alphaInput, width: width, height: height)
        guard rgbOutput.raw.count == alphaOutput.count * 4 else {
            throw MetalFXSpatialError.execution("alpha_size")
        }
        var combined = rgbOutput.raw
        for pixel in 0..<alphaOutput.count {
            combined[pixel * 4 + 3] = alphaOutput[pixel]
        }
        return (combined, "rgba_split_bicubic", rgbOutput.readbackMs)
    }

    func renderRaw(inputPath: String) throws -> MetalFXRawResult {
        let sourceURL = URL(fileURLWithPath: inputPath)
        guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw MetalFXSpatialError.execution("input_load")
        }
        let result = try upscale(image: image)
        return MetalFXRawResult(
            raw: result.raw,
            width: image.width * 2,
            height: image.height * 2,
            alphaMode: result.alphaMode,
            readbackMs: result.readbackMs
        )
    }

    func render(inputPath: String, outputPath: String) throws -> String {
        let result = try renderRaw(inputPath: inputPath)
        guard writePNGUnassociated(
            result.raw,
            width: result.width,
            height: result.height,
            outputPath: outputPath
        ) else {
            throw MetalFXSpatialError.output
        }
        return result.alphaMode
    }
}

@available(macOS 13.0, *)
private func metalFXSpatialProcess(
    runtime: MetalFXSpatialRuntime?,
    inputPath: String,
    outputPath: String,
    fallbackAlphaProcessing: Bool
) -> (success: Bool, effectiveBackend: String, alphaMode: String, reason: String?) {
    do {
        let activeRuntime = try runtime ?? MetalFXSpatialRuntime()
        let mode = try activeRuntime.render(inputPath: inputPath, outputPath: outputPath)
        return (true, "metalfx_spatial", mode, nil)
    } catch {
        let reason = String(describing: error)
        if fallbackAlphaProcessing && lanczosUpscale(inputPath: inputPath, outputPath: outputPath) {
            return (true, "ci_lanczos", "rgba_fallback", reason)
        }
        reportError("ASHelper: MetalFX Spatial failed input=\(inputPath) reason=\(reason)")
        return (false, "metalfx_spatial", "unknown", reason)
    }
}

@available(macOS 13.0, *)
func metalFXSpatialUpscale(inputPath: String, outputPath: String) -> Bool {
    let hasAlpha: Bool
    if let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: inputPath) as CFURL, nil),
       let image = CGImageSourceCreateImageAtIndex(source, 0, nil) {
        let raw = getRawRGBA(cgImage: image)
        let explicitAlpha: Bool
        switch image.alphaInfo {
        case .first, .last, .premultipliedFirst, .premultipliedLast, .alphaOnly:
            explicitAlpha = true
        case .none, .noneSkipFirst, .noneSkipLast:
            explicitAlpha = false
        @unknown default:
            explicitAlpha = false
        }
        hasAlpha = explicitAlpha
            || stride(from: 3, to: raw.count, by: 4).contains { raw[$0] < 255 }
    } else {
        hasAlpha = false
    }
    let result = metalFXSpatialProcess(
        runtime: nil,
        inputPath: inputPath,
        outputPath: outputPath,
        fallbackAlphaProcessing: hasAlpha
    )
    print(
        "backend=metalfx_spatial effective_backend=\(result.effectiveBackend) "
            + "dispatch=single alpha_mode=\(result.alphaMode)"
            + (result.reason.map { " fallback_reason=\($0)" } ?? "")
    )
    return result.success
}

@available(macOS 13.0, *)
func metalFXSpatialUpscaleBatch(pairs: [(String, String)]) -> Bool {
    guard !pairs.isEmpty else { return false }
    let runtime = try? MetalFXSpatialRuntime()
    var allSuccessful = true
    var successCount = 0
    var fallbackCount = 0
    var fallbackReasons: [String: Int] = [:]
    let started = CFAbsoluteTimeGetCurrent()
    for (index, pair) in pairs.enumerated() {
        let itemStarted = CFAbsoluteTimeGetCurrent()
        let result = metalFXSpatialProcess(
            runtime: runtime,
            inputPath: pair.0,
            outputPath: pair.1,
            fallbackAlphaProcessing: true
        )
        if result.success {
            successCount += 1
            if result.effectiveBackend == "ci_lanczos" { fallbackCount += 1 }
        } else {
            allSuccessful = false
        }
        if let reason = result.reason {
            fallbackReasons[reason, default: 0] += 1
        }
        let duration = (CFAbsoluteTimeGetCurrent() - itemStarted) * 1000.0
        print(
            "metalfx_batch_item=\(index + 1)/\(pairs.count) "
                + "backend=metalfx_spatial effective_backend=\(result.effectiveBackend) "
                + "dispatch=batch alpha_mode=\(result.alphaMode) "
                + "duration_ms=\(String(format: "%.2f", duration))"
                + (result.reason.map { " fallback_reason=\($0)" } ?? "")
        )
    }
    let duration = (CFAbsoluteTimeGetCurrent() - started) * 1000.0
    let reasonSummary = fallbackReasons.keys.sorted().map {
        "\($0):\(fallbackReasons[$0] ?? 0)"
    }.joined(separator: ",")
    let effectiveBackend = fallbackCount == 0
        ? "metalfx_spatial"
        : (fallbackCount == successCount ? "ci_lanczos" : "mixed")
    print(
        "backend=metalfx_spatial effective_backend=\(effectiveBackend) dispatch=batch "
            + "batch_tasks=\(pairs.count) batch_success=\(successCount) "
            + "batch_fallback=\(fallbackCount) duration_ms=\(String(format: "%.2f", duration))"
            + (reasonSummary.isEmpty ? "" : " fallback_reasons=\(reasonSummary)")
    )
    return allSuccessful
}

@available(macOS 13.0, *)
private struct MetalFXDirectDDSItemResult {
    let success: Bool
    let effectiveBackend: String
    let alphaMode: String
    let fallbackReason: String?
    let metalFXMs: Double
    let readbackMs: Double
    let ddsMs: Double
    let totalMs: Double
}

@available(macOS 13.0, *)
private func processMetalFXDirectDDSItem(
    item: MetalFXDirectDDSItem,
    runtime: MetalFXSpatialRuntime?
) -> MetalFXDirectDDSItemResult {
    let started = CFAbsoluteTimeGetCurrent()
    var metalFXMs = 0.0
    var readbackMs = 0.0
    var ddsMs = 0.0
    var alphaMode = "opaque"
    var fallbackReason: String?

    try? FileManager.default.removeItem(atPath: item.output)

    do {
        guard let runtime else {
            throw MetalFXSpatialError.execution("metal_runtime")
        }
        let metalFXStarted = CFAbsoluteTimeGetCurrent()
        let rawResult = try runtime.renderRaw(inputPath: item.input)
        metalFXMs = (CFAbsoluteTimeGetCurrent() - metalFXStarted) * 1000.0
        readbackMs = rawResult.readbackMs
        alphaMode = rawResult.alphaMode
        guard let outputImage = cgImageFromRGBAUnassociated(
            rawResult.raw,
            width: rawResult.width,
            height: rawResult.height
        ) else {
            throw MetalFXSpatialError.execution("output_image")
        }
        let ddsStarted = CFAbsoluteTimeGetCurrent()
        guard convertCGImageWithPreprocess(
            sourceImage: outputImage,
            sourceLabel: item.input,
            maskPath: item.mask,
            r: item.color.r,
            g: item.color.g,
            b: item.color.b,
            contrast: item.color.contrast,
            brightness: item.color.brightness,
            saturation: item.color.saturation,
            outputPath: item.output,
            format: item.format,
            useGPU: true
        ) else {
            throw MetalFXSpatialError.execution("dds_compress")
        }
        ddsMs = (CFAbsoluteTimeGetCurrent() - ddsStarted) * 1000.0
        return MetalFXDirectDDSItemResult(
            success: true,
            effectiveBackend: "metalfx_spatial",
            alphaMode: alphaMode,
            fallbackReason: nil,
            metalFXMs: metalFXMs,
            readbackMs: readbackMs,
            ddsMs: ddsMs,
            totalMs: (CFAbsoluteTimeGetCurrent() - started) * 1000.0
        )
    } catch {
        fallbackReason = String(describing: error)
    }

    let fallbackStarted = CFAbsoluteTimeGetCurrent()
    guard let fallbackImage = lanczosImage(inputPath: item.input),
          convertCGImageWithPreprocess(
              sourceImage: fallbackImage,
              sourceLabel: item.input,
              maskPath: item.mask,
              r: item.color.r,
              g: item.color.g,
              b: item.color.b,
              contrast: item.color.contrast,
              brightness: item.color.brightness,
              saturation: item.color.saturation,
              outputPath: item.output,
              format: item.format,
              useGPU: true
          ) else {
        return MetalFXDirectDDSItemResult(
            success: false,
            effectiveBackend: "failed",
            alphaMode: "unknown",
            fallbackReason: fallbackReason ?? "ci_lanczos_failed",
            metalFXMs: metalFXMs,
            readbackMs: readbackMs,
            ddsMs: 0.0,
            totalMs: (CFAbsoluteTimeGetCurrent() - started) * 1000.0
        )
    }
    ddsMs = (CFAbsoluteTimeGetCurrent() - fallbackStarted) * 1000.0
    return MetalFXDirectDDSItemResult(
        success: true,
        effectiveBackend: "ci_lanczos",
        alphaMode: "lanczos",
        fallbackReason: fallbackReason,
        metalFXMs: metalFXMs,
        readbackMs: readbackMs,
        ddsMs: ddsMs,
        totalMs: (CFAbsoluteTimeGetCurrent() - started) * 1000.0
    )
}

@available(macOS 13.0, *)
func metalFXSpatialDDSBatch(requestPath: String) -> Bool {
    do {
        let data = try Data(contentsOf: URL(fileURLWithPath: requestPath))
        let request = try JSONDecoder().decode(MetalFXDirectDDSRequest.self, from: data)
        guard request.version == 1, !request.items.isEmpty else {
            throw MetalFXSpatialError.invalidInput
        }
        guard request.items.allSatisfy({ $0.format == "BC1" || $0.format == "BC3" }) else {
            throw MetalFXSpatialError.execution("unsupported_format")
        }

        let runtime = try? MetalFXSpatialRuntime()
        var successCount = 0
        var fallbackCount = 0
        var failedCount = 0
        var fallbackReasons: [String: Int] = [:]
        let started = CFAbsoluteTimeGetCurrent()

        for (index, item) in request.items.enumerated() {
            let result = autoreleasepool {
                processMetalFXDirectDDSItem(item: item, runtime: runtime)
            }
            if result.success {
                successCount += 1
                if result.effectiveBackend == "ci_lanczos" {
                    fallbackCount += 1
                }
            } else {
                failedCount += 1
            }
            if let reason = result.fallbackReason {
                fallbackReasons[reason, default: 0] += 1
            }
            print(
                "metalfx_dds_item=\(index + 1)/\(request.items.count) "
                    + "backend=metalfx_spatial effective_backend=\(result.effectiveBackend) "
                    + "dispatch=direct_dds alpha_mode=\(result.alphaMode) "
                    + "metalfx_ms=\(String(format: "%.2f", result.metalFXMs)) "
                    + "readback_ms=\(String(format: "%.2f", result.readbackMs)) "
                    + "dds_ms=\(String(format: "%.2f", result.ddsMs)) "
                    + "total_ms=\(String(format: "%.2f", result.totalMs))"
                    + (result.fallbackReason.map { " fallback_reason=\($0)" } ?? "")
            )
        }

        let reasonSummary = fallbackReasons.keys.sorted().map {
            "\($0):\(fallbackReasons[$0] ?? 0)"
        }.joined(separator: ",")
        let effectiveBackend = fallbackCount == 0
            ? "metalfx_spatial"
            : (fallbackCount == successCount ? "ci_lanczos" : "mixed")
        print(
            "backend=metalfx_spatial effective_backend=\(effectiveBackend) dispatch=direct_dds "
                + "png_intermediate=false batch_tasks=\(request.items.count) "
                + "batch_success=\(successCount) batch_fallback=\(fallbackCount) "
                + "batch_failed=\(failedCount) duration_ms="
                + String(format: "%.2f", (CFAbsoluteTimeGetCurrent() - started) * 1000.0)
                + (reasonSummary.isEmpty ? "" : " fallback_reasons=\(reasonSummary)")
        )
        return failedCount == 0
    } catch {
        reportError("ASHelper: MetalFX direct DDS batch failed: \(error)")
        return false
    }
}

@available(macOS 27.0, *)
private struct TensorOpsDirectDDSItemResult {
    let success: Bool
    let effectiveBackend: String
    let tensorOpsDispatchObserved: Bool
    let alphaMode: String
    let fallbackReason: String?
    let tensorOpsMs: Double
    let readbackMs: Double
    let ddsMs: Double
    let totalMs: Double
    let rssMB: UInt64
}

@available(macOS 27.0, *)
private func processTensorOpsDirectDDSItem(
    item: MetalFXDirectDDSItem,
    runtime: FP8SRRuntime?
) -> TensorOpsDirectDDSItemResult {
    let started = CFAbsoluteTimeGetCurrent()
    var tensorOpsMs = 0.0
    var readbackMs = 0.0
    var ddsMs = 0.0
    var fallbackReason: String?
    try? FileManager.default.removeItem(atPath: item.output)

    do {
        guard let runtime else {
            throw FP8SRError.unavailable("tensorops_runtime")
        }
        let tensorOpsStarted = CFAbsoluteTimeGetCurrent()
        let rawResult = try runtime.upscaleRaw(inputPath: item.input)
        tensorOpsMs = max(
            rawResult.tensorOpsMs,
            (CFAbsoluteTimeGetCurrent() - tensorOpsStarted) * 1000.0
        )
        readbackMs = rawResult.readbackMs
        guard let outputImage = cgImageFromRGBAUnassociated(
            rawResult.raw,
            width: rawResult.width,
            height: rawResult.height
        ) else {
            throw FP8SRError.execution("output_image")
        }
        let ddsStarted = CFAbsoluteTimeGetCurrent()
        guard convertCGImageWithPreprocess(
            sourceImage: outputImage,
            sourceLabel: item.input,
            maskPath: item.mask,
            r: item.color.r,
            g: item.color.g,
            b: item.color.b,
            contrast: item.color.contrast,
            brightness: item.color.brightness,
            saturation: item.color.saturation,
            outputPath: item.output,
            format: item.format,
            useGPU: true
        ) else {
            throw FP8SRError.execution("dds_compress")
        }
        ddsMs = (CFAbsoluteTimeGetCurrent() - ddsStarted) * 1000.0
        return TensorOpsDirectDDSItemResult(
            success: true,
            effectiveBackend: "tensorops",
            tensorOpsDispatchObserved: true,
            alphaMode: "opaque",
            fallbackReason: nil,
            tensorOpsMs: tensorOpsMs,
            readbackMs: readbackMs,
            ddsMs: ddsMs,
            totalMs: (CFAbsoluteTimeGetCurrent() - started) * 1000.0,
            rssMB: residentMemoryMB()
        )
    } catch {
        fallbackReason = String(describing: error)
    }

    let fallbackStarted = CFAbsoluteTimeGetCurrent()
    guard let fallbackImage = lanczosImage(inputPath: item.input),
          convertCGImageWithPreprocess(
              sourceImage: fallbackImage,
              sourceLabel: item.input,
              maskPath: item.mask,
              r: item.color.r,
              g: item.color.g,
              b: item.color.b,
              contrast: item.color.contrast,
              brightness: item.color.brightness,
              saturation: item.color.saturation,
              outputPath: item.output,
              format: item.format,
              useGPU: true
          ) else {
        return TensorOpsDirectDDSItemResult(
            success: false,
            effectiveBackend: "failed",
            tensorOpsDispatchObserved: false,
            alphaMode: "unknown",
            fallbackReason: fallbackReason ?? "ci_lanczos_failed",
            tensorOpsMs: tensorOpsMs,
            readbackMs: readbackMs,
            ddsMs: 0.0,
            totalMs: (CFAbsoluteTimeGetCurrent() - started) * 1000.0,
            rssMB: residentMemoryMB()
        )
    }
    ddsMs = (CFAbsoluteTimeGetCurrent() - fallbackStarted) * 1000.0
    return TensorOpsDirectDDSItemResult(
        success: true,
        effectiveBackend: "ci_lanczos",
        tensorOpsDispatchObserved: false,
        alphaMode: "lanczos",
        fallbackReason: fallbackReason,
        tensorOpsMs: tensorOpsMs,
        readbackMs: readbackMs,
        ddsMs: ddsMs,
        totalMs: (CFAbsoluteTimeGetCurrent() - started) * 1000.0,
        rssMB: residentMemoryMB()
    )
}

@available(macOS 27.0, *)
func tensorOpsDirectDDSBatch(requestPath: String) -> Bool {
    do {
        let data = try Data(contentsOf: URL(fileURLWithPath: requestPath))
        let request = try JSONDecoder().decode(TensorOpsDirectDDSRequest.self, from: data)
        guard request.version == 1, !request.pack.isEmpty, !request.items.isEmpty else {
            throw FP8SRError.invalid("request")
        }
        guard request.items.allSatisfy({ $0.format == "BC1" || $0.format == "BC3" }) else {
            throw FP8SRError.invalid("unsupported_format")
        }
        let runtime = try? FP8SRRuntime.cached(packPath: request.pack)
        var successCount = 0
        var fallbackCount = 0
        var failedCount = 0
        var tensorOpsDispatchObserved = false
        var fallbackReasons: [String: Int] = [:]
        var peakRSSMB: UInt64 = 0
        let started = CFAbsoluteTimeGetCurrent()

        for (index, item) in request.items.enumerated() {
            let result = autoreleasepool {
                processTensorOpsDirectDDSItem(item: item, runtime: runtime)
            }
            peakRSSMB = max(peakRSSMB, result.rssMB)
            if result.success {
                successCount += 1
                tensorOpsDispatchObserved = tensorOpsDispatchObserved || result.tensorOpsDispatchObserved
                if result.effectiveBackend == "ci_lanczos" {
                    fallbackCount += 1
                }
            } else {
                failedCount += 1
            }
            if let reason = result.fallbackReason {
                fallbackReasons[reason, default: 0] += 1
            }
            let dtypeName = runtime?.weightDTypeName ?? "unknown"
            let tensorOpsMS = String(format: "%.2f", result.tensorOpsMs)
            let readbackMS = String(format: "%.2f", result.readbackMs)
            let ddsMS = String(format: "%.2f", result.ddsMs)
            let totalMS = String(format: "%.2f", result.totalMs)
            print(
                "tensorops_dds_item=\(index + 1)/\(request.items.count) "
                    + "backend=tensorops effective_backend=\(result.effectiveBackend) "
                    + "dispatch=direct_dds dtype=\(dtypeName) "
                    + "alpha_mode=\(result.alphaMode) "
                    + "activation_dtype=Float16 accumulation_dtype=Float16 "
                    + "png_intermediate=false tensorops_dispatch_observed=\(result.tensorOpsDispatchObserved) "
                    + "neural_accelerator_confirmed=false "
                    + "tensorops_ms=\(tensorOpsMS) readback_ms=\(readbackMS) "
                    + "dds_ms=\(ddsMS) total_ms=\(totalMS) "
                    + "rss_mb=\(result.rssMB)"
                    + (result.fallbackReason.map { " fallback_reason=\($0)" } ?? "")
            )
        }

        let reasonSummary = fallbackReasons.keys.sorted().map {
            "\($0):\(fallbackReasons[$0] ?? 0)"
        }.joined(separator: ",")
        print(
            "backend=tensorops effective_backend=tensorops dispatch=direct_dds "
                + "png_intermediate=false batch_tasks=\(request.items.count) "
                + "batch_success=\(successCount) batch_fallback=\(fallbackCount) "
                + "batch_failed=\(failedCount) batch_workers=1 batch_chunks=1 chunk_size=\(request.items.count) "
                + "tensorops_dispatch_observed=\(tensorOpsDispatchObserved) neural_accelerator_confirmed=false "
                + "peak_rss_mb=\(peakRSSMB) duration_ms="
                + String(format: "%.2f", (CFAbsoluteTimeGetCurrent() - started) * 1000.0)
                + (reasonSummary.isEmpty ? "" : " fallback_reasons=\(reasonSummary)")
        )
        return failedCount == 0
    } catch {
        reportError("ASHelper: TensorOps direct DDS batch failed: \(error)")
        return false
    }
}

func metalFXSpatialAvailable() -> Bool {
    guard #available(macOS 13.0, *),
          let device = MTLCreateSystemDefaultDevice() else {
        return false
    }
    return MTLFXSpatialScalerDescriptor.supportsDevice(device)
}

func fp8TensorOpsAvailable() -> Bool {
    guard #available(macOS 27.0, *),
          let device = MTLCreateSystemDefaultDevice(),
          device.makeMTL4CommandQueue() != nil,
          device.makeCommandAllocator() != nil,
          device.makeSharedEvent() != nil,
          let compiler = try? device.makeCompiler(descriptor: MTL4CompilerDescriptor()) else {
        return false
    }

    let compileOptions = MTLCompileOptions()
    compileOptions.languageVersion = .version4_1
    guard let library = try? device.makeLibrary(
        source: fp8TensorOpsSource,
        options: compileOptions
    ) else {
        return false
    }

    func makePipeline(_ name: String) -> MTLComputePipelineState? {
        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.library = library
        functionDescriptor.name = name
        let pipelineDescriptor = MTL4ComputePipelineDescriptor()
        pipelineDescriptor.computeFunctionDescriptor = functionDescriptor
        return try? compiler.makeComputePipelineState(descriptor: pipelineDescriptor)
    }

    let requiredPipelines = [
        "fp8sr_im2col_image",
        "fp8sr_im2col_features",
        "fp8sr_matmul",
        "fp8sr_postprocess",
        "fp8sr_pixel_shuffle",
    ]
    guard requiredPipelines.allSatisfy({ makePipeline($0) != nil }) else {
        return false
    }

    let argumentDescriptor = MTL4ArgumentTableDescriptor()
    argumentDescriptor.maxBufferBindCount = 4
    argumentDescriptor.maxTextureBindCount = 1
    guard (try? device.makeArgumentTable(descriptor: argumentDescriptor)) != nil else {
        return false
    }

    let residencyDescriptor = MTLResidencySetDescriptor()
    residencyDescriptor.initialCapacity = 1
    guard let residencySet = try? device.makeResidencySet(descriptor: residencyDescriptor),
          let probeBuffer = device.makeBuffer(
              length: 32 * 128,
              options: .storageModeShared
          ) else {
        return false
    }
    let tensorDescriptor = MTLTensorDescriptor()
    tensorDescriptor.dimensions = MTLTensorExtents([32, 32])!
    tensorDescriptor.strides = MTLTensorExtents([1, 128])!
    tensorDescriptor.dataType = MTLTensorDataType(rawValue: 142)!
    tensorDescriptor.usage = .compute
    tensorDescriptor.storageMode = .shared
    let attachments = MTLTensorBufferAttachments()
    attachments.setBuffer(probeBuffer, offset: 0, for: .data)
    guard (try? device.makeTensor(
        descriptor: tensorDescriptor,
        attachments: attachments
    )) != nil else {
        return false
    }
    residencySet.addAllocation(probeBuffer)
    residencySet.commit()
    return true
}

@available(macOS 27.0, *)
func tensorOpsAvailable() -> Bool {
    return fp8TensorOpsAvailable()
}

@available(macOS 27.0, *)
private struct FP8SRManifest: Decodable {
    let format: String
    let version: Int
    let upscaleFactor: Int
    let layout: String
    let inputChannels: Int
    let outputChannels: Int
    let weightDType: String
    let activationDType: String
    let accumulationDType: String
    let weightRowStrideBytes: Int
    let layers: [FP8SRManifestLayer]

    enum CodingKeys: String, CodingKey {
        case format, version, layout, layers
        case upscaleFactor = "upscale_factor"
        case inputChannels = "input_channels"
        case outputChannels = "output_channels"
        case weightDType = "weight_dtype"
        case activationDType = "activation_dtype"
        case accumulationDType = "accumulation_dtype"
        case weightRowStrideBytes = "weight_row_stride_bytes"
    }
}

@available(macOS 27.0, *)
private struct FP8SRManifestLayer: Decodable {
    let name: String
    let kernel: Int
    let inChannels: Int
    let outChannels: Int
    let weights: String
    let bias: String
    let scale: Float

    enum CodingKeys: String, CodingKey {
        case name, kernel, weights, bias, scale
        case inChannels = "in_channels"
        case outChannels = "out_channels"
    }
}

@available(macOS 27.0, *)
private enum FP8SRError: Error, CustomStringConvertible {
    case invalid(String)
    case unavailable(String)
    case execution(String)

    var description: String {
        switch self {
        case .invalid(let message): return "invalid_layout_\(message)"
        case .unavailable(let message): return message
        case .execution(let message): return message
        }
    }
}

@available(macOS 27.0, *)
private struct TensorOpsRawResult {
    let raw: [UInt8]
    let width: Int
    let height: Int
    let dispatch: String
    let tensorOpsMs: Double
    let readbackMs: Double
}

@available(macOS 27.0, *)
private struct FP8SRIm2ColParams {
    var width: UInt32
    var height: UInt32
    var sourceChannels: UInt32
    var sourceStride: UInt32
    var targetK: UInt32
    var kernelSize: UInt32
    var baseIndex: UInt64
}

@available(macOS 27.0, *)
private struct FP8SRPostParams {
    var pixelCount: UInt32
    var channels: UInt32
    var scale: Float
}

@available(macOS 27.0, *)
private struct FP8SRMatmulParams {
    var m: UInt32
    var n: UInt32
    var k: UInt32
}

@available(macOS 27.0, *)
private struct FP8SRPixelParams {
    var width: UInt32
    var height: UInt32
    var pixelCount: UInt32
    var channels: UInt32
}

@available(macOS 27.0, *)
private final class FP8SRRuntime {
    private struct Layer {
        let manifest: FP8SRManifestLayer
        let weightDType: String
        let weights: MTLTensor
        let weightBuffer: MTLBuffer
        let bias: MTLBuffer
    }

    private static let cacheLock = NSLock()
    private static var cache: [String: FP8SRRuntime] = [:]

    private let device: MTLDevice
    private let commandQueue: MTL4CommandQueue
    private let commandAllocator: MTL4CommandAllocator
    private let completionEvent: MTLSharedEvent
    private let argumentTable: MTL4ArgumentTable
    private let residencySet: MTLResidencySet
    private let imageIm2ColPipeline: MTLComputePipelineState
    private let featureIm2ColPipeline: MTLComputePipelineState
    private let matmulPipelines: [String: MTLComputePipelineState]
    private let postprocessPipeline: MTLComputePipelineState
    private let pixelShufflePipeline: MTLComputePipelineState
    private let layers: [Layer]
    private var transientBuffers: [MTLBuffer] = []
    private var nextCompletionValue: UInt64 = 1

    var weightDTypeName: String {
        layers.first?.weightDType ?? "unknown"
    }

    static func cached(packPath: String) throws -> FP8SRRuntime {
        let key = URL(fileURLWithPath: packPath).standardizedFileURL.path
        cacheLock.lock()
        if let runtime = cache[key] {
            cacheLock.unlock()
            return runtime
        }
        cacheLock.unlock()

        let runtime = try FP8SRRuntime(packPath: key)
        cacheLock.lock()
        cache[key] = runtime
        cacheLock.unlock()
        return runtime
    }

    private init(packPath: String) throws {
        guard FileManager.default.fileExists(atPath: packPath) else {
            throw FP8SRError.unavailable("fp8_model_unavailable")
        }
        let packURL = URL(fileURLWithPath: packPath).resolvingSymlinksInPath().standardizedFileURL
        let manifestURL = packURL.appendingPathComponent("manifest.json")
        guard let manifestData = try? Data(contentsOf: manifestURL) else {
            throw FP8SRError.unavailable("fp8_manifest_unavailable")
        }
        let manifest: FP8SRManifest
        do {
            manifest = try JSONDecoder().decode(FP8SRManifest.self, from: manifestData)
        } catch {
            throw FP8SRError.invalid("manifest_decode")
        }
        try FP8SRRuntime.validate(manifest: manifest, packURL: packURL)

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FP8SRError.unavailable("metal_unavailable")
        }
        self.device = device
        if ProcessInfo.processInfo.environment["MTL_CAPTURE_WAIT_FOR_SIGNAL"] == "1" {
            let configuredWait = ProcessInfo.processInfo.environment[
                "ORTHO4XP_GPU_CAPTURE_WAIT_SECONDS"
            ].flatMap(Double.init) ?? 3.0
            Thread.sleep(forTimeInterval: max(0.5, configuredWait))
        }
        guard let commandQueue = device.makeMTL4CommandQueue(),
              let commandAllocator = device.makeCommandAllocator(),
              let completionEvent = device.makeSharedEvent() else {
            throw FP8SRError.unavailable("metal4_unavailable")
        }
        self.commandQueue = commandQueue
        self.commandAllocator = commandAllocator
        self.completionEvent = completionEvent

        let compileOptions = MTLCompileOptions()
        compileOptions.languageVersion = .version4_1
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: fp8TensorOpsSource, options: compileOptions)
        } catch {
            reportError("ASHelper: TensorOps shader compile detail=\(error)")
            throw FP8SRError.unavailable("tensorops_shader_compile")
        }
        let compiler: MTL4Compiler
        do {
            compiler = try device.makeCompiler(descriptor: MTL4CompilerDescriptor())
        } catch {
            throw FP8SRError.unavailable("metal4_compiler_unavailable")
        }

        func makePipeline(_ name: String) throws -> MTLComputePipelineState {
            let functionDescriptor = MTL4LibraryFunctionDescriptor()
            functionDescriptor.library = library
            functionDescriptor.name = name
            let pipelineDescriptor = MTL4ComputePipelineDescriptor()
            pipelineDescriptor.computeFunctionDescriptor = functionDescriptor
            return try compiler.makeComputePipelineState(descriptor: pipelineDescriptor)
        }
        do {
            self.imageIm2ColPipeline = try makePipeline("fp8sr_im2col_image")
            self.featureIm2ColPipeline = try makePipeline("fp8sr_im2col_features")
            self.postprocessPipeline = try makePipeline("fp8sr_postprocess")
            self.pixelShufflePipeline = try makePipeline("fp8sr_pixel_shuffle")
        } catch {
            reportError("ASHelper: TensorOps pipeline compile detail=\(error)")
            throw FP8SRError.unavailable("tensorops_pipeline_compile")
        }
        var compiledMatmulPipelines: [String: MTLComputePipelineState] = [:]
        for dtype in ["Float16", "MetalFloat8E4M3", "MetalFloat4E2M1", "Int2"] {
            do {
                let functionName: String
                switch dtype {
                case "Float16": functionName = "fp8sr_matmul_fp16"
                case "MetalFloat8E4M3": functionName = "fp8sr_matmul"
                case "MetalFloat4E2M1": functionName = "fp8sr_matmul_fp4"
                case "Int2": functionName = "fp8sr_matmul_int2"
                default: continue
                }
                compiledMatmulPipelines[dtype] = try makePipeline(functionName)
            } catch {
                reportError("ASHelper: TensorOps dtype=\(dtype) unavailable detail=\(error)")
            }
        }
        guard compiledMatmulPipelines["MetalFloat8E4M3"] != nil else {
            throw FP8SRError.unavailable("fp8_pipeline_compile")
        }
        self.matmulPipelines = compiledMatmulPipelines

        let argumentDescriptor = MTL4ArgumentTableDescriptor()
        argumentDescriptor.maxBufferBindCount = 4
        argumentDescriptor.maxTextureBindCount = 1
        do {
            self.argumentTable = try device.makeArgumentTable(descriptor: argumentDescriptor)
        } catch {
            throw FP8SRError.unavailable("metal4_argument_table")
        }
        let residencyDescriptor = MTLResidencySetDescriptor()
        residencyDescriptor.initialCapacity = 32
        guard let residencySet = try? device.makeResidencySet(descriptor: residencyDescriptor) else {
            throw FP8SRError.unavailable("metal4_residency_set")
        }
        self.residencySet = residencySet

        var loadedLayers: [Layer] = []
        for layerManifest in manifest.layers {
            let weightsURL = packURL.appendingPathComponent(layerManifest.weights)
                .resolvingSymlinksInPath().standardizedFileURL
            let biasURL = packURL.appendingPathComponent(layerManifest.bias)
                .resolvingSymlinksInPath().standardizedFileURL
            let weightsData = try Data(contentsOf: weightsURL)
            let biasData = try Data(contentsOf: biasURL)
            let runtimeK = FP8SRRuntime.paddedKernelElements(layerManifest)
            let sourceK = ((layerManifest.kernel * layerManifest.kernel * layerManifest.inChannels + 31) / 32) * 32
            let weightsBuffer = try FP8SRRuntime.makeWeightBuffer(
                device: device,
                data: weightsData,
                sourceKernelElements: sourceK,
                runtimeKernelElements: runtimeK
            )
            let biasBuffer = try FP8SRRuntime.makeSharedBuffer(device: device, data: biasData)
            let kPadded = runtimeK
            let weightDescriptor = MTLTensorDescriptor()
            weightDescriptor.dimensions = MTLTensorExtents([32, kPadded])!
            weightDescriptor.strides = MTLTensorExtents([
                1, FP8SRRuntime.weightStrideElements(for: manifest.weightDType)
            ])!
            weightDescriptor.dataType = FP8SRRuntime.tensorDataType(for: manifest.weightDType)
            weightDescriptor.usage = .compute
            weightDescriptor.storageMode = .shared
            let attachments = MTLTensorBufferAttachments()
            attachments.setBuffer(weightsBuffer, offset: 0, for: .data)
            guard let weightsTensor = try? device.makeTensor(
                descriptor: weightDescriptor,
                attachments: attachments
            ) else {
                throw FP8SRError.invalid("weight_tensor_(layerManifest.name)")
            }
            loadedLayers.append(Layer(
                manifest: layerManifest,
                weightDType: manifest.weightDType,
                weights: weightsTensor,
                weightBuffer: weightsBuffer,
                bias: biasBuffer
            ))
            residencySet.addAllocation(weightsBuffer)
            residencySet.addAllocation(biasBuffer)
        }
        residencySet.commit()
        self.layers = loadedLayers
        let availableDTypes = ["Float16", "MetalFloat8E4M3", "MetalFloat4E2M1", "Int2"]
            .filter { compiledMatmulPipelines[$0] != nil }
            .joined(separator: ",")
        print("tensorops_dispatch=ready dtype=\(manifest.weightDType) activation=Float16 accumulation=Float16 layout=NHWC "
            + "available_dtypes=\(availableDTypes) simdgroup=\(compiledMatmulPipelines[manifest.weightDType]?.threadExecutionWidth ?? 0) "
            + "threadgroup=\(compiledMatmulPipelines[manifest.weightDType]?.maxTotalThreadsPerThreadgroup ?? 0)")
    }

    private static func supportedWeightDTypes() -> Set<String> {
        return ["Float16", "MetalFloat8E4M3", "MetalFloat4E2M1", "Int2"]
    }

    private static func tensorDataType(for dtype: String) -> MTLTensorDataType {
        switch dtype {
        case "Float16": return .float16
        case "MetalFloat8E4M3": return MTLTensorDataType(rawValue: 142)!
        case "MetalFloat4E2M1": return MTLTensorDataType(rawValue: 148)!
        case "Int2": return MTLTensorDataType(rawValue: 150)!
        default: return .float16
        }
    }

    private static func weightStrideElements(for dtype: String) -> Int {
        switch dtype {
        case "Float16": return 64
        case "MetalFloat8E4M3": return 128
        case "MetalFloat4E2M1": return 256
        case "Int2": return 512
        default: return 128
        }
    }

    private static func paddedKernelElements(_ layer: FP8SRManifestLayer) -> Int {
        let elements = layer.kernel * layer.kernel * layer.inChannels
        // Keep the external pack contract 32-aligned, but use the wider
        // dynamic-K tile required by the current M5 FP8 TensorOps runtime for
        // the 32-channel convolution layers. K=288 is therefore zero-padded
        // to 320 only in the GPU working buffers.
        if elements <= 32 { return ((elements + 31) / 32) * 32 }
        return ((elements + 63) / 64) * 64
    }

    private static func makeSharedBuffer(device: MTLDevice, data: Data) throws -> MTLBuffer {
        let length = max(128, (data.count + 127) & ~127)
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw FP8SRError.unavailable("buffer_allocation")
        }
        buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: length)
        data.copyBytes(to: buffer.contents().assumingMemoryBound(to: UInt8.self), count: data.count)
        return buffer
    }

    private static func makeWeightBuffer(
        device: MTLDevice,
        data: Data,
        sourceKernelElements: Int,
        runtimeKernelElements: Int
    ) throws -> MTLBuffer {
        let rowBytes = 128
        let length = max(128, runtimeKernelElements * rowBytes)
        guard data.count == sourceKernelElements * rowBytes,
              let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw FP8SRError.invalid("weight_padding")
        }
        buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: length)
        data.copyBytes(to: buffer.contents().assumingMemoryBound(to: UInt8.self), count: data.count)
        return buffer
    }

    private static func validate(manifest: FP8SRManifest, packURL: URL) throws {
        let exact = manifest.format == "FP8SR"
            && (manifest.version == 1 || manifest.version == 2)
            && manifest.upscaleFactor == 2
            && manifest.layout == "NHWC"
            && manifest.inputChannels == 3
            && manifest.outputChannels == 3
            && FP8SRRuntime.supportedWeightDTypes().contains(manifest.weightDType)
            && manifest.activationDType == "Float16"
            && manifest.accumulationDType == "Float16"
            && manifest.weightRowStrideBytes == 128
        guard exact else { throw FP8SRError.invalid("manifest_contract") }
        if manifest.version == 1 && manifest.weightDType != "MetalFloat8E4M3" {
            throw FP8SRError.invalid("legacy_manifest_dtype")
        }
        let expected: [(String, Int, Int, Int)] = [
            ("conv0", 3, 3, 32),
            ("conv1", 3, 32, 32),
            ("conv2", 3, 32, 12),
        ]
        guard manifest.layers.count == expected.count else {
            throw FP8SRError.invalid("layer_count")
        }
        for (layer, expectedLayer) in zip(manifest.layers, expected) {
            guard layer.name == expectedLayer.0,
                  layer.kernel == expectedLayer.1,
                  layer.inChannels == expectedLayer.2,
                  layer.outChannels == expectedLayer.3,
                  layer.scale.isFinite,
                  layer.scale > 0 else {
                throw FP8SRError.invalid("layer_\(layer.name)")
            }
            let kPadded = ((layer.kernel * layer.kernel * layer.inChannels + 31) / 32) * 32
            let outPadded = ((layer.outChannels + 31) / 32) * 32
            guard kPadded % 32 == 0, outPadded % 32 == 0,
                  manifest.weightRowStrideBytes == 128 else {
                throw FP8SRError.invalid("tensor_alignment_\(layer.name)")
            }
            let weightsURL = packURL.appendingPathComponent(layer.weights)
                .resolvingSymlinksInPath().standardizedFileURL
            let biasURL = packURL.appendingPathComponent(layer.bias)
                .resolvingSymlinksInPath().standardizedFileURL
            guard weightsURL.path.hasPrefix(packURL.path + "/"),
                  biasURL.path.hasPrefix(packURL.path + "/"),
                  FileManager.default.fileExists(atPath: weightsURL.path),
                  FileManager.default.fileExists(atPath: biasURL.path) else {
                throw FP8SRError.invalid("layer_files_\(layer.name)")
            }
            let weightSize = (try? FileManager.default.attributesOfItem(atPath: weightsURL.path)[.size] as? NSNumber)?.intValue ?? -1
            let biasSize = (try? FileManager.default.attributesOfItem(atPath: biasURL.path)[.size] as? NSNumber)?.intValue ?? -1
            guard weightSize == kPadded * 128, biasSize == outPadded * 2 else {
                throw FP8SRError.invalid("buffer_size_\(layer.name)")
            }
        }
    }

    private func makeTensor(
        buffer: MTLBuffer,
        dimensions: [Int],
        strides: [Int],
        dataType: MTLTensorDataType
    ) throws -> MTLTensor {
        let descriptor = MTLTensorDescriptor()
        descriptor.dimensions = MTLTensorExtents(dimensions)!
        descriptor.strides = MTLTensorExtents(strides)!
        descriptor.dataType = dataType
        descriptor.usage = .compute
        descriptor.storageMode = .shared
        let attachments = MTLTensorBufferAttachments()
        attachments.setBuffer(buffer, offset: 0, for: .data)
        guard let tensor = try? device.makeTensor(descriptor: descriptor, attachments: attachments) else {
            throw FP8SRError.invalid("activation_tensor")
        }
        return tensor
    }

    private func makeParamsBuffer<T>(_ value: inout T) throws -> MTLBuffer {
        let data = withUnsafeBytes(of: &value) { Data($0) }
        let buffer = try FP8SRRuntime.makeSharedBuffer(device: device, data: data)
        transientBuffers.append(buffer)
        residencySet.addAllocation(buffer)
        return buffer
    }

    private func encodeDispatchBarrier(_ encoder: MTL4ComputeCommandEncoder) {
        // MTL4 keeps multiple dispatches in one encoder, so dependent kernels
        // need an explicit intra-pass barrier. Without it, a matmul may read
        // the zero-filled activation buffer before im2col has completed.
        encoder.barrier(
            afterEncoderStages: .dispatch,
            beforeEncoderStages: .dispatch,
            visibilityOptions: .device
        )
    }

    private func encodeImageIm2Col(
        encoder: MTL4ComputeCommandEncoder,
        image: MTLTexture,
        destination: MTLBuffer,
        params: inout FP8SRIm2ColParams,
        elementCount: Int
    ) throws {
        let maxChunk = 1 << 30
        var baseIndex = 0
        while baseIndex < elementCount {
            let chunk = min(maxChunk, elementCount - baseIndex)
            var chunkParams = params
            chunkParams.baseIndex = UInt64(baseIndex)
            let paramsBuffer = try makeParamsBuffer(&chunkParams)
            argumentTable.setTexture(image.gpuResourceID, index: 0)
            argumentTable.setAddress(destination.gpuAddress, index: 1)
            argumentTable.setAddress(paramsBuffer.gpuAddress, index: 2)
            encoder.setComputePipelineState(imageIm2ColPipeline)
            encoder.setArgumentTable(argumentTable)
            encoder.dispatchThreads(
                threadsPerGrid: MTLSize(width: chunk, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            baseIndex += chunk
            if baseIndex < elementCount {
                encodeDispatchBarrier(encoder)
            }
        }
        encodeDispatchBarrier(encoder)
    }

    private func encodeFeatureIm2Col(
        encoder: MTL4ComputeCommandEncoder,
        source: MTLBuffer,
        destination: MTLBuffer,
        params: inout FP8SRIm2ColParams,
        elementCount: Int
    ) throws {
        let maxChunk = 1 << 30
        var baseIndex = 0
        while baseIndex < elementCount {
            let chunk = min(maxChunk, elementCount - baseIndex)
            var chunkParams = params
            chunkParams.baseIndex = UInt64(baseIndex)
            let paramsBuffer = try makeParamsBuffer(&chunkParams)
            argumentTable.setAddress(source.gpuAddress, index: 0)
            argumentTable.setAddress(destination.gpuAddress, index: 1)
            argumentTable.setAddress(paramsBuffer.gpuAddress, index: 2)
            encoder.setComputePipelineState(featureIm2ColPipeline)
            encoder.setArgumentTable(argumentTable)
            encoder.dispatchThreads(
                threadsPerGrid: MTLSize(width: chunk, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            baseIndex += chunk
            if baseIndex < elementCount {
                encodeDispatchBarrier(encoder)
            }
        }
        encodeDispatchBarrier(encoder)
    }

    private func encodeMatmul(
        encoder: MTL4ComputeCommandEncoder,
        activation: MTLBuffer,
        weights: MTLBuffer,
        output: MTLBuffer,
        params: inout FP8SRMatmulParams,
        weightDType: String,
        pixelCount: Int
    ) throws {
        guard let matmulPipeline = matmulPipelines[weightDType] else {
            throw FP8SRError.unavailable("tensorops_dtype_\(weightDType)")
        }
        let paramsBuffer = try makeParamsBuffer(&params)
        argumentTable.setAddress(activation.gpuAddress, index: 0)
        argumentTable.setAddress(weights.gpuAddress, index: 1)
        argumentTable.setAddress(output.gpuAddress, index: 2)
        argumentTable.setAddress(paramsBuffer.gpuAddress, index: 3)
        encoder.setComputePipelineState(matmulPipeline)
        encoder.setArgumentTable(argumentTable)
        let threadgroups = MTLSize(width: 1, height: (pixelCount + 63) / 64, depth: 1)
        let width = max(1, matmulPipeline.threadExecutionWidth) * 4
        encoder.dispatchThreadgroups(
            threadgroupsPerGrid: threadgroups,
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encodeDispatchBarrier(encoder)
    }

    private func encodePostprocess(
        encoder: MTL4ComputeCommandEncoder,
        values: MTLBuffer,
        bias: MTLBuffer,
        params: inout FP8SRPostParams,
        error: MTLBuffer
    ) throws {
        let paramsBuffer = try makeParamsBuffer(&params)
        argumentTable.setAddress(values.gpuAddress, index: 0)
        argumentTable.setAddress(bias.gpuAddress, index: 1)
        argumentTable.setAddress(paramsBuffer.gpuAddress, index: 2)
        argumentTable.setAddress(error.gpuAddress, index: 3)
        encoder.setComputePipelineState(postprocessPipeline)
        encoder.setArgumentTable(argumentTable)
        encoder.dispatchThreads(
            threadsPerGrid: MTLSize(width: Int(params.pixelCount) * Int(params.channels), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encodeDispatchBarrier(encoder)
    }

    private func encodePixelShuffle(
        encoder: MTL4ComputeCommandEncoder,
        values: MTLBuffer,
        destination: MTLBuffer,
        params: inout FP8SRPixelParams
    ) throws {
        let paramsBuffer = try makeParamsBuffer(&params)
        argumentTable.setAddress(values.gpuAddress, index: 0)
        argumentTable.setAddress(destination.gpuAddress, index: 1)
        argumentTable.setAddress(paramsBuffer.gpuAddress, index: 2)
        encoder.setComputePipelineState(pixelShufflePipeline)
        encoder.setArgumentTable(argumentTable)
        encoder.dispatchThreads(
            threadsPerGrid: MTLSize(width: Int(params.pixelCount), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encodeDispatchBarrier(encoder)
    }

    private static let coreTileDimension = 2048
    private static let tileHalo = 1
    private static let maxSingleTileDimension = coreTileDimension + tileHalo * 2
    private static let safeActivationBytes = 3 * 1024 * 1024 * 1024

    private func validateSingleTileGeometry(width: Int, height: Int) throws -> Int {
        guard width > 0, height > 0,
              width <= Self.maxSingleTileDimension,
              height <= Self.maxSingleTileDimension else {
            throw FP8SRError.execution("oversize_preflight")
        }
        let pixelCount = try checkedMultiply(width, height, label: "pixel_count")
        guard pixelCount <= Int(UInt32.max) else {
            throw FP8SRError.execution("im2col_index_overflow")
        }
        for layer in layers {
            let kPadded = FP8SRRuntime.paddedKernelElements(layer.manifest)
            let activationBytes = try checkedMultiply(
                kPadded, pixelCount, 2, label: "activation_bytes"
            )
            guard activationBytes <= Self.safeActivationBytes else {
                throw FP8SRError.execution("oversize_preflight")
            }
            let im2colElements = try checkedMultiply(
                pixelCount, kPadded, label: "im2col_elements"
            )
            guard im2colElements <= Int(UInt32.max) else {
                throw FP8SRError.execution("im2col_index_overflow")
            }
        }
        let postprocessElements = try checkedMultiply(pixelCount, 32, label: "postprocess_elements")
        guard postprocessElements <= Int(UInt32.max) else {
            throw FP8SRError.execution("im2col_index_overflow")
        }
        return pixelCount
    }

    private func cropRGBA(
        _ raw: [UInt8],
        sourceWidth: Int,
        sourceHeight: Int,
        x: Int,
        y: Int,
        width: Int,
        height: Int
    ) throws -> [UInt8] {
        guard x >= 0, y >= 0, width > 0, height > 0,
              x + width <= sourceWidth, y + height <= sourceHeight else {
            throw FP8SRError.execution("tile_bounds")
        }
        let outputCount = try checkedMultiply(width, height, 4, label: "tile_rgba")
        var result = [UInt8](repeating: 0, count: outputCount)
        let sourceRowBytes = try checkedMultiply(sourceWidth, 4, label: "source_row_bytes")
        let tileRowBytes = try checkedMultiply(width, 4, label: "tile_row_bytes")
        for row in 0..<height {
            let sourceOffset = try checkedMultiply(y + row, sourceRowBytes, label: "tile_source_offset")
                + x * 4
            let destinationOffset = row * tileRowBytes
            result[destinationOffset..<(destinationOffset + tileRowBytes)] =
                raw[sourceOffset..<(sourceOffset + tileRowBytes)]
        }
        return result
    }

    private func upscaleTiled(raw: [UInt8], width: Int, height: Int) throws -> [UInt8] {
        let outputWidth = try checkedMultiply(width, 2, label: "tiled_output_width")
        let outputHeight = try checkedMultiply(height, 2, label: "tiled_output_height")
        let outputCount = try checkedMultiply(outputWidth, outputHeight, 4, label: "tiled_output")
        var result = [UInt8](repeating: 0, count: outputCount)

        let tileColumns = (width + Self.coreTileDimension - 1) / Self.coreTileDimension
        let tileRows = (height + Self.coreTileDimension - 1) / Self.coreTileDimension
        let tileCount = tileColumns * tileRows
        var tileIndex = 0
        for coreY in stride(from: 0, to: height, by: Self.coreTileDimension) {
            let coreHeight = min(Self.coreTileDimension, height - coreY)
            for coreX in stride(from: 0, to: width, by: Self.coreTileDimension) {
                let coreWidth = min(Self.coreTileDimension, width - coreX)
                let tileX = max(0, coreX - Self.tileHalo)
                let tileY = max(0, coreY - Self.tileHalo)
                let tileMaxX = min(width, coreX + coreWidth + Self.tileHalo)
                let tileMaxY = min(height, coreY + coreHeight + Self.tileHalo)
                let tileWidth = tileMaxX - tileX
                let tileHeight = tileMaxY - tileY
                let tileRaw = try cropRGBA(
                    raw,
                    sourceWidth: width,
                    sourceHeight: height,
                    x: tileX,
                    y: tileY,
                    width: tileWidth,
                    height: tileHeight
                )
                let tileOutputRaw: [UInt8]
                do {
                    tileOutputRaw = try autoreleasepool {
                        try upscaleSingleRaw(
                            raw: tileRaw,
                            width: tileWidth,
                            height: tileHeight,
                            emitCompletionLog: false
                        )
                    }
                } catch let error as FP8SRError {
                    switch error {
                    case .execution("nonfinite"):
                        throw FP8SRError.execution("tile_nonfinite")
                    case .execution("gpu_timeout"):
                        throw FP8SRError.execution("tensorops_gpu_failure")
                    default:
                        throw error
                    }
                }
                let tileOutputWidth = try checkedMultiply(tileWidth, 2, label: "tile_output_width")
                let tileOutputHeight = try checkedMultiply(tileHeight, 2, label: "tile_output_height")
                let expectedTileOutputBytes = try checkedMultiply(
                    tileOutputWidth, tileOutputHeight, 4, label: "tile_output_rgba"
                )
                guard tileOutputRaw.count == expectedTileOutputBytes else {
                    throw FP8SRError.execution("tile_output_invalid")
                }
                let copyX = (coreX - tileX) * 2
                let copyY = (coreY - tileY) * 2
                let destinationRowBytes = outputWidth * 4
                let sourceRowBytes = tileOutputWidth * 4
                let copyRowBytes = coreWidth * 2 * 4
                for row in 0..<(coreHeight * 2) {
                    let sourceOffset = (copyY + row) * sourceRowBytes + copyX * 4
                    let destinationOffset = (coreY * 2 + row) * destinationRowBytes + coreX * 2 * 4
                    result[destinationOffset..<(destinationOffset + copyRowBytes)] =
                        tileOutputRaw[sourceOffset..<(sourceOffset + copyRowBytes)]
                }
                tileIndex += 1
                print("tensorops_dispatch=tiled tile=\(tileIndex)/\(tileCount) core=\(coreWidth)x\(coreHeight) input=\(tileWidth)x\(tileHeight) halo=\(Self.tileHalo)")
            }
        }
        return result
    }

    func upscale(inputPath: String, outputPath: String) throws {
        let result = try upscaleRaw(inputPath: inputPath)
        guard writePNG(
            result.raw,
            width: result.width,
            height: result.height,
            outputPath: outputPath
        ) else {
            throw FP8SRError.execution("output_write")
        }
        let dtype = layers.first?.weightDType ?? "unknown"
        print(
            "tensorops_dispatch=completed dtype=\(dtype) activation=Float16 "
                + "accumulation=Float16 dispatch=\(result.dispatch) "
                + "png_intermediate=true "
                + "output=\(result.width)x\(result.height)"
        )
    }

    func upscaleRaw(inputPath: String) throws -> TensorOpsRawResult {
        let sourceURL = URL(fileURLWithPath: inputPath)
        guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw FP8SRError.execution("input_decode")
        }
        let raw = getRawRGBA(cgImage: image)
        let inputByteCount = try checkedMultiply(image.width, image.height, 4, label: "input_rgba")
        guard raw.count == inputByteCount else {
            throw FP8SRError.execution("input_normalize")
        }
        guard image.width > 0, image.height > 0,
              image.width <= 8192, image.height <= 8192 else {
            throw FP8SRError.execution("input_dimensions")
        }
        if stride(from: 3, to: raw.count, by: 4).contains(where: { raw[$0] < 255 }) {
            throw FP8SRError.execution("alpha")
        }
        let started = CFAbsoluteTimeGetCurrent()
        if max(image.width, image.height) > Self.coreTileDimension {
            let tiledOutput = try upscaleTiled(raw: raw, width: image.width, height: image.height)
            return TensorOpsRawResult(
                raw: tiledOutput,
                width: image.width * 2,
                height: image.height * 2,
                dispatch: "tiled",
                tensorOpsMs: (CFAbsoluteTimeGetCurrent() - started) * 1000.0,
                readbackMs: 0.0
            )
        }
        let readbackStarted = CFAbsoluteTimeGetCurrent()
        let outputRaw = try upscaleSingleRaw(
            raw: raw,
            width: image.width,
            height: image.height,
            emitCompletionLog: false
        )
        let readbackMs = (CFAbsoluteTimeGetCurrent() - readbackStarted) * 1000.0
        return TensorOpsRawResult(
            raw: outputRaw,
            width: image.width * 2,
            height: image.height * 2,
            dispatch: "single",
            tensorOpsMs: (CFAbsoluteTimeGetCurrent() - started) * 1000.0,
            readbackMs: readbackMs
        )
    }

    private func upscaleSingleRaw(
        raw: [UInt8],
        width: Int,
        height: Int,
        emitCompletionLog: Bool = true
    ) throws -> [UInt8] {
        let pixelCount = try validateSingleTileGeometry(width: width, height: height)
        let inputByteCount = try checkedMultiply(width, height, 4, label: "input_rgba")
        guard raw.count == inputByteCount else {
            throw FP8SRError.execution("input_normalize")
        }
        if stride(from: 3, to: raw.count, by: 4).contains(where: { raw[$0] < 255 }) {
            throw FP8SRError.execution("alpha")
        }
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        textureDescriptor.storageMode = .shared
        textureDescriptor.usage = .shaderRead
        guard let inputTexture = device.makeTexture(descriptor: textureDescriptor) else {
            throw FP8SRError.execution("input_texture")
        }
        raw.withUnsafeBytes { bytes in
            inputTexture.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: bytes.baseAddress!,
                bytesPerRow: width * 4
            )
        }

        guard let commandBuffer = device.makeCommandBuffer(),
              let errorBuffer = device.makeBuffer(length: 128, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(
                  length: max(128, try checkedMultiply(width, height, 16, label: "output_buffer")),
                  options: .storageModeShared
              ) else {
            throw FP8SRError.execution("command_buffer")
        }
        errorBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: errorBuffer.length)
        outputBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: outputBuffer.length)
        transientBuffers.removeAll(keepingCapacity: true)
        var temporaryResources: [MTLAllocation] = [inputTexture, errorBuffer, outputBuffer]
        defer {
            for resource in temporaryResources {
                residencySet.removeAllocation(resource)
            }
            for buffer in transientBuffers {
                residencySet.removeAllocation(buffer)
            }
            transientBuffers.removeAll(keepingCapacity: true)
            residencySet.commit()
        }
        residencySet.addAllocation(inputTexture)
        residencySet.addAllocation(errorBuffer)
        residencySet.addAllocation(outputBuffer)
        commandAllocator.reset()
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            commandBuffer.endCommandBuffer()
            throw FP8SRError.execution("compute_encoder")
        }

        var previousOutput: MTLBuffer?
        for (index, layer) in layers.enumerated() {
            let kPadded = FP8SRRuntime.paddedKernelElements(layer.manifest)
            let activationBuffer = try FP8SRRuntime.makeSharedBuffer(
                device: device,
                data: Data(count: max(128, try checkedMultiply(kPadded, pixelCount, 2, label: "activation_buffer")))
            )
            residencySet.addAllocation(activationBuffer)
            temporaryResources.append(activationBuffer)
            _ = try makeTensor(
                buffer: activationBuffer,
                dimensions: [kPadded, pixelCount],
                strides: [1, kPadded],
                dataType: .float16
            )
            if index == 0 {
                var params = FP8SRIm2ColParams(
                    width: UInt32(width), height: UInt32(height),
                    sourceChannels: 3, sourceStride: 3,
                    targetK: UInt32(kPadded), kernelSize: UInt32(layer.manifest.kernel), baseIndex: 0
                )
                try encodeImageIm2Col(
                    encoder: encoder,
                    image: inputTexture,
                    destination: activationBuffer,
                    params: &params,
                    elementCount: try checkedMultiply(pixelCount, kPadded, label: "image_im2col")
                )
            } else if let previousOutput {
                var params = FP8SRIm2ColParams(
                    width: UInt32(width), height: UInt32(height),
                    sourceChannels: UInt32(layer.manifest.inChannels), sourceStride: 32,
                    targetK: UInt32(kPadded), kernelSize: UInt32(layer.manifest.kernel), baseIndex: 0
                )
                try encodeFeatureIm2Col(
                    encoder: encoder,
                    source: previousOutput,
                    destination: activationBuffer,
                    params: &params,
                    elementCount: try checkedMultiply(pixelCount, kPadded, label: "feature_im2col")
                )
            }

            let outputLength = max(128, try checkedMultiply(32, pixelCount, 2, label: "layer_output") + 128)
            guard let layerOutput = device.makeBuffer(length: outputLength, options: .storageModeShared) else {
                throw FP8SRError.execution("layer_output")
            }
            layerOutput.contents().initializeMemory(as: UInt8.self, repeating: 0, count: layerOutput.length)
            residencySet.addAllocation(layerOutput)
            temporaryResources.append(layerOutput)
            _ = try makeTensor(
                buffer: layerOutput,
                dimensions: [32, pixelCount],
                strides: [1, 32],
                dataType: .float16
            )
            var matmulParams = FP8SRMatmulParams(
                m: UInt32(pixelCount), n: 32, k: UInt32(kPadded)
            )
            try encodeMatmul(
                encoder: encoder,
                activation: activationBuffer,
                weights: layer.weightBuffer,
                output: layerOutput,
                params: &matmulParams,
                weightDType: layer.weightDType,
                pixelCount: pixelCount
            )
            var postParams = FP8SRPostParams(
                pixelCount: UInt32(pixelCount), channels: 32, scale: layer.manifest.scale
            )
            try encodePostprocess(
                encoder: encoder,
                values: layerOutput,
                bias: layer.bias,
                params: &postParams,
                error: errorBuffer
            )
            previousOutput = layerOutput
        }
        guard let finalOutput = previousOutput else {
            commandBuffer.endCommandBuffer()
            throw FP8SRError.execution("no_layers")
        }
        var pixelParams = FP8SRPixelParams(
            width: UInt32(width), height: UInt32(height),
            pixelCount: UInt32(pixelCount), channels: 32
        )
        try encodePixelShuffle(
            encoder: encoder,
            values: finalOutput,
            destination: outputBuffer,
            params: &pixelParams
        )
        encoder.endEncoding()
        residencySet.commit()
        commandBuffer.useResidencySet(residencySet)
        commandBuffer.endCommandBuffer()
        let completionValue = nextCompletionValue
        nextCompletionValue += 1
        commandQueue.commit([commandBuffer])
        commandQueue.signalEvent(completionEvent, value: completionValue)
        completionEvent.wait(untilSignaledValue: completionValue, timeoutMS: 120_000)
        guard completionEvent.signaledValue >= completionValue else {
            throw FP8SRError.execution("gpu_timeout")
        }
        let errorValue = errorBuffer.contents().assumingMemoryBound(to: UInt32.self).pointee
        guard errorValue == 0 else {
            throw FP8SRError.execution("nonfinite")
        }
        let outputWidth = try checkedMultiply(width, 2, label: "output_width")
        let outputHeight = try checkedMultiply(height, 2, label: "output_height")
        let outputBytes = try checkedMultiply(outputWidth, outputHeight, 4, label: "output_rgba")
        let outputRaw = Array(
            UnsafeBufferPointer(
                start: outputBuffer.contents().assumingMemoryBound(to: UInt8.self),
                count: outputBytes
            )
        )
        guard outputRaw.count == outputBytes,
              outputRaw.count >= 4,
              stride(from: 3, to: outputRaw.count, by: 4).allSatisfy({
                  outputRaw[$0] == 255
              }) else {
            throw FP8SRError.execution("output_invalid")
        }
        if emitCompletionLog {
            let dtype = layers.first?.weightDType ?? "unknown"
            print("tensorops_dispatch=completed dtype=\(dtype) activation=Float16 accumulation=Float16 output=\(outputWidth)x\(outputHeight)")
        }
        return outputRaw
    }

    private func upscaleSingle(
        inputPath: String,
        outputPath: String,
        emitCompletionLog: Bool = true
    ) throws {
        let sourceURL = URL(fileURLWithPath: inputPath)
        guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw FP8SRError.execution("input_decode")
        }
        let raw = getRawRGBA(cgImage: image)
        let pixelCount = try validateSingleTileGeometry(width: image.width, height: image.height)
        let inputByteCount = try checkedMultiply(image.width, image.height, 4, label: "input_rgba")
        guard raw.count == inputByteCount else {
            throw FP8SRError.execution("input_normalize")
        }
        if stride(from: 3, to: raw.count, by: 4).contains(where: { raw[$0] < 255 }) {
            throw FP8SRError.execution("alpha")
        }
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: image.width,
            height: image.height,
            mipmapped: false
        )
        textureDescriptor.storageMode = .shared
        textureDescriptor.usage = .shaderRead
        guard let inputTexture = device.makeTexture(descriptor: textureDescriptor) else {
            throw FP8SRError.execution("input_texture")
        }
        raw.withUnsafeBytes { bytes in
            inputTexture.replace(
                region: MTLRegionMake2D(0, 0, image.width, image.height),
                mipmapLevel: 0,
                withBytes: bytes.baseAddress!,
                bytesPerRow: image.width * 4
            )
        }

        guard let commandBuffer = device.makeCommandBuffer(),
              let errorBuffer = device.makeBuffer(length: 128, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(
                  length: max(128, try checkedMultiply(image.width, image.height, 16, label: "output_buffer")),
                  options: .storageModeShared
              ) else {
            throw FP8SRError.execution("command_buffer")
        }
        errorBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: errorBuffer.length)
        outputBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: outputBuffer.length)
        transientBuffers.removeAll(keepingCapacity: true)
        var temporaryResources: [MTLAllocation] = [inputTexture, errorBuffer, outputBuffer]
        defer {
            for resource in temporaryResources {
                residencySet.removeAllocation(resource)
            }
            for buffer in transientBuffers {
                residencySet.removeAllocation(buffer)
            }
            transientBuffers.removeAll(keepingCapacity: true)
            residencySet.commit()
        }
        residencySet.addAllocation(inputTexture)
        residencySet.addAllocation(errorBuffer)
        residencySet.addAllocation(outputBuffer)
        commandAllocator.reset()
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            commandBuffer.endCommandBuffer()
            throw FP8SRError.execution("compute_encoder")
        }

        var previousOutput: MTLBuffer?
        for (index, layer) in layers.enumerated() {
            let kPadded = FP8SRRuntime.paddedKernelElements(layer.manifest)
            let activationBuffer = try FP8SRRuntime.makeSharedBuffer(
                device: device,
                data: Data(count: max(128, try checkedMultiply(kPadded, pixelCount, 2, label: "activation_buffer")))
            )
            residencySet.addAllocation(activationBuffer)
            temporaryResources.append(activationBuffer)
            _ = try makeTensor(
                buffer: activationBuffer,
                dimensions: [kPadded, pixelCount],
                strides: [1, kPadded],
                dataType: .float16
            )
            if index == 0 {
                var params = FP8SRIm2ColParams(
                    width: UInt32(image.width), height: UInt32(image.height),
                    sourceChannels: 3, sourceStride: 3,
                    targetK: UInt32(kPadded), kernelSize: UInt32(layer.manifest.kernel), baseIndex: 0
                )
                try encodeImageIm2Col(
                    encoder: encoder,
                    image: inputTexture,
                    destination: activationBuffer,
                    params: &params,
                    elementCount: try checkedMultiply(pixelCount, kPadded, label: "image_im2col")
                )
            } else if let previousOutput {
                var params = FP8SRIm2ColParams(
                    width: UInt32(image.width), height: UInt32(image.height),
                    sourceChannels: UInt32(layer.manifest.inChannels), sourceStride: 32,
                    targetK: UInt32(kPadded), kernelSize: UInt32(layer.manifest.kernel), baseIndex: 0
                )
                try encodeFeatureIm2Col(
                    encoder: encoder,
                    source: previousOutput,
                    destination: activationBuffer,
                    params: &params,
                    elementCount: try checkedMultiply(pixelCount, kPadded, label: "feature_im2col")
                )
            }

            let outputLength = max(128, try checkedMultiply(32, pixelCount, 2, label: "layer_output") + 128)
            guard let layerOutput = device.makeBuffer(length: outputLength, options: .storageModeShared) else {
                throw FP8SRError.execution("layer_output")
            }
            layerOutput.contents().initializeMemory(as: UInt8.self, repeating: 0, count: layerOutput.length)
            residencySet.addAllocation(layerOutput)
            temporaryResources.append(layerOutput)
            _ = try makeTensor(
                buffer: layerOutput,
                dimensions: [32, pixelCount],
                strides: [1, 32],
                dataType: .float16
            )
            var matmulParams = FP8SRMatmulParams(
                m: UInt32(pixelCount), n: 32, k: UInt32(kPadded)
            )
            try encodeMatmul(
                encoder: encoder,
                activation: activationBuffer,
                weights: layer.weightBuffer,
                output: layerOutput,
                params: &matmulParams,
                weightDType: layer.weightDType,
                pixelCount: pixelCount
            )
            var postParams = FP8SRPostParams(
                pixelCount: UInt32(pixelCount), channels: 32, scale: layer.manifest.scale
            )
            try encodePostprocess(
                encoder: encoder,
                values: layerOutput,
                bias: layer.bias,
                params: &postParams,
                error: errorBuffer
            )
            previousOutput = layerOutput
        }
        guard let finalOutput = previousOutput else {
            commandBuffer.endCommandBuffer()
            throw FP8SRError.execution("no_layers")
        }
        var pixelParams = FP8SRPixelParams(
            width: UInt32(image.width), height: UInt32(image.height),
            pixelCount: UInt32(pixelCount), channels: 32
        )
        try encodePixelShuffle(
            encoder: encoder,
            values: finalOutput,
            destination: outputBuffer,
            params: &pixelParams
        )
        encoder.endEncoding()
        residencySet.commit()
        commandBuffer.useResidencySet(residencySet)
        commandBuffer.endCommandBuffer()
        let completionValue = nextCompletionValue
        nextCompletionValue += 1
        commandQueue.commit([commandBuffer])
        // MTL4 queue events are ordered against work already submitted to the
        // queue. Signal only after committing this command buffer; signaling
        // before commit would let the CPU read back the zero-filled buffer.
        commandQueue.signalEvent(completionEvent, value: completionValue)
        completionEvent.wait(untilSignaledValue: completionValue, timeoutMS: 120_000)
        guard completionEvent.signaledValue >= completionValue else {
            throw FP8SRError.execution("gpu_timeout")
        }
        let errorValue = errorBuffer.contents().assumingMemoryBound(to: UInt32.self).pointee
        guard errorValue == 0 else {
            throw FP8SRError.execution("nonfinite")
        }
        try FileManager.default.createDirectory(
            at: URL(fileURLWithPath: outputPath).deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        guard let context = CGContext(
            data: outputBuffer.contents(),
            width: image.width * 2,
            height: image.height * 2,
            bitsPerComponent: 8,
            bytesPerRow: image.width * 2 * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw FP8SRError.execution("output_context")
        }
        guard let outputImage = context.makeImage() else {
            throw FP8SRError.execution("output_image")
        }
        guard let destination = CGImageDestinationCreateWithURL(
            URL(fileURLWithPath: outputPath) as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw FP8SRError.execution("output_destination")
        }
        CGImageDestinationAddImage(destination, outputImage, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw FP8SRError.execution("output_write")
        }
        if emitCompletionLog {
            let dtype = layers.first?.weightDType ?? "unknown"
            print("tensorops_dispatch=completed dtype=\(dtype) activation=Float16 accumulation=Float16 output=\(image.width * 2)x\(image.height * 2)")
        }
    }
}

@available(macOS 27.0, *)
func fp8TensorOpsUpscale(inputPath: String, outputPath: String, packPath: String) -> Bool {
    do {
        let captureRequested = ProcessInfo.processInfo.environment["MTL_CAPTURE_WAIT_FOR_SIGNAL"] == "1"
        let runtime = try FP8SRRuntime.cached(packPath: packPath)
        try runtime.upscale(inputPath: inputPath, outputPath: outputPath)
        if captureRequested {
            let configuredWait = ProcessInfo.processInfo.environment[
                "ORTHO4XP_GPU_CAPTURE_WAIT_SECONDS"
            ].flatMap(Double.init) ?? 3.0
            Thread.sleep(forTimeInterval: max(0.5, configuredWait))
        }
        return true
    } catch {
        reportError("ASHelper: TensorOps fallback reason=\(error)")
        return false
    }
}

func convert(inputPath: String, outputPath: String, format: String, useGPU: Bool) -> Bool {
    guard format == "BC1" || format == "BC3" else {
        fail("ASHelper supports BC1/BC3 output. Use nvcompress for BC7 output.")
    }
    let url = URL(fileURLWithPath: inputPath)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        reportError("ASHelper: Failed to load source image '\(inputPath)'.")
        return false
    }
    let width = image.width
    let height = image.height
    let mode: UInt32 = format == "BC3" ? 1 : 0
    let mipCount = UInt32(floor(log2(Double(max(width, height)))) + 1)
    let topLevelSize = UInt32(
        ((width + 3) / 4) * ((height + 3) / 4) * (mode == 0 ? 8 : 16)
    )
    let header = DDSHeader(
        height: UInt32(height),
        width: UInt32(width),
        pitchOrLinearSize: topLevelSize,
        mipmapCount: mipCount,
        fourCC: mode == 0 ? 0x31545844 : 0x35545844
    )
    var output = header.toData()
    let payload: Data?
    if useGPU {
        payload = compressWithMipmaps(cgImage: image, mode: mode)
            ?? compressImageWithCPUMipmaps(image, mode: mode)
    } else {
        payload = compressImageWithCPUMipmaps(image, mode: mode)
    }
    guard let payload else {
        reportError("ASHelper: Failed to compress image '\(inputPath)'.")
        return false
    }
    output.append(payload)
    return writeDDS(output, to: outputPath)
}

// MARK: - Resident JSON Lines server

// The server protocol deliberately carries only paths and scalar options.  It
// keeps the large image payloads on the staging filesystem and lets the
// Python-side tile scheduler apply backpressure without starting one helper
// process per batch.
private func serverString(_ request: [String: Any], _ key: String) -> String? {
    guard let value = request[key] as? String, !value.isEmpty else { return nil }
    return value
}

private func serverTaskID(_ task: [String: Any], index: Int) -> String {
    return (task["id"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        ?? "task-\(index + 1)"
}

private func serverDouble(
    _ task: [String: Any],
    _ key: String,
    defaultValue: Double
) -> Double {
    if let number = task[key] as? NSNumber { return number.doubleValue }
    if let string = task[key] as? String, let value = Double(string) { return value }
    return defaultValue
}

private func serverTaskResult(
    _ taskID: String,
    success: Bool,
    backend: String,
    error: String? = nil,
    extra: [String: Any] = [:]
) -> [String: Any] {
    var result: [String: Any] = [
        "id": taskID,
        "ok": success,
        "backend": backend,
    ]
    if let error { result["error"] = error }
    for (key, value) in extra { result[key] = value }
    return result
}

private func serverConvertBatch(_ request: [String: Any]) -> [[String: Any]] {
    let useGPU = (request["gpu"] as? Bool) ?? false
    guard let tasks = request["tasks"] as? [[String: Any]], !tasks.isEmpty else {
        return [serverTaskResult("batch", success: false, backend: "server", error: "tasks_required")]
    }

    var results = Array(repeating: [String: Any](), count: tasks.count)
    let resultLock = NSLock()
    DispatchQueue.concurrentPerform(iterations: tasks.count) { index in
        let task = tasks[index]
        let taskID = serverTaskID(task, index: index)
        guard let input = serverString(task, "input"),
              let output = serverString(task, "output"),
              let format = serverString(task, "format"),
              format == "BC1" || format == "BC3" else {
            resultLock.lock()
            results[index] = serverTaskResult(
                taskID,
                success: false,
                backend: "server",
                error: "invalid_task"
            )
            resultLock.unlock()
            return
        }

        let ok = convertWithPreprocess(
            jpegPath: input,
            maskPath: serverString(task, "mask") ?? "none",
            r: serverDouble(task, "r", defaultValue: 1.0),
            g: serverDouble(task, "g", defaultValue: 1.0),
            b: serverDouble(task, "b", defaultValue: 1.0),
            contrast: serverDouble(task, "contrast", defaultValue: 1.0),
            brightness: serverDouble(task, "brightness", defaultValue: 0.0),
            saturation: serverDouble(task, "saturation", defaultValue: 1.0),
            outputPath: output,
            format: format,
            useGPU: useGPU
        )
        resultLock.lock()
        results[index] = serverTaskResult(
            taskID,
            success: ok,
            backend: useGPU ? "metal" : "cpu",
            error: ok ? nil : "conversion_failed"
        )
        resultLock.unlock()
    }
    return results
}

@available(macOS 13.0, *)
private func serverMetalFXUpscaleBatch(_ request: [String: Any]) -> [[String: Any]] {
    guard let tasks = request["tasks"] as? [[String: Any]], !tasks.isEmpty else {
        return [serverTaskResult("batch", success: false, backend: "metalfx", error: "tasks_required")]
    }
    let runtime = try? MetalFXSpatialRuntime()
    return tasks.enumerated().map { index, task in
        let taskID = serverTaskID(task, index: index)
        guard let input = serverString(task, "input"),
              let output = serverString(task, "output") else {
            return serverTaskResult(taskID, success: false, backend: "metalfx", error: "invalid_task")
        }
        let result = metalFXSpatialProcess(
            runtime: runtime,
            inputPath: input,
            outputPath: output,
            fallbackAlphaProcessing: true
        )
        return serverTaskResult(
            taskID,
            success: result.success,
            backend: result.effectiveBackend,
            error: result.success ? nil : (result.reason ?? "metalfx_failed"),
            extra: [
                "alpha_mode": result.alphaMode,
                "fallback": result.effectiveBackend == "ci_lanczos",
            ]
        )
    }
}

@available(macOS 27.0, *)
private func serverTensorOpsUpscaleBatch(_ request: [String: Any]) -> [[String: Any]] {
    guard let pack = serverString(request, "pack"),
          let tasks = request["tasks"] as? [[String: Any]],
          !tasks.isEmpty else {
        return [serverTaskResult("batch", success: false, backend: "tensorops", error: "pack_and_tasks_required")]
    }
    let runtime = try? FP8SRRuntime.cached(packPath: pack)
    return tasks.enumerated().map { index, task in
        let taskID = serverTaskID(task, index: index)
        guard let input = serverString(task, "input"),
              let output = serverString(task, "output"),
              let runtime else {
            return serverTaskResult(taskID, success: false, backend: "tensorops", error: "tensorops_runtime_unavailable")
        }
        do {
            try autoreleasepool {
                try runtime.upscale(inputPath: input, outputPath: output)
            }
            return serverTaskResult(taskID, success: true, backend: "tensorops")
        } catch {
            return serverTaskResult(
                taskID,
                success: false,
                backend: "tensorops",
                error: String(describing: error)
            )
        }
    }
}

private func serverUnsupportedBatch(
    _ request: [String: Any],
    backend: String,
    operation: String
) -> [[String: Any]] {
    let tasks = request["tasks"] as? [[String: Any]] ?? []
    if tasks.isEmpty {
        return [serverTaskResult("batch", success: false, backend: backend, error: "tasks_required")]
    }
    return tasks.enumerated().map { index, task in
        serverTaskResult(
            serverTaskID(task, index: index),
            success: false,
            backend: backend,
            error: "unsupported_\(operation)"
        )
    }
}

private func serverResponse(
    id: String,
    operation: String,
    results: [[String: Any]],
    error: String? = nil,
    shutdown: Bool = false
) -> [String: Any] {
    var response: [String: Any] = [
        "id": id,
        "op": operation,
        "ok": error == nil && results.allSatisfy { ($0["ok"] as? Bool) == true },
        "results": results,
    ]
    if let error { response["error"] = error }
    if shutdown { response["shutdown"] = true }
    return response
}

private func runJSONLRequest(_ request: [String: Any]) -> ([String: Any], Bool) {
    let requestID = (request["id"] as? String) ?? UUID().uuidString
    let operation = (request["op"] as? String) ?? ""
    switch operation {
    case "convert_batch":
        return (serverResponse(
            id: requestID,
            operation: operation,
            results: serverConvertBatch(request)
        ), false)
    case "metalfx_upscale_batch":
        guard #available(macOS 13.0, *) else {
            return (serverResponse(
                id: requestID,
                operation: operation,
                results: serverUnsupportedBatch(request, backend: "metalfx", operation: operation)
            ), false)
        }
        return (serverResponse(
            id: requestID,
            operation: operation,
            results: serverMetalFXUpscaleBatch(request)
        ), false)
    case "tensorops_upscale_batch":
        guard #available(macOS 27.0, *) else {
            return (serverResponse(
                id: requestID,
                operation: operation,
                results: serverUnsupportedBatch(request, backend: "tensorops", operation: operation)
            ), false)
        }
        return (serverResponse(
            id: requestID,
            operation: operation,
            results: serverTensorOpsUpscaleBatch(request)
        ), false)
    case "mask_blur_batch", "dem_smooth_batch":
        let results = serverUnsupportedBatch(request, backend: "cpu", operation: operation)
        return (serverResponse(id: requestID, operation: operation, results: results), false)
    case "shutdown":
        return (serverResponse(id: requestID, operation: operation, results: [], shutdown: true), true)
    default:
        return (serverResponse(
            id: requestID,
            operation: operation,
            results: [],
            error: "unsupported_operation"
        ), false)
    }
}

private func runJSONLinesServer() -> Int32 {
    let responseFileDescriptor = dup(STDOUT_FILENO)
    guard responseFileDescriptor >= 0 else {
        reportError("ASHelper: --serve could not duplicate stdout.")
        return 1
    }
    // Existing image/Metal routines use print for diagnostics.  Keep those
    // diagnostics on stderr so stdout stays a strict one-request/one-response
    // JSON Lines stream.
    guard dup2(STDERR_FILENO, STDOUT_FILENO) >= 0 else {
        close(responseFileDescriptor)
        reportError("ASHelper: --serve could not redirect diagnostics.")
        return 1
    }
    let responseHandle = FileHandle(
        fileDescriptor: responseFileDescriptor,
        closeOnDealloc: true
    )
    defer { responseHandle.closeFile() }

    func writeResponse(_ response: [String: Any]) {
        do {
            let data = try JSONSerialization.data(withJSONObject: response, options: [])
            responseHandle.write(data)
            responseHandle.write(Data([0x0a]))
        } catch {
            reportError("ASHelper: --serve response serialization failed: \(error)")
        }
    }

    while let line = readLine() {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { continue }
        guard let data = trimmed.data(using: .utf8) else {
            writeResponse(serverResponse(
                id: UUID().uuidString,
                operation: "unknown",
                results: [],
                error: "request_not_utf8"
            ))
            continue
        }
        do {
            guard let request = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw NSError(domain: "ASHelper", code: 2, userInfo: [
                    NSLocalizedDescriptionKey: "request_object_required"
                ])
            }
            let (response, shouldShutdown) = runJSONLRequest(request)
            writeResponse(response)
            if shouldShutdown { break }
        } catch {
            writeResponse(serverResponse(
                id: UUID().uuidString,
                operation: "unknown",
                results: [],
                error: String(describing: error)
            ))
        }
    }
    return 0
}

let args = ProcessInfo.processInfo.arguments
guard args.count >= 2 else { fail("ASHelper: missing command.") }
if args[1] == "--serve" {
    guard args.count == 2 else { fail("ASHelper: --serve takes no arguments.") }
    exit(runJSONLinesServer())
}
else if args[1] == "--capabilities" {
    guard args.count == 2 else { fail("ASHelper: --capabilities takes no arguments.") }
    print("metal_available=\(MetalCompressor.shared != nil)")
    print("metalfx_spatial_available=\(metalFXSpatialAvailable())")
    var tensorops = false
    if #available(macOS 27.0, *) {
        tensorops = tensorOpsAvailable()
    }
    print("tensorops_available=\(tensorops)")
    print("fp8_tensorops_available=\(tensorops)")
}
else if args[1] == "--ci-lanczos-upscale" || args[1] == "--lanczos-upscale" || args[1] == "--upscale" {
    // The older spellings remain compatibility aliases.
    guard args.count == 4 else { fail("ASHelper: --ci-lanczos-upscale expects input and output paths.") }
    if !lanczosUpscale(inputPath: args[2], outputPath: args[3]) {
        exit(1)
    }
}
else if args[1] == "--metalfx-spatial-upscale" {
    guard args.count == 4 else { fail("ASHelper: --metalfx-spatial-upscale expects input and output paths.") }
    guard #available(macOS 13.0, *) else {
        fail("ASHelper: MetalFX Spatial requires macOS 13 or newer.")
    }
    if !metalFXSpatialUpscale(inputPath: args[2], outputPath: args[3]) {
        exit(1)
    }
}
else if args[1] == "--metalfx-spatial-upscale-batch" {
    guard args.count >= 4 else {
        fail("ASHelper: --metalfx-spatial-upscale-batch expects at least one input/output pair.")
    }
    guard (args.count - 2) % 2 == 0 else {
        fail("ASHelper: --metalfx-spatial-upscale-batch argument count is invalid.")
    }
    guard #available(macOS 13.0, *) else {
        fail("ASHelper: MetalFX Spatial requires macOS 13 or newer.")
    }
    var pairs: [(String, String)] = []
    var index = 2
    while index + 1 < args.count {
        pairs.append((args[index], args[index + 1]))
        index += 2
    }
    if !metalFXSpatialUpscaleBatch(pairs: pairs) {
        exit(1)
    }
}
else if args[1] == "--metalfx-spatial-dds-batch" {
    guard args.count == 3 else {
        fail("ASHelper: --metalfx-spatial-dds-batch expects a request JSON path.")
    }
    guard #available(macOS 13.0, *) else {
        fail("ASHelper: MetalFX Spatial requires macOS 13 or newer.")
    }
    if !metalFXSpatialDDSBatch(requestPath: args[2]) {
        exit(1)
    }
}
else if args[1] == "--tensorops-dds-batch" || args[1] == "--fp8-tensorops-dds-batch" {
    guard args.count == 3 else {
        fail("ASHelper: --tensorops-dds-batch expects a request JSON path.")
    }
    guard #available(macOS 27.0, *) else {
        fail("ASHelper: TensorOps requires macOS 27 or newer.")
    }
    if !tensorOpsDirectDDSBatch(requestPath: args[2]) {
        exit(1)
    }
}
else if args[1] == "--tensorops-upscale" || args[1] == "--fp8-tensorops-upscale" {
    guard args.count == 5 else { fail("ASHelper: --tensorops-upscale expects pack, input, and output paths.") }
    guard #available(macOS 27.0, *) else {
        fail("ASHelper: TensorOps requires macOS 27 or newer.")
    }
    if !fp8TensorOpsUpscale(inputPath: args[3], outputPath: args[4], packPath: args[2]) {
        exit(1)
    }
}
else if args[1] == "--tensorops-upscale-batch" || args[1] == "--fp8-tensorops-upscale-batch" {
    guard args.count >= 5 else { fail("ASHelper: --tensorops-upscale-batch expects pack and at least one input/output pair.") }
    guard (args.count - 3) % 2 == 0 else { fail("ASHelper: --tensorops-upscale-batch argument count is invalid.") }
    guard #available(macOS 27.0, *) else {
        fail("ASHelper: TensorOps requires macOS 27 or newer.")
    }
    do {
        let runtime = try FP8SRRuntime.cached(packPath: args[2])
        var index = 3
        while index + 1 < args.count {
            try autoreleasepool {
                try runtime.upscale(inputPath: args[index], outputPath: args[index + 1])
            }
            index += 2
        }
    } catch {
        reportError("ASHelper: TensorOps batch fallback reason=\(error)")
        exit(1)
    }
}
else if args[1] == "--convert" {
    guard args.count >= 4 else { fail("ASHelper: --convert expects input and output paths.") }
    if !convert(inputPath: args[2], outputPath: args[3], format: args.count > 4 ? args[4] : "BC3", useGPU: args.contains("--gpu")) {
        exit(1)
    }
}
else if args[1] == "--convert-batch" {
    guard args.count >= 6 else { fail("ASHelper: --convert-batch expects a format, GPU flag, and at least one pair.") }
    guard (args.count - 4) % 2 == 0 else { fail("ASHelper: --convert-batch argument count is invalid.") }
    let format = args[2]
    guard format != "BC7" else {
        fail("ASHelper does not support BC7 output. Use nvcompress instead.")
    }
    let useGPU = args[3] == "true"
    var idx = 4
    while idx + 1 < args.count {
        if !convert(inputPath: args[idx], outputPath: args[idx+1], format: format, useGPU: useGPU) {
            exit(1)
        }
        idx += 2
    }
}
else if args[1] == "--convert-batch-v2" {
    guard args.count >= 6 else { fail("ASHelper: --convert-batch-v2 expects a GPU flag and at least one triplet.") }
    guard (args.count - 3) % 3 == 0 else { fail("ASHelper: --convert-batch-v2 argument count is invalid.") }
    let useGPU = args[2] == "true"
    var idx = 3
    while idx + 2 < args.count {
        if !convert(inputPath: args[idx], outputPath: args[idx+1], format: args[idx+2], useGPU: useGPU) {
            exit(1)
        }
        idx += 3
    }
}
else if args[1] == "--convert-batch-v3" {
    guard args.count >= 13 else { fail("ASHelper: --convert-batch-v3 expects a GPU flag and at least one task.") }
    guard (args.count - 3) % 10 == 0 else { fail("ASHelper: --convert-batch-v3 argument count is invalid.") }
    let useGPU = args[2] == "true"
    
    struct BatchTask {
        let jpeg: String
        let mask: String
        let r: Double
        let g: Double
        let b: Double
        let contrast: Double
        let brightness: Double
        let saturation: Double
        let output: String
        let format: String
    }
    
    var tasks: [BatchTask] = []
    var idx = 3
    while idx + 9 < args.count {
        tasks.append(BatchTask(
            jpeg: args[idx],
            mask: args[idx+1],
            r: Double(args[idx+2]) ?? 1.0,
            g: Double(args[idx+3]) ?? 1.0,
            b: Double(args[idx+4]) ?? 1.0,
            contrast: Double(args[idx+5]) ?? 1.0,
            brightness: Double(args[idx+6]) ?? 0.0,
            saturation: Double(args[idx+7]) ?? 1.0,
            output: args[idx+8],
            format: args[idx+9]
        ))
        idx += 10
    }
    
    if tasks.contains(where: { $0.format == "BC7" }) {
        fail("ASHelper does not support BC7 output. Use nvcompress instead.")
    }
    
    let failureState = BatchFailureState()
    let concurrencyLimit = min(8, max(1, tasks.count))
    let semaphore = DispatchSemaphore(value: concurrencyLimit)
    DispatchQueue.concurrentPerform(iterations: tasks.count) { i in
        semaphore.wait()
        defer { semaphore.signal() }
        let t = tasks[i]
        if !convertWithPreprocess(jpegPath: t.jpeg, maskPath: t.mask, r: t.r, g: t.g, b: t.b, contrast: t.contrast, brightness: t.brightness, saturation: t.saturation, outputPath: t.output, format: t.format, useGPU: useGPU) {
            failureState.recordFailure(
                index: i,
                input: t.jpeg,
                mask: t.mask,
                output: t.output
            )
        }
    }
    if failureState.hasFailure() {
        failureState.reportFailures()
        exit(1)
    }
}
