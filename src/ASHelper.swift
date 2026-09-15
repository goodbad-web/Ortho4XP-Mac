import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Vision
import CoreImage
import Metal
import MetalKit
import MetalFX

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

let metalSource = """
#include <metal_stdlib>
using namespace metal;
kernel void compressTexture(texture2d<float, access::read> input [[texture(0)]], device uchar *output [[buffer(0)]], constant uint &formatMode [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
    uint2 pos = gid * 4; if (pos.x >= input.get_width() || pos.y >= input.get_height()) return;
    float3 minC = float3(1.0); float3 maxC = float3(0.0); float minA = 1.0; float maxA = 0.0;
    for (uint y = 0; y < 4; y++) { for (uint x = 0; x < 4; x++) {
        uint2 readPos = min(pos + uint2(x, y), uint2(input.get_width() - 1, input.get_height() - 1));
        float4 color = input.read(readPos);
        minC = min(minC, color.rgb); maxC = max(maxC, color.rgb); minA = min(minA, color.a); maxA = max(maxA, color.a);
    }}
    uint blocksPerRow = (input.get_width() + 3) / 4;
    uint offset = (gid.y * blocksPerRow + gid.x) * (formatMode == 0 ? 8 : 16);
    if (formatMode >= 1) {
        output[offset] = (uchar)(maxA * 255.0); output[offset + 1] = (uchar)(minA * 255.0);
        uint64_t aIndices = 0;
        float a0 = maxA;
        float a1 = minA;
        float step = (a0 - a1) / 7.0;
        for (int i = 0; i < 16; i++) {
            uint2 readPos = min(pos + uint2(i % 4, i / 4), uint2(input.get_width() - 1, input.get_height() - 1));
            float a = input.read(readPos).a;
            uint index = 0;
            if (a0 > a1) {
                float minDist = abs(a - a0);
                for (uint j = 1; j <= 6; j++) {
                    float val = a0 - float(j) * step;
                    float dist = abs(a - val);
                    if (dist < minDist) { minDist = dist; index = j + 1; }
                }
                if (abs(a - a1) < minDist) { index = 1; }
            }
            aIndices |= ((uint64_t)index << (i * 3));
        }
        for (int i = 0; i < 6; i++) output[offset + 2 + i] = (uchar)((aIndices >> (i * 8)) & 0xFF);
        offset += 8;
    }
    ushort c0 = ((ushort)(maxC.r * 31.0) << 11) | ((ushort)(maxC.g * 63.0) << 5) | (ushort)(maxC.b * 31.0);
    ushort c1 = ((ushort)(minC.r * 31.0) << 11) | ((ushort)(minC.g * 63.0) << 5) | (ushort)(minC.b * 31.0);
    output[offset] = c0 & 0xFF; output[offset + 1] = c0 >> 8; output[offset + 2] = c1 & 0xFF; output[offset + 3] = c1 >> 8;
    uint32_t indices = 0;
    float3 c2 = (2.0 * maxC + minC) / 3.0;
    float3 c3 = (maxC + 2.0 * minC) / 3.0;
    for (int i = 0; i < 16; i++) {
        uint2 readPos = min(pos + uint2(i % 4, i / 4), uint2(input.get_width() - 1, input.get_height() - 1));
        float3 pixel = input.read(readPos).rgb;
        float3 diff0 = pixel - maxC; float d0 = dot(diff0, diff0);
        float3 diff1 = pixel - minC; float d1 = dot(diff1, diff1);
        float3 diff2 = pixel - c2;   float d2 = dot(diff2, diff2);
        float3 diff3 = pixel - c3;   float d3 = dot(diff3, diff3);
        uint index = 0;
        float minDist = d0;
        if (d1 < minDist) { minDist = d1; index = 1; }
        if (d2 < minDist) { minDist = d2; index = 2; }
        if (d3 < minDist) { minDist = d3; index = 3; }
        indices |= (index << (i * 2));
    }
    output[offset + 4] = indices & 0xFF; output[offset + 5] = (indices >> 8) & 0xFF; output[offset + 6] = (indices >> 16) & 0xFF; output[offset + 7] = (indices >> 24) & 0xFF;
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
    uint pixelCount = params.width * params.height;
    uint total = pixelCount * params.targetK;
    if (gid >= total) return;
    uint pixel = gid / params.targetK;
    uint feature = gid % params.targetK;
    if (feature >= params.kernelSize * params.kernelSize * params.sourceChannels) {
        destination[gid] = half(0.0h);
        return;
    }
    uint inputChannel = feature % params.sourceChannels;
    uint kernelIndex = feature / params.sourceChannels;
    int x = int(pixel % params.width) + int(kernelIndex % params.kernelSize) - int(params.kernelSize / 2);
    int y = int(pixel / params.width) + int(kernelIndex / params.kernelSize) - int(params.kernelSize / 2);
    x = clamp(x, 0, int(params.width) - 1);
    y = clamp(y, 0, int(params.height) - 1);
    float4 value = source.read(uint2(x, y));
    destination[gid] = half(inputChannel == 0 ? value.r : (inputChannel == 1 ? value.g : value.b));
}

kernel void fp8sr_im2col_features(
    device const half *source [[buffer(0)]],
    device half *destination [[buffer(1)]],
    constant FP8SRIm2ColParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    uint pixelCount = params.width * params.height;
    uint total = pixelCount * params.targetK;
    if (gid >= total) return;
    uint pixel = gid / params.targetK;
    uint feature = gid % params.targetK;
    if (feature >= params.kernelSize * params.kernelSize * params.sourceChannels) {
        destination[gid] = half(0.0h);
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
    destination[gid] = source[sourcePixel * params.sourceStride + inputChannel];
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

func appendCPUCompressedDDS(finalCI: CIImage, bounds: CGRect, mode: UInt32, out: inout Data) -> Bool {
    let ctx = CIContext(options: [.useSoftwareRenderer: false])
    guard let cgImage = ctx.createCGImage(finalCI, from: bounds) else { return false }
    let raw = getRawRGBA(cgImage: cgImage)
    let w = Int(bounds.width)
    let h = Int(bounds.height)
    let bW = (w + 3) / 4
    let bH = (h + 3) / 4
    var dds = Data()
    for by in 0..<bH {
        for bx in 0..<bW {
            if mode >= 1 {
                var minA: UInt8 = 255
                var maxA: UInt8 = 0
                for i in 0..<16 {
                    let a = raw[(min(by * 4 + i / 4, h - 1) * w + min(bx * 4 + i % 4, w - 1)) * 4 + 3]
                    minA = min(minA, a)
                    maxA = max(maxA, a)
                }
                var ab = [UInt8](repeating: 0, count: 8)
                ab[0] = maxA
                ab[1] = minA
                var ai: UInt64 = 0
                let a0 = Double(maxA)
                let a1 = Double(minA)
                let step = (a0 - a1) / 7.0
                for i in 0..<16 {
                    let a = Double(raw[(min(by * 4 + i / 4, h - 1) * w + min(bx * 4 + i % 4, w - 1)) * 4 + 3])
                    var index: UInt64 = 0
                    if a0 > a1 {
                        var minDist = abs(a - a0)
                        for j in 1...6 {
                            let val = a0 - Double(j) * step
                            let dist = abs(a - val)
                            if dist < minDist { minDist = dist; index = UInt64(j + 1) }
                        }
                        if abs(a - a1) < minDist { index = 1 }
                    }
                    ai |= (index << (i * 3))
                }
                for i in 0..<6 { ab[i + 2] = UInt8((ai >> (i * 8)) & 0xFF) }
                dds.append(contentsOf: ab)
            }

            var minC = (r: 255, g: 255, b: 255)
            var maxC = (r: 0, g: 0, b: 0)
            for i in 0..<16 {
                let o = (min(by * 4 + i / 4, h - 1) * w + min(bx * 4 + i % 4, w - 1)) * 4
                let r = Int(raw[o]), g = Int(raw[o + 1]), b = Int(raw[o + 2])
                if (r + g + b) < (minC.r + minC.g + minC.b) { minC = (r, g, b) }
                if (r + g + b) > (maxC.r + maxC.g + maxC.b) { maxC = (r, g, b) }
            }
            let c0 = UInt16(((UInt32(maxC.r) >> 3) << 11) | ((UInt32(maxC.g) >> 2) << 5) | (UInt32(maxC.b) >> 3))
            let c1 = UInt16(((UInt32(minC.r) >> 3) << 11) | ((UInt32(minC.g) >> 2) << 5) | (UInt32(minC.b) >> 3))
            let r0 = Double(maxC.r), g0 = Double(maxC.g), b0 = Double(maxC.b)
            let r1 = Double(minC.r), g1 = Double(minC.g), b1 = Double(minC.b)
            let r2 = (2.0 * r0 + r1) / 3.0, g2 = (2.0 * g0 + g1) / 3.0, b2 = (2.0 * b0 + b1) / 3.0
            let r3 = (r0 + 2.0 * r1) / 3.0, g3 = (g0 + 2.0 * g1) / 3.0, b3 = (b0 + 2.0 * b1) / 3.0
            var blk = [UInt8](repeating: 0, count: 8)
            blk[0] = UInt8(c0 & 0xFF)
            blk[1] = UInt8(c0 >> 8)
            blk[2] = UInt8(c1 & 0xFF)
            blk[3] = UInt8(c1 >> 8)
            var idx: UInt32 = 0
            for i in 0..<16 {
                let o = (min(by * 4 + i / 4, h - 1) * w + min(bx * 4 + i % 4, w - 1)) * 4
                let r = Double(raw[o]), g = Double(raw[o + 1]), b = Double(raw[o + 2])
                let d0 = (r - r0) * (r - r0) + (g - g0) * (g - g0) + (b - b0) * (b - b0)
                let d1 = (r - r1) * (r - r1) + (g - g1) * (g - g1) + (b - b1) * (b - b1)
                let d2 = (r - r2) * (r - r2) + (g - g2) * (g - g2) + (b - b2) * (b - b2)
                let d3 = (r - r3) * (r - r3) + (g - g3) * (g - g3) + (b - b3) * (b - b3)
                var index: UInt32 = 0
                var minDist = d0
                if d1 < minDist { minDist = d1; index = 1 }
                if d2 < minDist { minDist = d2; index = 2 }
                if d3 < minDist { minDist = d3; index = 3 }
                idx |= (index << (i * 2))
            }
            blk[4] = UInt8(idx & 0xFF)
            blk[5] = UInt8((idx >> 8) & 0xFF)
            blk[6] = UInt8((idx >> 16) & 0xFF)
            blk[7] = UInt8((idx >> 24) & 0xFF)
            dds.append(contentsOf: blk)
        }
    }
    out.append(dds)
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

func convertWithPreprocess(jpegPath: String, maskPath: String, r: Double, g: Double, b: Double, contrast: Double, brightness: Double, saturation: Double, outputPath: String, format: String, useGPU: Bool) -> Bool {
    guard format != "BC7" else {
        fail("ASHelper does not support BC7 output. Use nvcompress instead.")
    }
    let jpegURL = URL(fileURLWithPath: jpegPath)
    guard let srcCI = CIImage(contentsOf: jpegURL) else {
        reportError("ASHelper: Failed to load source image '\(jpegPath)'.")
        return false
    }
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
               let loaded = CIImage(contentsOf: maskURL),
               let hasExplicitAlpha = imageHasExplicitAlpha(at: maskURL) {
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
    var hdr = DDSHeader(height: UInt32(h), width: UInt32(w), pitchOrLinearSize: sz, mipmapCount: mipCount, fourCC: isBC7 ? 0x30315844 : (isBC3 ? 0x35545844 : 0x31545844))
    var out = hdr.toData()
    if isBC7 { out.append(DDSHeaderDX10(dxgiFormat: 98).toData()) }
    var compressed = false
    if useGPU, let gData = compressWithPreprocessedCIImage(finalCI: finalCI, mode: formatCode, useGPU: useGPU) {
        out.append(gData)
        compressed = true
    }
    if !compressed {
        hdr.mipmapCount = 1
        out = hdr.toData()
        if isBC7 { out.append(DDSHeaderDX10(dxgiFormat: 98).toData()) }
        if !appendCPUCompressedDDS(finalCI: finalCI, bounds: bounds, mode: formatCode, out: &out) {
            reportError("ASHelper: Failed to compress image '\(jpegPath)'.")
            return false
        }
    }

    return writeDDS(out, to: outputPath)
}

func lanczosUpscale(inputPath: String, outputPath: String) -> Bool {
    let url = URL(fileURLWithPath: inputPath)
    guard let ci = CIImage(contentsOf: url), let f = CIFilter(name: "CILanczosScaleTransform") else {
        reportError("ASHelper: Failed to load image or create Lanczos upscale filter for '\(inputPath)'.")
        return false
    }
    f.setValue(ci, forKey: kCIInputImageKey); f.setValue(2.0, forKey: kCIInputScaleKey)
    guard let out = f.outputImage, let cg = CIContext(options: nil).createCGImage(out, from: out.extent) else {
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
func metalFXSpatialUpscale(inputPath: String, outputPath: String) -> Bool {
    guard let device = MTLCreateSystemDefaultDevice() else {
        reportError("ASHelper: MetalFX Spatial requires a Metal device.")
        return false
    }
    guard MTLFXSpatialScalerDescriptor.supportsDevice(device) else {
        reportError("ASHelper: MetalFX Spatial is not supported by '\(device.name)'.")
        return false
    }
    let sourceURL = URL(fileURLWithPath: inputPath)
    guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        reportError("ASHelper: Failed to load MetalFX Spatial input '\(inputPath)'.")
        return false
    }

    let raw = getRawRGBA(cgImage: image)
    guard raw.count == image.width * image.height * 4 else {
        reportError("ASHelper: Failed to normalize MetalFX Spatial input '\(inputPath)'.")
        return false
    }
    if stride(from: 3, to: raw.count, by: 4).contains(where: { raw[$0] < 255 }) {
        reportError("ASHelper: MetalFX Spatial requires an opaque input image (alpha must be 255).")
        return false
    }

    let descriptor = MTLFXSpatialScalerDescriptor()
    descriptor.colorTextureFormat = .rgba8Unorm
    descriptor.outputTextureFormat = .rgba8Unorm
    descriptor.inputWidth = image.width
    descriptor.inputHeight = image.height
    descriptor.outputWidth = image.width * 2
    descriptor.outputHeight = image.height * 2
    descriptor.colorProcessingMode = .perceptual
    guard let scaler = descriptor.makeSpatialScaler(device: device) else {
        reportError("ASHelper: Failed to create MetalFX Spatial scaler.")
        return false
    }

    let inputDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm,
        width: image.width,
        height: image.height,
        mipmapped: false
    )
    inputDescriptor.storageMode = .shared
    inputDescriptor.usage = scaler.colorTextureUsage
    guard let inputTexture = device.makeTexture(descriptor: inputDescriptor) else {
        reportError("ASHelper: Failed to allocate MetalFX Spatial input texture.")
        return false
    }
    raw.withUnsafeBytes { bytes in
        inputTexture.replace(
            region: MTLRegionMake2D(0, 0, image.width, image.height),
            mipmapLevel: 0,
            withBytes: bytes.baseAddress!,
            bytesPerRow: image.width * 4
        )
    }

    let outputDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm,
        width: image.width * 2,
        height: image.height * 2,
        mipmapped: false
    )
    outputDescriptor.storageMode = .private
    outputDescriptor.usage = scaler.outputTextureUsage
    guard let outputTexture = device.makeTexture(descriptor: outputDescriptor),
          let queue = device.makeCommandQueue(),
          let commandBuffer = queue.makeCommandBuffer(),
          let readback = device.makeBuffer(
              length: image.width * image.height * 16,
              options: .storageModeShared
          ) else {
        reportError("ASHelper: Failed to allocate MetalFX Spatial output resources.")
        return false
    }

    scaler.colorTexture = inputTexture
    scaler.inputContentWidth = image.width
    scaler.inputContentHeight = image.height
    scaler.outputTexture = outputTexture
    scaler.encode(commandBuffer: commandBuffer)
    guard let blit = commandBuffer.makeBlitCommandEncoder() else {
        reportError("ASHelper: Failed to create MetalFX Spatial readback encoder.")
        return false
    }
    blit.copy(
        from: outputTexture,
        sourceSlice: 0,
        sourceLevel: 0,
        sourceOrigin: MTLOriginMake(0, 0, 0),
        sourceSize: MTLSizeMake(image.width * 2, image.height * 2, 1),
        to: readback,
        destinationOffset: 0,
        destinationBytesPerRow: image.width * 2 * 4,
        destinationBytesPerImage: image.width * image.height * 16
    )
    blit.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    guard commandBuffer.status == .completed else {
        reportError(
            "ASHelper: MetalFX Spatial command buffer failed: "
                + (commandBuffer.error?.localizedDescription ?? "unknown error")
        )
        return false
    }

    guard let outputContext = CGContext(
        data: readback.contents(),
        width: image.width * 2,
        height: image.height * 2,
        bitsPerComponent: 8,
        bytesPerRow: image.width * 2 * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ), let outputImage = outputContext.makeImage(),
          let destination = CGImageDestinationCreateWithURL(
              URL(fileURLWithPath: outputPath) as CFURL,
              UTType.png.identifier as CFString,
              1,
              nil
          ) else {
        reportError("ASHelper: Failed to create MetalFX Spatial output '\(outputPath)'.")
        return false
    }
    CGImageDestinationAddImage(destination, outputImage, nil)
    guard CGImageDestinationFinalize(destination) else {
        reportError("ASHelper: Failed to write MetalFX Spatial output '\(outputPath)'.")
        return false
    }
    return true
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
          device.makeMTL4CommandQueue() != nil else {
        return false
    }
    return true
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
private struct FP8SRIm2ColParams {
    var width: UInt32
    var height: UInt32
    var sourceChannels: UInt32
    var sourceStride: UInt32
    var targetK: UInt32
    var kernelSize: UInt32
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
    private let matmulPipeline: MTLComputePipelineState
    private let postprocessPipeline: MTLComputePipelineState
    private let pixelShufflePipeline: MTLComputePipelineState
    private let layers: [Layer]
    private var transientBuffers: [MTLBuffer] = []
    private var nextCompletionValue: UInt64 = 1

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
            reportError("ASHelper: FP8 TensorOps shader compile detail=\(error)")
            throw FP8SRError.unavailable("fp8_shader_compile")
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
            self.matmulPipeline = try makePipeline("fp8sr_matmul")
            self.postprocessPipeline = try makePipeline("fp8sr_postprocess")
            self.pixelShufflePipeline = try makePipeline("fp8sr_pixel_shuffle")
        } catch {
            reportError("ASHelper: FP8 TensorOps pipeline compile detail=\(error)")
            throw FP8SRError.unavailable("fp8_pipeline_compile")
        }

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
            weightDescriptor.strides = MTLTensorExtents([1, 128])!
            weightDescriptor.dataType = MTLTensorDataType(rawValue: 142)!
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
                weights: weightsTensor,
                weightBuffer: weightsBuffer,
                bias: biasBuffer
            ))
            residencySet.addAllocation(weightsBuffer)
            residencySet.addAllocation(biasBuffer)
        }
        residencySet.commit()
        self.layers = loadedLayers
        print("fp8_dispatch=ready dtype=MetalFloat8E4M3 activation=Float16 accumulation=Float16 layout=NHWC "
            + "simdgroup=\(matmulPipeline.threadExecutionWidth) threadgroup=\(matmulPipeline.maxTotalThreadsPerThreadgroup)")
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
            && manifest.version == 1
            && manifest.upscaleFactor == 2
            && manifest.layout == "NHWC"
            && manifest.inputChannels == 3
            && manifest.outputChannels == 3
            && manifest.weightDType == "MetalFloat8E4M3"
            && manifest.activationDType == "Float16"
            && manifest.accumulationDType == "Float16"
            && manifest.weightRowStrideBytes == 128
        guard exact else { throw FP8SRError.invalid("manifest_contract") }
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
            guard kPadded % 32 == 0, outPadded % 32 == 0 else {
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
        pixelCount: Int
    ) throws {
        let paramsBuffer = try makeParamsBuffer(&params)
        argumentTable.setTexture(image.gpuResourceID, index: 0)
        argumentTable.setAddress(destination.gpuAddress, index: 1)
        argumentTable.setAddress(paramsBuffer.gpuAddress, index: 2)
        encoder.setComputePipelineState(imageIm2ColPipeline)
        encoder.setArgumentTable(argumentTable)
        encoder.dispatchThreads(
            threadsPerGrid: MTLSize(width: pixelCount * Int(params.targetK), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encodeDispatchBarrier(encoder)
    }

    private func encodeFeatureIm2Col(
        encoder: MTL4ComputeCommandEncoder,
        source: MTLBuffer,
        destination: MTLBuffer,
        params: inout FP8SRIm2ColParams,
        pixelCount: Int
    ) throws {
        let paramsBuffer = try makeParamsBuffer(&params)
        argumentTable.setAddress(source.gpuAddress, index: 0)
        argumentTable.setAddress(destination.gpuAddress, index: 1)
        argumentTable.setAddress(paramsBuffer.gpuAddress, index: 2)
        encoder.setComputePipelineState(featureIm2ColPipeline)
        encoder.setArgumentTable(argumentTable)
        encoder.dispatchThreads(
            threadsPerGrid: MTLSize(width: pixelCount * Int(params.targetK), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encodeDispatchBarrier(encoder)
    }

    private func encodeMatmul(
        encoder: MTL4ComputeCommandEncoder,
        activation: MTLBuffer,
        weights: MTLBuffer,
        output: MTLBuffer,
        params: inout FP8SRMatmulParams,
        pixelCount: Int
    ) throws {
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

    func upscale(inputPath: String, outputPath: String) throws {
        let sourceURL = URL(fileURLWithPath: inputPath)
        guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw FP8SRError.execution("input_decode")
        }
        let raw = getRawRGBA(cgImage: image)
        guard raw.count == image.width * image.height * 4 else {
            throw FP8SRError.execution("input_normalize")
        }
        if stride(from: 3, to: raw.count, by: 4).contains(where: { raw[$0] < 255 }) {
            throw FP8SRError.execution("alpha")
        }
        guard image.width > 0, image.height > 0,
              image.width <= 8192, image.height <= 8192 else {
            throw FP8SRError.execution("input_dimensions")
        }
        let pixelCount = image.width * image.height
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
                  length: max(128, image.width * image.height * 16),
                  options: .storageModeShared
              ) else {
            throw FP8SRError.execution("command_buffer")
        }
        errorBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: errorBuffer.length)
        outputBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: outputBuffer.length)
        transientBuffers.removeAll(keepingCapacity: true)
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
                data: Data(count: max(128, kPadded * pixelCount * 2))
            )
            residencySet.addAllocation(activationBuffer)
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
                    targetK: UInt32(kPadded), kernelSize: UInt32(layer.manifest.kernel)
                )
                try encodeImageIm2Col(
                    encoder: encoder,
                    image: inputTexture,
                    destination: activationBuffer,
                    params: &params,
                    pixelCount: pixelCount
                )
            } else if let previousOutput {
                var params = FP8SRIm2ColParams(
                    width: UInt32(image.width), height: UInt32(image.height),
                    sourceChannels: UInt32(layer.manifest.inChannels), sourceStride: 32,
                    targetK: UInt32(kPadded), kernelSize: UInt32(layer.manifest.kernel)
                )
                try encodeFeatureIm2Col(
                    encoder: encoder,
                    source: previousOutput,
                    destination: activationBuffer,
                    params: &params,
                    pixelCount: pixelCount
                )
            }

            let outputLength = max(128, 32 * pixelCount * 2 + 128)
            guard let layerOutput = device.makeBuffer(length: outputLength, options: .storageModeShared) else {
                throw FP8SRError.execution("layer_output")
            }
            layerOutput.contents().initializeMemory(as: UInt8.self, repeating: 0, count: layerOutput.length)
            residencySet.addAllocation(layerOutput)
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
        print("fp8_dispatch=completed dtype=MetalFloat8E4M3 accumulation=Float16 output=\(image.width * 2)x\(image.height * 2)")
    }
}

@available(macOS 27.0, *)
func fp8TensorOpsUpscale(inputPath: String, outputPath: String, packPath: String) -> Bool {
    do {
        let runtime = try FP8SRRuntime.cached(packPath: packPath)
        try runtime.upscale(inputPath: inputPath, outputPath: outputPath)
        return true
    } catch {
        reportError("ASHelper: FP8 TensorOps fallback reason=\(error)")
        return false
    }
}

func convert(inputPath: String, outputPath: String, format: String, useGPU: Bool) -> Bool {
    guard format != "BC7" else {
        fail("ASHelper does not support BC7 output. Use nvcompress instead.")
    }
    let url = URL(fileURLWithPath: inputPath)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        reportError("ASHelper: Failed to load source image '\(inputPath)'.")
        return false
    }
    let w = img.width; let h = img.height; let isBC7 = (format == "BC7"); let isBC3 = (format == "BC3")
    let formatCode: UInt32 = isBC7 ? 2 : (isBC3 ? 1 : 0)
    let mipCount = UInt32(floor(log2(Double(max(w, h)))) + 1)
    let sz = UInt32(((w + 3) / 4) * ((h + 3) / 4) * (formatCode == 0 ? 8 : 16))
    var hdr = DDSHeader(height: UInt32(h), width: UInt32(w), pitchOrLinearSize: sz, mipmapCount: mipCount, fourCC: isBC7 ? 0x30315844 : (isBC3 ? 0x35545844 : 0x31545844))
    var out = hdr.toData()
    if isBC7 { out.append(DDSHeaderDX10(dxgiFormat: 98).toData()) }

    var compressed = false
    if useGPU, let gData = compressWithMipmaps(cgImage: img, mode: formatCode) {
        out.append(gData)
        compressed = true
    }
    if !compressed {
        hdr.mipmapCount = 1
        out = hdr.toData()
        if isBC7 { out.append(DDSHeaderDX10(dxgiFormat: 98).toData()) }
        let raw = getRawRGBA(cgImage: img)
        var dds = Data()
        let bW = (w+3)/4
        let bH = (h+3)/4
        for by in 0..<bH { for bx in 0..<bW {
            if formatCode >= 1 {
                var minA: UInt8 = 255; var maxA: UInt8 = 0
                for i in 0..<16 { let a = raw[(min(by*4+i/4,h-1)*w+min(bx*4+i%4,w-1))*4+3]; minA=min(minA,a); maxA=max(maxA,a) }
                var ab=[UInt8](repeating:0,count:8); ab[0]=maxA; ab[1]=minA; var ai:UInt64=0
                let a0 = Double(maxA); let a1 = Double(minA); let step = (a0 - a1) / 7.0
                for i in 0..<16 {
                    let a = Double(raw[(min(by*4+i/4,h-1)*w+min(bx*4+i%4,w-1))*4+3])
                    var index: UInt64 = 0
                    if a0 > a1 {
                        var minDist = abs(a - a0)
                        for j in 1...6 {
                            let val = a0 - Double(j) * step
                            let dist = abs(a - val)
                            if dist < minDist { minDist = dist; index = UInt64(j + 1) }
                        }
                        if abs(a - a1) < minDist { index = 1 }
                    }
                    ai |= (index << (i * 3))
                }
                for i in 0..<6 { ab[i+2] = UInt8((ai >> (i * 8)) & 0xFF) }; dds.append(contentsOf: ab)
            }
            var minC=(r:255,g:255,b:255); var maxC=(r:0,g:0,b:0)
            for i in 0..<16 { let o=(min(by*4+i/4,h-1)*w+min(bx*4+i%4,w-1))*4; let r=Int(raw[o]),g=Int(raw[o+1]),b=Int(raw[o+2]); if (r+g+b)<(minC.r+minC.g+minC.b){minC=(r,g,b)}; if (r+g+b)>(maxC.r+maxC.g+maxC.b){maxC=(r,g,b)} }
            let c0=UInt16(((UInt32(maxC.r)>>3)<<11)|((UInt32(maxC.g)>>2)<<5)|(UInt32(maxC.b)>>3))
            let c1=UInt16(((UInt32(minC.r)>>3)<<11)|((UInt32(minC.g)>>2)<<5)|(UInt32(minC.b)>>3))

            let r0 = Double(maxC.r), g0 = Double(maxC.g), b0 = Double(maxC.b)
            let r1 = Double(minC.r), g1 = Double(minC.g), b1 = Double(minC.b)
            let r2 = (2.0 * r0 + r1) / 3.0, g2 = (2.0 * g0 + g1) / 3.0, b2 = (2.0 * b0 + b1) / 3.0
            let r3 = (r0 + 2.0 * r1) / 3.0, g3 = (g0 + 2.0 * g1) / 3.0, b3 = (b0 + 2.0 * b1) / 3.0

            var blk=[UInt8](repeating:0,count:8); blk[0]=UInt8(c0&0xFF); blk[1]=UInt8(c0>>8); blk[2]=UInt8(c1&0xFF); blk[3]=UInt8(c1>>8); var idx:UInt32=0
            for i in 0..<16 {
                let o=(min(by*4+i/4,h-1)*w+min(bx*4+i%4,w-1))*4
                let r=Double(raw[o]), g=Double(raw[o+1]), b=Double(raw[o+2])
                let d0 = (r-r0)*(r-r0) + (g-g0)*(g-g0) + (b-b0)*(b-b0)
                let d1 = (r-r1)*(r-r1) + (g-g1)*(g-g1) + (b-b1)*(b-b1)
                let d2 = (r-r2)*(r-r2) + (g-g2)*(g-g2) + (b-b2)*(b-b2)
                let d3 = (r-r3)*(r-r3) + (g-g3)*(g-g3) + (b-b3)*(b-b3)
                var index: UInt32 = 0
                var minDist = d0
                if d1 < minDist { minDist = d1; index = 1 }
                if d2 < minDist { minDist = d2; index = 2 }
                if d3 < minDist { minDist = d3; index = 3 }
                idx |= (index << (i * 2))
            }
            blk[4]=UInt8(idx&0xFF); blk[5]=UInt8((idx>>8)&0xFF); blk[6]=UInt8((idx>>16)&0xFF); blk[7]=UInt8((idx>>24)&0xFF); dds.append(contentsOf: blk)
        }}
        out.append(dds)
    }
    return writeDDS(out, to: outputPath)
}

let args = ProcessInfo.processInfo.arguments
guard args.count >= 2 else { fail("ASHelper: missing command.") }
if args[1] == "--capabilities" {
    guard args.count == 2 else { fail("ASHelper: --capabilities takes no arguments.") }
    print("metal_available=\(MetalCompressor.shared != nil)")
    print("metalfx_spatial_available=\(metalFXSpatialAvailable())")
    print("fp8_tensorops_available=\(fp8TensorOpsAvailable())")
}
else if args[1] == "--lanczos-upscale" || args[1] == "--upscale" {
    // --upscale remains as a compatibility alias for older scripts.
    guard args.count == 4 else { fail("ASHelper: --lanczos-upscale expects input and output paths.") }
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
else if args[1] == "--fp8-tensorops-upscale" {
    guard args.count == 5 else { fail("ASHelper: --fp8-tensorops-upscale expects pack, input, and output paths.") }
    guard #available(macOS 27.0, *) else {
        fail("ASHelper: FP8 TensorOps requires macOS 27 or newer.")
    }
    if !fp8TensorOpsUpscale(inputPath: args[3], outputPath: args[4], packPath: args[2]) {
        exit(1)
    }
}
else if args[1] == "--fp8-tensorops-upscale-batch" {
    guard args.count >= 5 else { fail("ASHelper: --fp8-tensorops-upscale-batch expects pack and at least one input/output pair.") }
    guard (args.count - 3) % 2 == 0 else { fail("ASHelper: --fp8-tensorops-upscale-batch argument count is invalid.") }
    guard #available(macOS 27.0, *) else {
        fail("ASHelper: FP8 TensorOps requires macOS 27 or newer.")
    }
    do {
        let runtime = try FP8SRRuntime.cached(packPath: args[2])
        var index = 3
        while index + 1 < args.count {
            try runtime.upscale(inputPath: args[index], outputPath: args[index + 1])
            index += 2
        }
    } catch {
        reportError("ASHelper: FP8 TensorOps batch fallback reason=\(error)")
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
