import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Vision
import CoreImage
import Metal

// MARK: - Metal Shader Source
let metalSource = """
#include <metal_stdlib>
using namespace metal;

kernel void compressDXT(texture2d<float, access::read> input [[texture(0)]],
                        device uchar *output [[buffer(0)]],
                        constant bool &isDXT5 [[buffer(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint2 pos = gid * 4;
    if (pos.x >= input.get_width() || pos.y >= input.get_height()) return;

    float3 minC = float3(1.0);
    float3 maxC = float3(0.0);
    float minA = 1.0;
    float maxA = 0.0;

    for (uint y = 0; y < 4; y++) {
        for (uint x = 0; x < 4; x++) {
            float4 color = input.read(pos + uint2(x, y));
            minC = min(minC, color.rgb);
            maxC = max(maxC, color.rgb);
            minA = min(minA, color.a);
            maxA = max(maxA, color.a);
        }
    }

    uint offset = (gid.y * ((input.get_width() + 3) / 4) + gid.x) * (isDXT5 ? 16 : 8);
    
    if (isDXT5) {
        output[offset] = (uchar)(maxA * 255.0);
        output[offset + 1] = (uchar)(minA * 255.0);
        uint64_t aIndices = 0;
        float midA = (maxA + minA) / 2.0;
        for (int i = 0; i < 16; i++) {
            float a = input.read(pos + uint2(i % 4, i / 4)).a;
            if (a < midA) aIndices |= (1ULL << (i * 3));
        }
        for (int i = 0; i < 6; i++) output[offset + 2 + i] = (uchar)((aIndices >> (i * 8)) & 0xFF);
        offset += 8;
    }

    ushort c0 = ((ushort)(maxC.r * 31.0) << 11) | ((ushort)(maxC.g * 63.0) << 5) | (ushort)(maxC.b * 31.0);
    ushort c1 = ((ushort)(minC.r * 31.0) << 11) | ((ushort)(minC.g * 63.0) << 5) | (ushort)(minC.b * 31.0);
    output[offset] = c0 & 0xFF; output[offset + 1] = c0 >> 8;
    output[offset + 2] = c1 & 0xFF; output[offset + 3] = c1 >> 8;

    uint32_t indices = 0;
    for (int i = 0; i < 16; i++) {
        float3 c = input.read(pos + uint2(i % 4, i / 4)).rgb;
        if (distance(c, minC) < distance(c, maxC)) indices |= (1 << (i * 2));
    }
    output[offset + 4] = indices & 0xFF; output[offset + 5] = (indices >> 8) & 0xFF;
    output[offset + 6] = (indices >> 16) & 0xFF; output[offset + 7] = (indices >> 24) & 0xFF;
}
"""

struct DDSHeader {
    var magic: UInt32 = 0x20534444; var size: UInt32 = 124; var flags: UInt32 = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000 
    var height: UInt32; var width: UInt32; var pitchOrLinearSize: UInt32
    var depth: UInt32 = 0; var mipmapCount: UInt32 = 1; var reserved1 = [UInt32](repeating: 0, count: 11)
    var pfSize: UInt32 = 32; var pfFlags: UInt32 = 0x4; var fourCC: UInt32
    var pfRGBBitCount: UInt32 = 0; var pfRBitMask: UInt32 = 0; var pfGBitMask: UInt32 = 0; var pfBBitMask: UInt32 = 0; var pfABitMask: UInt32 = 0
    var caps: UInt32 = 0x1000; var caps2: UInt32 = 0; var caps3: UInt32 = 0; var caps4: UInt32 = 0; var reserved2: UInt32 = 0
    func toData() -> Data { var d = Data(); var t = self; withUnsafeBytes(of: &t) { d.append(contentsOf: $0) }; return d }
}

struct DDSHeaderDX10 {
    var dxgiFormat: UInt32; var resourceDimension: UInt32 = 3; var miscFlag: UInt32 = 0; var arraySize: UInt32 = 1; var miscFlags2: UInt32 = 0
    func toData() -> Data { var d = Data(); var t = self; withUnsafeBytes(of: &t) { d.append(contentsOf: $0) }; return d }
}

func compressCPU(rgba: UnsafePointer<UInt8>, width: Int, height: Int, isDXT5: Bool) -> Data {
    var ddsData = Data()
    let blocksWide = (width + 3) / 4; let blocksHigh = (height + 3) / 4
    for by in 0..<blocksHigh {
        for bx in 0..<blocksWide {
            if isDXT5 {
                var minA: UInt8 = 255; var maxA: UInt8 = 0
                for y in 0..<4 {
                    for x in 0..<4 {
                        let a = rgba[(min(by * 4 + y, height - 1) * width + min(bx * 4 + x, width - 1)) * 4 + 3]
                        minA = min(minA, a); maxA = max(maxA, a)
                    }
                }
                var ab = [UInt8](repeating: 0, count: 8); ab[0] = maxA; ab[1] = minA
                var ai: UInt64 = 0
                if maxA > minA {
                    for i in 0..<16 {
                        if rgba[(min(by * 4 + (i/4), height - 1) * width + min(bx * 4 + (i%4), width - 1)) * 4 + 3] < (UInt32(maxA) + UInt32(minA)) / 2 { ai |= (1 << (i * 3)) }
                    }
                }
                for i in 0..<6 { ab[i+2] = UInt8((ai >> (i * 8)) & 0xFF) }; ddsData.append(contentsOf: ab)
            }
            var minC = (r: 255, g: 255, b: 255); var maxC = (r: 0, g: 0, b: 0)
            for i in 0..<16 {
                let off = (min(by * 4 + (i/4), height - 1) * width + min(bx * 4 + (i%4), width - 1)) * 4
                let r = Int(rgba[off]); let g = Int(rgba[off+1]); let b = Int(rgba[off+2])
                if (r + g + b) < (minC.r + minC.g + minC.b) { minC = (r, g, b) }
                if (r + g + b) > (maxC.r + maxC.g + maxC.b) { maxC = (r, g, b) }
            }
            let c0 = UInt16(((UInt32(maxC.r) >> 3) << 11) | ((UInt32(maxC.g) >> 2) << 5) | (UInt32(maxC.b) >> 3))
            let c1 = UInt16(((UInt32(minC.r) >> 3) << 11) | ((UInt32(minC.g) >> 2) << 5) | (UInt32(minC.b) >> 3))
            var blk = [UInt8](repeating: 0, count: 8); blk[0] = UInt8(c0 & 0xFF); blk[1] = UInt8(c0 >> 8); blk[2] = UInt8(c1 & 0xFF); blk[3] = UInt8(c1 >> 8)
            var idx: UInt32 = 0
            for i in 0..<16 {
                let off = (min(by * 4 + (i/4), height - 1) * width + min(bx * 4 + (i%4), width - 1)) * 4
                let r = Int(rgba[off]); let g = Int(rgba[off+1]); let b = Int(rgba[off+2])
                if (abs(r - minC.r) + abs(g - minC.g) + abs(b - minC.b)) < (abs(r - maxC.r) + abs(g - maxC.g) + abs(b - maxC.b)) { idx |= (1 << (i * 2)) }
            }
            blk[4] = UInt8(idx & 0xFF); blk[5] = UInt8((idx >> 8) & 0xFF); blk[6] = UInt8((idx >> 16) & 0xFF); blk[7] = UInt8((idx >> 24) & 0xFF); ddsData.append(contentsOf: blk)
        }
    }
    return ddsData
}

func compressGPU(cgImage: CGImage, isDXT5: Bool) -> Data? {
    guard let dev = MTLCreateSystemDefaultDevice(), let lib = try? dev.makeLibrary(source: metalSource, options: nil),
          let fn = lib.makeFunction(name: "compressDXT"), let pipe = try? dev.makeComputePipelineState(function: fn),
          let q = dev.makeCommandQueue() else { return nil }
    let w = cgImage.width; let h = cgImage.height
    let texD = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false); texD.usage = [.shaderRead]
    guard let tex = dev.makeTexture(descriptor: texD) else { return nil }
    var raw = [UInt8](repeating: 0, count: w * h * 4)
    let ctx = CGContext(data: &raw, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    ctx?.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
    tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: raw, bytesPerRow: w * 4)
    let bW = (w + 3) / 4; let bH = (h + 3) / 4; let sz = bW * bH * (isDXT5 ? 16 : 8)
    guard let buf = dev.makeBuffer(length: sz, options: .storageModeShared), let cmb = q.makeCommandBuffer(), let enc = cmb.makeComputeCommandEncoder() else { return nil }
    var dxt5 = isDXT5; enc.setComputePipelineState(pipe); enc.setTexture(tex, index: 0); enc.setBuffer(buf, offset: 0, index: 0); enc.setBytes(&dxt5, length: 1, index: 1)
    enc.dispatchThreadgroups(MTLSize(width: (bW + 15) / 16, height: (bH + 15) / 16, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    enc.endEncoding(); cmb.commit(); cmb.waitUntilCompleted(); return Data(bytes: buf.contents(), count: sz)
}

func convert(inputPath: String, outputPath: String, format: String, useGPU: Bool) {
    let url = URL(fileURLWithPath: inputPath); guard let src = CGImageSourceCreateWithURL(url as CFURL, nil), let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { exit(1) }
    let bc7 = (format == "BC7"); let dxt5 = (format == "BC3" || bc7); let w = img.width; let h = img.height
    let sz = UInt32(((w + 3) / 4) * ((h + 3) / 4) * (dxt5 ? 16 : 8))
    var hdr = DDSHeader(height: UInt32(h), width: UInt32(w), pitchOrLinearSize: sz, fourCC: bc7 ? 0x30315844 : (dxt5 ? 0x35545844 : 0x31545844))
    var out = hdr.toData(); if bc7 { out.append(DDSHeaderDX10(dxgiFormat: 98).toData()) }
    if useGPU, let gData = compressGPU(cgImage: img, isDXT5: dxt5) { out.append(gData) }
    else {
        var raw = [UInt8](repeating: 0, count: w * h * 4)
        let ctx = CGContext(data: &raw, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        ctx?.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        out.append(compressCPU(rgba: raw, width: w, height: h, isDXT5: dxt5))
    }
    try? out.write(to: URL(fileURLWithPath: outputPath))
}

func upscale(inputPath: String, outputPath: String) {
    let url = URL(fileURLWithPath: inputPath); guard let ci = CIImage(contentsOf: url), let f = CIFilter(name: "CILanczosScaleTransform") else { exit(1) }
    f.setValue(ci, forKey: kCIInputImageKey); f.setValue(2.0, forKey: kCIInputScaleKey)
    guard let out = f.outputImage, let cg = CIContext(options: nil).createCGImage(out, from: out.extent) else { exit(1) }
    let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: outputPath) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, cg, nil); CGImageDestinationFinalize(dest)
}

let args = ProcessInfo.processInfo.arguments; if args.count < 4 { exit(1) }
if args[1] == "--upscale" { upscale(inputPath: args[2], outputPath: args[3]) }
else if args[1] == "--convert" { convert(inputPath: args[2], outputPath: args[3], format: args.count > 4 ? args[4] : "BC3", useGPU: args.contains("--gpu")) }
