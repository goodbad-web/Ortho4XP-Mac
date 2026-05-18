import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Vision
import CoreImage
import Metal
import MetalKit

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

// Memory Cache for mask images to eliminate repetitive disk I/O
let maskCacheLock = NSLock()
var maskCache: [String: CIImage] = [:]

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
    
    // 1. Create an empty empty MTLTexture in VRAM (with mipmapped = true)
    let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: true)
    desc.usage = [.shaderRead, .shaderWrite]
    guard let mtlTexture = dev.makeTexture(descriptor: desc) else { return nil }
    
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
    if let mipEncoder = cmb.makeBlitCommandEncoder() {
        mipEncoder.generateMipmaps(for: mtlTexture)
        mipEncoder.endEncoding()
    }
    
    // 4. Run Metal compute kernels to compress each mipmap level
    var buffers: [MTLBuffer] = []
    for level in 0..<mtlTexture.mipmapLevelCount {
        let lW = max(1, w >> level); let lH = max(1, h >> level); let bW = (lW + 3) / 4; let bH = (lH + 3) / 4
        let sz = bW * bH * (mode == 0 ? 8 : 16)
        guard let buf = dev.makeBuffer(length: sz, options: .storageModeShared) else { break }
        buffers.append(buf)
        
        guard let enc = cmb.makeComputeCommandEncoder() else { break }
        var m = mode
        enc.setComputePipelineState(pipe)
        let view = mtlTexture.makeTextureView(pixelFormat: mtlTexture.pixelFormat, textureType: mtlTexture.textureType, levels: level..<level+1, slices: 0..<1)
        enc.setTexture(view, index: 0)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBytes(&m, length: 4, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: (bW + 15) / 16, height: (bH + 15) / 16, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
    
    cmb.commit()
    cmb.waitUntilCompleted()
    
    var outData = Data()
    for buf in buffers {
        outData.append(Data(bytes: buf.contents(), count: buf.length))
    }
    return outData
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
    
    guard let cmb = q.makeCommandBuffer() else { return nil }
    var buffers: [MTLBuffer] = []
    
    for level in 0..<tex.mipmapLevelCount {
        let lW = max(1, w >> level); let lH = max(1, h >> level); let bW = (lW + 3) / 4; let bH = (lH + 3) / 4
        let sz = bW * bH * (mode == 0 ? 8 : 16)
        guard let buf = dev.makeBuffer(length: sz, options: .storageModeShared) else { break }
        buffers.append(buf)
        
        guard let enc = cmb.makeComputeCommandEncoder() else { break }
        var m = mode
        enc.setComputePipelineState(pipe)
        let view = tex.makeTextureView(pixelFormat: tex.pixelFormat, textureType: tex.textureType, levels: level..<level+1, slices: 0..<1)
        enc.setTexture(view, index: 0)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBytes(&m, length: 4, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: (bW + 15) / 16, height: (bH + 15) / 16, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
    
    cmb.commit()
    cmb.waitUntilCompleted()
    
    var outData = Data()
    for buf in buffers {
        outData.append(Data(bytes: buf.contents(), count: buf.length))
    }
    return outData
}

func convertWithPreprocess(jpegPath: String, maskPath: String, r: Double, g: Double, b: Double, contrast: Double, brightness: Double, saturation: Double, outputPath: String, format: String, useGPU: Bool) {
    let jpegURL = URL(fileURLWithPath: jpegPath)
    guard let srcCI = CIImage(contentsOf: jpegURL) else { exit(1) }
    var finalCI = srcCI
    
    // 1. Blend mask if present (via in-memory Mask Cache)
    if maskPath != "none" && maskPath != "" {
        var maskCI: CIImage? = nil
        maskCacheLock.lock()
        if let cached = maskCache[maskPath] {
            maskCI = cached
            maskCacheLock.unlock()
        } else {
            maskCacheLock.unlock()
            if FileManager.default.fileExists(atPath: maskPath), let loaded = CIImage(contentsOf: URL(fileURLWithPath: maskPath)) {
                maskCacheLock.lock()
                maskCache[maskPath] = loaded
                maskCI = loaded
                maskCacheLock.unlock()
            }
        }
        
        if let mCI = maskCI {
            // [Fix] Scale the mask to match the high-resolution source image (srcCI) extent,
            // preventing CIBlendWithAlphaMask from downscaling finalCI to the mask's low resolution.
            let scaleX = srcCI.extent.width / mCI.extent.width
            let scaleY = srcCI.extent.height / mCI.extent.height
            var resizedMask = mCI
            if scaleX != 1.0 || scaleY != 1.0 {
                let scaleFilter = CIFilter(name: "CILanczosScaleTransform")!
                scaleFilter.setValue(mCI, forKey: kCIInputImageKey)
                scaleFilter.setValue(scaleX, forKey: kCIInputScaleKey)
                scaleFilter.setValue(scaleY / scaleX, forKey: "inputAspectRatio")
                if let scaled = scaleFilter.outputImage {
                    let originReset = scaled.transformed(by: CGAffineTransform(translationX: -scaled.extent.origin.x, y: -scaled.extent.origin.y))
                    resizedMask = originReset.cropped(to: srcCI.extent)
                }
            }
            
            let blendFilter = CIFilter(name: "CIBlendWithAlphaMask")!
            blendFilter.setValue(srcCI, forKey: kCIInputImageKey)
            blendFilter.setValue(resizedMask, forKey: kCIInputMaskImageKey)
            if let blended = blendFilter.outputImage {
                finalCI = blended
            }
        }
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
    
    if useGPU, let gData = compressWithPreprocessedCIImage(finalCI: finalCI, mode: formatCode, useGPU: useGPU) {
        out.append(gData)
    } else {
        // Fallback for CPU (CPU manual compression)
        let ctx = CIContext(options: [.useSoftwareRenderer: false])
        guard let cgImage = ctx.createCGImage(finalCI, from: bounds) else { exit(1) }
        hdr.mipmapCount = 1
        out = hdr.toData()
        if isBC7 { out.append(DDSHeaderDX10(dxgiFormat: 98).toData()) }
        let raw = getRawRGBA(cgImage: cgImage)
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
    
    try? out.write(to: URL(fileURLWithPath: outputPath))
}

func upscale(inputPath: String, outputPath: String) {
    let url = URL(fileURLWithPath: inputPath); guard let ci = CIImage(contentsOf: url), let f = CIFilter(name: "CILanczosScaleTransform") else { exit(1) }
    f.setValue(ci, forKey: kCIInputImageKey); f.setValue(2.0, forKey: kCIInputScaleKey)
    guard let out = f.outputImage, let cg = CIContext(options: nil).createCGImage(out, from: out.extent) else { exit(1) }
    let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: outputPath) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, cg, nil); CGImageDestinationFinalize(dest)
}

func convert(inputPath: String, outputPath: String, format: String, useGPU: Bool) {
    let url = URL(fileURLWithPath: inputPath); guard let src = CGImageSourceCreateWithURL(url as CFURL, nil), let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { exit(1) }
    let w = img.width; let h = img.height; let isBC7 = (format == "BC7"); let isBC3 = (format == "BC3")
    let formatCode: UInt32 = isBC7 ? 2 : (isBC3 ? 1 : 0)
    let mipCount = UInt32(floor(log2(Double(max(w, h)))) + 1)
    let sz = UInt32(((w + 3) / 4) * ((h + 3) / 4) * (formatCode == 0 ? 8 : 16))
    var hdr = DDSHeader(height: UInt32(h), width: UInt32(w), pitchOrLinearSize: sz, mipmapCount: mipCount, fourCC: isBC7 ? 0x30315844 : (isBC3 ? 0x35545844 : 0x31545844))
    var out = hdr.toData(); if isBC7 { out.append(DDSHeaderDX10(dxgiFormat: 98).toData()) }
    if useGPU, let gData = compressWithMipmaps(cgImage: img, mode: formatCode) { out.append(gData) }
    else {
        hdr.mipmapCount = 1; out = hdr.toData(); if isBC7 { out.append(DDSHeaderDX10(dxgiFormat: 98).toData()) }
        let raw = getRawRGBA(cgImage: img); var dds = Data(); let bW = (w+3)/4; let bH = (h+3)/4
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
    try? out.write(to: URL(fileURLWithPath: outputPath))
}

let args = ProcessInfo.processInfo.arguments; if args.count < 4 { exit(1) }
if args[1] == "--upscale" { upscale(inputPath: args[2], outputPath: args[3]) }
else if args[1] == "--convert" { convert(inputPath: args[2], outputPath: args[3], format: args.count > 4 ? args[4] : "BC3", useGPU: args.contains("--gpu")) }
else if args[1] == "--convert-batch" {
    guard args.count >= 6 else { exit(1) }
    let format = args[2]
    let useGPU = args[3] == "true"
    var idx = 4
    while idx + 1 < args.count {
        convert(inputPath: args[idx], outputPath: args[idx+1], format: format, useGPU: useGPU)
        idx += 2
    }
}
else if args[1] == "--convert-batch-v2" {
    guard args.count >= 6 else { exit(1) }
    let useGPU = args[2] == "true"
    var idx = 3
    while idx + 2 < args.count {
        convert(inputPath: args[idx], outputPath: args[idx+1], format: args[idx+2], useGPU: useGPU)
        idx += 3
    }
}
else if args[1] == "--convert-batch-v3" {
    guard args.count >= 12 else { exit(1) }
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
    
    DispatchQueue.concurrentPerform(iterations: tasks.count) { i in
        let t = tasks[i]
        convertWithPreprocess(jpegPath: t.jpeg, maskPath: t.mask, r: t.r, g: t.g, b: t.b, contrast: t.contrast, brightness: t.brightness, saturation: t.saturation, outputPath: t.output, format: t.format, useGPU: useGPU)
    }
}
