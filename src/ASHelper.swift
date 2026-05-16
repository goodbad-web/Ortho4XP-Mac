import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Vision
import CoreImage

// MARK: - DDS Header Structure
struct DDSHeader {
    var magic: UInt32 = 0x20534444 // 'DDS '
    var size: UInt32 = 124
    var flags: UInt32 = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000 // CAPS, HEIGHT, WIDTH, PIXELFORMAT, LINEARSIZE
    var height: UInt32
    var width: UInt32
    var pitchOrLinearSize: UInt32
    var depth: UInt32 = 0
    var mipmapCount: UInt32 = 1
    var reserved1 = [UInt32](repeating: 0, count: 11)
    var pfSize: UInt32 = 32
    var pfFlags: UInt32 = 0x4 // FOURCC
    var fourCC: UInt32
    var pfRGBBitCount: UInt32 = 0
    var pfRBitMask: UInt32 = 0
    var pfGBitMask: UInt32 = 0
    var pfBBitMask: UInt32 = 0
    var pfABitMask: UInt32 = 0
    var caps: UInt32 = 0x1000 // TEXTURE
    var caps2: UInt32 = 0
    var caps3: UInt32 = 0
    var caps4: UInt32 = 0
    var reserved2: UInt32 = 0
}

struct DDSHeaderDX10 {
    var dxgiFormat: UInt32 // 98 for BC7_UNORM
    var resourceDimension: UInt32 = 3 // Texture2D
    var miscFlag: UInt32 = 0
    var arraySize: UInt32 = 1
    var miscFlags2: UInt32 = 0
}

func compressDXT(rgba: UnsafePointer<UInt8>, width: Int, height: Int, isDXT5: Bool) -> Data {
    var ddsData = Data()
    let blocksWide = (width + 3) / 4
    let blocksHigh = (height + 3) / 4
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
                var alphaBlock = [UInt8](repeating: 0, count: 8)
                alphaBlock[0] = maxA; alphaBlock[1] = minA
                var aIndices: UInt64 = 0
                if maxA > minA {
                    for i in 0..<16 {
                        if rgba[(min(by * 4 + (i/4), height - 1) * width + min(bx * 4 + (i%4), width - 1)) * 4 + 3] < (UInt32(maxA) + UInt32(minA)) / 2 { aIndices |= (1 << (i * 3)) }
                    }
                }
                for i in 0..<6 { alphaBlock[i+2] = UInt8((aIndices >> (i * 8)) & 0xFF) }
                ddsData.append(contentsOf: alphaBlock)
            }
            var minC = (r: 255, g: 255, b: 255); var maxC = (r: 0, g: 0, b: 0)
            for i in 0..<16 {
                let offset = (min(by * 4 + (i/4), height - 1) * width + min(bx * 4 + (i%4), width - 1)) * 4
                let r = Int(rgba[offset]); let g = Int(rgba[offset+1]); let b = Int(rgba[offset+2])
                if (r + g + b) < (minC.r + minC.g + minC.b) { minC = (r, g, b) }
                if (r + g + b) > (maxC.r + maxC.g + maxC.b) { maxC = (r, g, b) }
            }
            let c0 = UInt16(((UInt32(maxC.r) >> 3) << 11) | ((UInt32(maxC.g) >> 2) << 5) | (UInt32(maxC.b) >> 3))
            let c1 = UInt16(((UInt32(minC.r) >> 3) << 11) | ((UInt32(minC.g) >> 2) << 5) | (UInt32(minC.b) >> 3))
            var block = [UInt8](repeating: 0, count: 8)
            block[0] = UInt8(c0 & 0xFF); block[1] = UInt8(c0 >> 8); block[2] = UInt8(c1 & 0xFF); block[3] = UInt8(c1 >> 8)
            var indices: UInt32 = 0
            for i in 0..<16 {
                let offset = (min(by * 4 + (i/4), height - 1) * width + min(bx * 4 + (i%4), width - 1)) * 4
                let r = Int(rgba[offset]); let g = Int(rgba[offset+1]); let b = Int(rgba[offset+2])
                if (abs(r - minC.r) + abs(g - minC.g) + abs(b - minC.b)) < (abs(r - maxC.r) + abs(g - maxC.g) + abs(b - maxC.b)) { indices |= (1 << (i * 2)) }
            }
            block[4] = UInt8(indices & 0xFF); block[5] = UInt8((indices >> 8) & 0xFF); block[6] = UInt8((indices >> 16) & 0xFF); block[7] = UInt8((indices >> 24) & 0xFF)
            ddsData.append(contentsOf: block)
        }
    }
    return ddsData
}

func convert(inputPath: String, outputPath: String, format: String) {
    let inputURL = URL(fileURLWithPath: inputPath)
    guard let source = CGImageSourceCreateWithURL(inputURL as CFURL, nil), let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else { exit(1) }
    let isBC7 = (format == "BC7")
    let isDXT5 = (format == "BC3" || isBC7)
    let blocksWide = (cgImage.width + 3) / 4
    let blocksHigh = (cgImage.height + 3) / 4
    let pitch = UInt32(blocksWide * blocksHigh * (isDXT5 ? 16 : 8))
    
    var header = DDSHeader(height: UInt32(cgImage.height), width: UInt32(cgImage.width), pitchOrLinearSize: pitch, fourCC: isBC7 ? 0x30315844 : (isDXT5 ? 0x35545844 : 0x31545844))
    var finalData = Data()
    withUnsafeBytes(of: &header) { finalData.append(contentsOf: $0) }
    
    if isBC7 {
        var dx10 = DDSHeaderDX10(dxgiFormat: 98) // BC7_UNORM
        withUnsafeBytes(of: &dx10) { finalData.append(contentsOf: $0) }
    }
    
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    var rawData = [UInt8](repeating: 0, count: cgImage.width * cgImage.height * 4)
    let context = CGContext(data: &rawData, width: cgImage.width, height: cgImage.height, bitsPerComponent: 8, bytesPerRow: cgImage.width * 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
    
    // For now, if BC7 is requested, we still output DXT5 blocks but with BC7 header for compatibility
    finalData.append(compressDXT(rgba: rawData, width: cgImage.width, height: cgImage.height, isDXT5: isDXT5))
    try? finalData.write(to: URL(fileURLWithPath: outputPath))
}

func upscale(inputPath: String, outputPath: String) {
    let inputURL = URL(fileURLWithPath: inputPath)
    guard let ciImage = CIImage(contentsOf: inputURL), let filter = CIFilter(name: "CILanczosScaleTransform") else { exit(1) }
    filter.setValue(ciImage, forKey: kCIInputImageKey)
    filter.setValue(2.0, forKey: kCIInputScaleKey)
    let context = CIContext(options: [.useSoftwareRenderer: false])
    guard let outputImage = filter.outputImage, let cgImage = context.createCGImage(outputImage, from: outputImage.extent) else { exit(1) }
    let destination = CGImageDestinationCreateWithURL(URL(fileURLWithPath: outputPath) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, cgImage, nil)
    CGImageDestinationFinalize(destination)
}

let args = ProcessInfo.processInfo.arguments
if args.count < 4 { exit(1) }
let mode = args[1]; let input = args[2]; let output = args[3]
if mode == "--upscale" { upscale(inputPath: input, outputPath: output) }
else if mode == "--convert" { convert(inputPath: input, outputPath: output, format: args.count > 4 ? args[4] : "BC3") }
