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
    var flags: UInt32 = 0x1 | 0x2 | 0x4 | 0x1000 // CAPS, HEIGHT, WIDTH, PIXELFORMAT
    var height: UInt32
    var width: UInt32
    var pitchOrLinearSize: UInt32 = 0
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
    
    func toData() -> Data {
        var data = Data()
        var temp = self
        withUnsafeBytes(of: &temp) { data.append(contentsOf: $0) }
        return data
    }
}

// MARK: - DXT Encoder
func compressDXT(rgba: UnsafePointer<UInt8>, width: Int, height: Int, isDXT5: Bool) -> Data {
    var ddsData = Data()
    let blocksWide = (width + 3) / 4
    let blocksHigh = (height + 3) / 4
    
    for by in 0..<blocksHigh {
        for bx in 0..<blocksWide {
            if isDXT5 {
                // Simplified Alpha Block (Straight Alpha 255)
                var alphaBlock = [UInt8](repeating: 0xFF, count: 8)
                alphaBlock[0] = 255; alphaBlock[1] = 0
                ddsData.append(contentsOf: alphaBlock)
            }
            
            // Color Block (4x4 pixels)
            var minColor = (r: 255, g: 255, b: 255)
            var maxColor = (r: 0, g: 0, b: 0)
            
            // Scan block for min/max
            for y in 0..<4 {
                for x in 0..<4 {
                    let px = min(bx * 4 + x, width - 1)
                    let py = min(by * 4 + y, height - 1)
                    let offset = (py * width + px) * 4
                    let r = Int(rgba[offset]); let g = Int(rgba[offset+1]); let b = Int(rgba[offset+2])
                    if (r + g + b) < (minColor.r + minColor.g + minColor.b) { minColor = (r, g, b) }
                    if (r + g + b) > (maxColor.r + maxColor.g + maxColor.b) { maxColor = (r, g, b) }
                }
            }
            
            let c0 = UInt16(((UInt32(maxColor.r) >> 3) << 11) | ((UInt32(maxColor.g) >> 2) << 5) | (UInt32(maxColor.b) >> 3))
            let c1 = UInt16(((UInt32(minColor.r) >> 3) << 11) | ((UInt32(minColor.g) >> 2) << 5) | (UInt32(minColor.b) >> 3))
            
            var block = [UInt8](repeating: 0, count: 8)
            block[0] = UInt8(c0 & 0xFF); block[1] = UInt8(c0 >> 8)
            block[2] = UInt8(c1 & 0xFF); block[3] = UInt8(c1 >> 8)
            
            // Calculate indices (simplified)
            var indices: UInt32 = 0
            for i in 0..<16 {
                let x = i % 4; let y = i / 4
                let px = min(bx * 4 + x, width - 1); let py = min(by * 4 + y, height - 1)
                let offset = (py * width + px) * 4
                let r = Int(rgba[offset]); let g = Int(rgba[offset+1]); let b = Int(rgba[offset+2])
                let dist0 = abs(r - maxColor.r) + abs(g - maxColor.g) + abs(b - maxColor.b)
                let dist1 = abs(r - minColor.r) + abs(g - minColor.g) + abs(b - minColor.b)
                if dist1 < dist0 { indices |= (1 << (i * 2)) }
            }
            block[4] = UInt8(indices & 0xFF); block[5] = UInt8((indices >> 8) & 0xFF)
            block[6] = UInt8((indices >> 16) & 0xFF); block[7] = UInt8((indices >> 24) & 0xFF)
            
            ddsData.append(contentsOf: block)
        }
    }
    return ddsData
}

func convert(inputPath: String, outputPath: String, format: String) {
    let inputURL = URL(fileURLWithPath: inputPath)
    guard let source = CGImageSourceCreateWithURL(inputURL as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else { exit(1) }
    
    let isDXT5 = (format == "BC3" || format == "BC7")
    let header = DDSHeader(height: UInt32(cgImage.height), width: UInt32(cgImage.width), fourCC: isDXT5 ? 0x35545844 : 0x31545844)
    var finalData = header.toData()
    
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    var rawData = [UInt8](repeating: 0, count: cgImage.width * cgImage.height * 4)
    let context = CGContext(data: &rawData, width: cgImage.width, height: cgImage.height, bitsPerComponent: 8, bytesPerRow: cgImage.width * 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
    
    finalData.append(compressDXT(rgba: rawData, width: cgImage.width, height: cgImage.height, isDXT5: isDXT5))
    try? finalData.write(to: URL(fileURLWithPath: outputPath))
}

func upscale(inputPath: String, outputPath: String) {
    let inputURL = URL(fileURLWithPath: inputPath)
    guard let ciImage = CIImage(contentsOf: inputURL),
          let filter = CIFilter(name: "CILanczosScaleTransform") else { exit(1) }
    filter.setValue(ciImage, forKey: kCIInputImageKey)
    filter.setValue(2.0, forKey: kCIInputScaleKey)
    let context = CIContext(options: [.useSoftwareRenderer: false])
    guard let outputImage = filter.outputImage,
          let cgImage = context.createCGImage(outputImage, from: outputImage.extent) else { exit(1) }
    let destination = CGImageDestinationCreateWithURL(URL(fileURLWithPath: outputPath) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, cgImage, nil)
    CGImageDestinationFinalize(destination)
}

let args = ProcessInfo.processInfo.arguments
if args.count < 4 { exit(1) }
let mode = args[1]; let input = args[2]; let output = args[3]
if mode == "--upscale" { upscale(inputPath: input, outputPath: output) }
else if mode == "--convert" { convert(inputPath: input, outputPath: output, format: args.count > 4 ? args[4] : "BC3") }
