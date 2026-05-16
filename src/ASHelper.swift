import Foundation
import Vision
import CoreImage
import ImageIO
import UniformTypeIdentifiers

func upscale(inputPath: String, outputPath: String) {
    let inputURL = URL(fileURLWithPath: inputPath)
    
    guard let ciImage = CIImage(contentsOf: inputURL) else {
        print("Failed to load image from \(inputPath)")
        exit(1)
    }
    
    // Fallback to high quality Lanczos scaling (GPU accelerated)
    // In a real scenario, we could use a CoreML model here.
    let filter = CIFilter(name: "CILanczosScaleTransform")!
    filter.setValue(ciImage, forKey: kCIInputImageKey)
    filter.setValue(2.0, forKey: kCIInputScaleKey) // 2x upscale
    filter.setValue(1.0, forKey: kCIInputAspectRatioKey)
    
    guard let outputImage = filter.outputImage else {
        print("Failed to apply upscale filter")
        exit(1)
    }
    
    let context = CIContext(options: [.useSoftwareRenderer: false])
    guard let cgImage = context.createCGImage(outputImage, from: outputImage.extent) else {
        print("Failed to create CGImage")
        exit(1)
    }
    
    let outputURL = URL(fileURLWithPath: outputPath)
    guard let destination = CGImageDestinationCreateWithURL(outputURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        print("Failed to create image destination")
        exit(1)
    }
    
    CGImageDestinationAddImage(destination, cgImage, nil)
    if !CGImageDestinationFinalize(destination) {
        print("Failed to finalize image destination")
        exit(1)
    }
    print("Successfully upscaled to \(outputPath)")
}

let args = ProcessInfo.processInfo.arguments
if args.count < 4 {
    print("Usage: ASHelper --upscale <input> <output>")
    exit(1)
}

let mode = args[1]
let input = args[2]
let output = args[3]

if mode == "--upscale" {
    upscale(inputPath: input, outputPath: output)
} else {
    print("Unknown mode: \(mode)")
    exit(1)
}
