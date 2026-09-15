import CoreGraphics
import CoreImage
import CoreML
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("CoreMLReference: " + message + "\n").utf8))
    exit(1)
}

let args = ProcessInfo.processInfo.arguments
guard args.count == 4 else {
    fail("expects model.mlmodelc, input image, and output image")
}

let modelURL = URL(fileURLWithPath: args[1])
let inputURL = URL(fileURLWithPath: args[2])
let outputURL = URL(fileURLWithPath: args[3])

do {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .all
    let model = try MLModel(contentsOf: modelURL, configuration: configuration)
    guard let input = model.modelDescription.inputDescriptionsByName.values.first(where: {
        $0.type == .image
    }) else {
        fail("the model has no image input")
    }
    let inputValue: MLFeatureValue
    if let constraint = input.imageConstraint {
        inputValue = try MLFeatureValue(
            imageAt: inputURL,
            constraint: constraint,
            options: nil
        )
    } else {
        guard let source = CGImageSourceCreateWithURL(inputURL as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            fail("could not decode input image")
        }
        inputValue = try MLFeatureValue(
            cgImage: image,
            pixelsWide: image.width,
            pixelsHigh: image.height,
            pixelFormatType: kCVPixelFormatType_32BGRA,
            options: nil
        )
    }
    let provider = try MLDictionaryFeatureProvider(
        dictionary: [input.name: inputValue]
    )
    let prediction = try model.prediction(from: provider)
    guard let output = model.modelDescription.outputDescriptionsByName.values.first(where: {
        $0.type == .image
    }),
    let pixelBuffer = prediction.featureValue(for: output.name)?.imageBufferValue else {
        fail("the model has no image output")
    }
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let context = CIContext(options: nil)
    guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent),
          let destination = CGImageDestinationCreateWithURL(
              outputURL as CFURL,
              UTType.png.identifier as CFString,
              1,
              nil
          ) else {
        fail("could not create output image")
    }
    try FileManager.default.createDirectory(
        at: outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    CGImageDestinationAddImage(destination, cgImage, nil)
    guard CGImageDestinationFinalize(destination) else {
        fail("could not write output image")
    }
    print("coreml_reference=completed compute_units=all output=\(cgImage.width)x\(cgImage.height)")
} catch {
    fail(error.localizedDescription)
}
