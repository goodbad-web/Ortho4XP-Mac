import Foundation
import CoreImage
import Metal

print("metal_probe_version=1")

guard let device = MTLCreateSystemDefaultDevice() else {
    print("metal_available=false")
    print("metal_device=unavailable")
    exit(0)
}

print("metal_available=true")
print("metal_device=\(device.name)")
print("metal_registry_id=\(device.registryID)")
print("metal_max_threads=\(device.maxThreadsPerThreadgroup.width)x\(device.maxThreadsPerThreadgroup.height)x\(device.maxThreadsPerThreadgroup.depth)")

let exactDescriptor = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: .rgba8Unorm,
    width: 8,
    height: 8,
    mipmapped: true
)
exactDescriptor.usage = [.shaderRead, .shaderWrite]

if let exactTexture = device.makeTexture(descriptor: exactDescriptor) {
    print("ashelper_texture=available")
    let view = exactTexture.makeTextureView(
        pixelFormat: exactTexture.pixelFormat,
        textureType: exactTexture.textureType,
        levels: 0..<1,
        slices: 0..<1
    )
    print("ashelper_texture_view=\(view == nil ? "unavailable" : "available")")
} else {
    print("ashelper_texture=unavailable")
}

let ciContext = CIContext(mtlDevice: device)
print("coreimage_metal_context=available")

let renderDescriptor = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: .rgba8Unorm,
    width: 8,
    height: 8,
    mipmapped: false
)
renderDescriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]

if let renderTexture = device.makeTexture(descriptor: renderDescriptor) {
    let image = CIImage(color: CIColor(red: 1, green: 0, blue: 0, alpha: 1))
        .cropped(to: CGRect(x: 0, y: 0, width: 8, height: 8))
    ciContext.render(
        image,
        to: renderTexture,
        commandBuffer: nil,
        bounds: image.extent,
        colorSpace: CGColorSpaceCreateDeviceRGB()
    )
    print("coreimage_metal_render=issued")
} else {
    print("coreimage_metal_render=texture-unavailable")
}

guard let queue = device.makeCommandQueue(),
      let commandBuffer = queue.makeCommandBuffer(),
      let texture = device.makeTexture(descriptor: exactDescriptor),
      let encoder = commandBuffer.makeBlitCommandEncoder() else {
    print("command_buffer_probe=unavailable")
    exit(0)
}

encoder.generateMipmaps(for: texture)
encoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

print("command_buffer_status=\(commandBuffer.status.rawValue)")
if let error = commandBuffer.error {
    print("command_buffer_error=\(error.localizedDescription)")
} else {
    print("command_buffer_error=none")
}
