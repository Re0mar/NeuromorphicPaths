package com.neuromorphicpaths.core

/**
 * One camera image plus what the later stages need to place objects in the world.
 *
 * Pixels are RGBA, 8 bits per channel, row-major from the top-left, with no row padding, so the
 * array holds exactly width * height * 4 bytes and the image is upright. Every source converts to
 * this once, so detectors, locators and displays never learn where a frame came from.
 */
class Frame(
    val timestampNanos: Long,
    val width: Int,
    val height: Int,
    val rgba: ByteArray,
    val pose: CameraPose,
    val intrinsics: CameraIntrinsics,
) {
    init {
        require(width > 0 && height > 0) { "Frame needs a positive size, got ${width}x$height" }
        require(rgba.size == width * height * BYTES_PER_PIXEL) {
            "Frame of ${width}x$height needs ${width * height * BYTES_PER_PIXEL} bytes, got ${rgba.size}"
        }
    }

    companion object {
        const val BYTES_PER_PIXEL = 4
    }
}
