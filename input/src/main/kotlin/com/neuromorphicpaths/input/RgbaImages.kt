package com.neuromorphicpaths.input

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import com.neuromorphicpaths.core.CameraIntrinsics
import com.neuromorphicpaths.core.CameraPose
import com.neuromorphicpaths.core.Frame
import java.nio.ByteBuffer

/** Pixels plus size, for the steps between a platform image and a [Frame]. */
internal class RgbaImage(val width: Int, val height: Int, val rgba: ByteArray)

/** Copies an RGBA_8888 analysis image into a tightly packed, upright RGBA array. */
internal fun ImageProxy.toRgbaImage(): RgbaImage {
    val plane = planes[0]
    val buffer = plane.buffer
    val rowBytes = width * Frame.BYTES_PER_PIXEL
    val packed = ByteArray(rowBytes * height)
    if (plane.rowStride == rowBytes) {
        buffer.rewind()
        buffer.get(packed)
    } else {
        // The camera pads rows to its own alignment. Copy row by row to drop the padding.
        for (row in 0 until height) {
            buffer.position(row * plane.rowStride)
            buffer.get(packed, row * rowBytes, rowBytes)
        }
    }
    return RgbaImage(width, height, packed).rotatedClockwise(imageInfo.rotationDegrees)
}

/** Copies a bitmap into a tightly packed RGBA array. ARGB_8888 bitmaps store bytes as R, G, B, A. */
internal fun Bitmap.toRgbaImage(): RgbaImage {
    val source: Bitmap = if (config == Bitmap.Config.ARGB_8888) {
        this
    } else {
        copy(Bitmap.Config.ARGB_8888, false) ?: error("Bitmap could not be converted to ARGB_8888")
    }
    val rowBytes = width * Frame.BYTES_PER_PIXEL
    val packed = ByteArray(rowBytes * height)
    if (source.rowBytes == rowBytes) {
        source.copyPixelsToBuffer(ByteBuffer.wrap(packed))
    } else {
        val padded = ByteBuffer.allocate(source.rowBytes * height)
        source.copyPixelsToBuffer(padded)
        for (row in 0 until height) {
            padded.position(row * source.rowBytes)
            padded.get(packed, row * rowBytes, rowBytes)
        }
    }
    return RgbaImage(width, height, packed)
}

internal fun RgbaImage.toFrame(timestampNanos: Long, pose: CameraPose, intrinsics: CameraIntrinsics): Frame =
    Frame(timestampNanos, width, height, rgba, pose, intrinsics)

/** Rotates by a multiple of 90 degrees so the image reads upright. Cameras report no other angles. */
internal fun RgbaImage.rotatedClockwise(degrees: Int): RgbaImage {
    val quarterTurns = ((degrees % 360) + 360) % 360 / 90
    var image = this
    repeat(quarterTurns) { image = image.rotatedQuarterTurn() }
    return image
}

private fun RgbaImage.rotatedQuarterTurn(): RgbaImage {
    val bytesPerPixel = Frame.BYTES_PER_PIXEL
    val out = ByteArray(rgba.size)
    val outWidth = height
    for (y in 0 until height) {
        val outX = height - 1 - y
        for (x in 0 until width) {
            val src = (y * width + x) * bytesPerPixel
            val dst = (x * outWidth + outX) * bytesPerPixel
            System.arraycopy(rgba, src, out, dst, bytesPerPixel)
        }
    }
    return RgbaImage(width = height, height = width, rgba = out)
}
