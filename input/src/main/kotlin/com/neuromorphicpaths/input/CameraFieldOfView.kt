package com.neuromorphicpaths.input

import android.hardware.camera2.CameraCharacteristics
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
import com.neuromorphicpaths.core.CameraIntrinsics
import kotlin.math.atan

/**
 * Horizontal field of view of the upright image, from what the camera reports about itself.
 *
 * The sensor's physical width and the lens focal length give the angle across the sensor.
 * When the image is rotated to stand upright, the sensor's short side becomes the image's
 * width, so that side's angle is the one that counts. Digital stabilization and non-4:3
 * streams crop the sensor and make this slightly wide. Null when the camera does not report
 * enough to compute it.
 */
@OptIn(ExperimentalCamera2Interop::class)
internal fun horizontalFieldOfView(cameraInfo: CameraInfo, uprightRotationDegrees: Int): CameraIntrinsics? {
    val characteristics = Camera2CameraInfo.from(cameraInfo)
    val sensorSize = characteristics.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return null
    val focalLengthMm = characteristics.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
        ?.firstOrNull()
        ?: return null
    if (focalLengthMm <= 0f) return null

    // The active array can be a little smaller than the full pixel array. Scale the physical
    // size down by the same ratio so the angle matches what the stream actually covers.
    val pixelArray = characteristics.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
    val activeArray = characteristics.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
    val widthFraction = if (pixelArray != null && activeArray != null && pixelArray.width > 0) activeArray.width().toDouble() / pixelArray.width else 1.0
    val heightFraction = if (pixelArray != null && activeArray != null && pixelArray.height > 0) activeArray.height().toDouble() / pixelArray.height else 1.0

    val sideways = uprightRotationDegrees % 180 != 0
    val imageWidthMm = if (sideways) sensorSize.height * heightFraction else sensorSize.width * widthFraction
    return CameraIntrinsics(horizontalFovRadians = 2.0 * atan(imageWidthMm / (2.0 * focalLengthMm)))
}
