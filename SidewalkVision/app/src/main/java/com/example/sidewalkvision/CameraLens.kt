package com.example.sidewalkvision

import android.hardware.camera2.CameraCharacteristics
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import kotlin.math.max

/**
 * The bound camera's focal length relative to its sensor, read from Camera2, or null when the
 * camera doesn't report it.
 *
 * Focal length over sensor size, both in millimeters, equals focal length over image size in
 * pixels, as long as the image spans the sensor's long side. CameraX's analysis images do, with
 * any cropping falling on the short side.
 */
@OptIn(ExperimentalCamera2Interop::class)
fun cameraIntrinsics(camera: Camera): CameraIntrinsics? {
    val cameraInfo = Camera2CameraInfo.from(camera.cameraInfo)
    val focalLengthMm = cameraInfo.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
        ?.firstOrNull() ?: return null
    val sensorSizeMm = cameraInfo.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return null
    val longSideMm = max(sensorSizeMm.width, sensorSizeMm.height)
    if (focalLengthMm <= 0f || longSideMm <= 0f) return null
    return CameraIntrinsics(focalLengthOverLongSide = focalLengthMm.toDouble() / longSideMm)
}
