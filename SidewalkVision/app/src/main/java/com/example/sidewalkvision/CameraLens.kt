package com.example.sidewalkvision

import android.content.ContentResolver
import android.hardware.camera2.CameraCharacteristics
import android.media.ExifInterface
import android.net.Uri
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import java.io.IOException
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

/**
 * Intrinsics for a picked photo from its EXIF 35 mm equivalent focal length, or null when the
 * photo doesn't carry one, as screenshots and edited images often don't.
 */
fun photoIntrinsics(contentResolver: ContentResolver, uri: Uri, widthPx: Int, heightPx: Int): CameraIntrinsics? {
    val equivalentFocalLengthMm = try {
        contentResolver.openInputStream(uri)?.use { stream ->
            ExifInterface(stream).getAttributeInt(ExifInterface.TAG_FOCAL_LENGTH_IN_35MM_FILM, 0)
        } ?: 0
    } catch (ioException: IOException) {
        // Unreadable metadata costs the photo its pitch and height, not its detection.
        Log.w("CameraLens", "Could not read EXIF from $uri: ${ioException.message}")
        0
    }
    return intrinsicsFrom35mmEquivalent(equivalentFocalLengthMm.toDouble(), widthPx, heightPx)
}
