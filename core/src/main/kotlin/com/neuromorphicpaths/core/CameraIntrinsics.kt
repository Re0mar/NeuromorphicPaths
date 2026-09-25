package com.neuromorphicpaths.core

import kotlin.math.atan
import kotlin.math.tan

/**
 * The one lens number the geometry needs: how wide the camera sees, in radians.
 *
 * The vertical field of view follows from the frame's aspect ratio, assuming square pixels,
 * which holds for every phone and glasses camera this project uses.
 */
data class CameraIntrinsics(
    val horizontalFovRadians: Double,
) {
    init {
        require(horizontalFovRadians > 0.0 && horizontalFovRadians < Math.PI) {
            "Horizontal field of view must be between 0 and pi radians, got $horizontalFovRadians"
        }
    }

    fun verticalFovRadians(width: Int, height: Int): Double =
        2.0 * atan(tan(horizontalFovRadians / 2.0) * height / width)
}
