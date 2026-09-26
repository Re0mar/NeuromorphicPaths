package com.neuromorphicpaths.core

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.tan

/** A point on the ground in the walker's frame: meters ahead and meters to the right. */
data class GroundPoint(val forwardMeters: Double, val rightMeters: Double)

/** A point in the frame as fractions of its size from the top-left. */
data class FramePoint(val x: Double, val y: Double)

/**
 * The pinhole geometry between frame pixels and the ground plane, both ways.
 *
 * The camera sits [CameraPose.heightMeters] above flat ground, pitched down by the pose's pitch
 * and turned by its yaw relative to the walker's direction of travel. Roll is ignored, as the
 * box locator ignores it. A pixel below the horizon meets the ground at one point, and a ground
 * point ahead of the camera lands on one pixel, so the two functions are inverses on the region
 * where both are defined. On the center column the range agrees with the box locator's
 * height-over-tangent formula exactly.
 */
class GroundPlaneProjection(
    private val pose: CameraPose,
    intrinsics: CameraIntrinsics,
    frameWidth: Int,
    frameHeight: Int,
) {
    private val halfTanHorizontal = tan(intrinsics.horizontalFovRadians / 2.0)
    private val halfTanVertical = halfTanHorizontal * frameHeight / frameWidth
    private val cosPitch = cos(pose.pitchRadians)
    private val sinPitch = sin(pose.pitchRadians)
    private val cosYaw = cos(pose.yawRadians)
    private val sinYaw = sin(pose.yawRadians)

    /**
     * Where the ray through a frame point meets the ground, or null when it never does because
     * the point is at or above the horizon, or so close to it that the range is noise.
     */
    fun groundAt(point: FramePoint): GroundPoint? {
        // The ray in camera axes: one unit forward, some units right, some units up.
        val rightInCamera = (point.x - 0.5) * 2.0 * halfTanHorizontal
        val upInCamera = -(point.y - 0.5) * 2.0 * halfTanVertical
        // The same ray in the walker's axes before yaw, with the camera at the origin. Pitching the
        // camera down by p turns its forward axis toward the ground and its up axis forward.
        val forward = cosPitch + upInCamera * sinPitch
        val down = sinPitch - upInCamera * cosPitch
        if (down <= MIN_DOWNWARD_SLOPE) return null
        val scale = pose.heightMeters / down
        return rotateByYaw(forward * scale, rightInCamera * scale)
    }

    /**
     * The frame point a ground point lands on, or null when it is behind the camera or outside
     * the frame.
     */
    fun frameAt(point: GroundPoint): FramePoint? {
        // Undo the yaw so the point is in the camera's own forward and right axes.
        val forward = point.forwardMeters * cosYaw + point.rightMeters * sinYaw
        val right = -point.forwardMeters * sinYaw + point.rightMeters * cosYaw
        val depth = forward * cosPitch + pose.heightMeters * sinPitch
        if (depth <= 0.0) return null
        val up = forward * sinPitch - pose.heightMeters * cosPitch
        val x = 0.5 + (right / depth) / (2.0 * halfTanHorizontal)
        val y = 0.5 - (up / depth) / (2.0 * halfTanVertical)
        if (x < 0.0 || x > 1.0 || y < 0.0 || y > 1.0) return null
        return FramePoint(x, y)
    }

    private fun rotateByYaw(forward: Double, right: Double): GroundPoint =
        GroundPoint(
            forwardMeters = forward * cosYaw - right * sinYaw,
            rightMeters = forward * sinYaw + right * cosYaw,
        )

    private companion object {
        // A ray this nearly level lands hundreds of meters out, which is noise, not a reading.
        // The same cutoff as the box locator's minimum angle below the horizon, as a slope.
        const val MIN_DOWNWARD_SLOPE = 0.01
    }
}
