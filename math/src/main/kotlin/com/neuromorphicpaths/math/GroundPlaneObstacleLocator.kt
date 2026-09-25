package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.CameraPose
import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.NormalizedBox
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.ObstacleLocator
import kotlin.math.atan
import kotlin.math.tan

/**
 * Places a detection by where its box meets the ground.
 *
 * Bearing comes from how far the box center sits from the frame center, scaled by the
 * horizontal field of view. Range comes from the box's bottom edge: the camera's height and
 * pitch say how far down the horizon is, and a pixel row that far below the horizon lands on
 * the ground at a known distance. When the bottom edge sits on the frame edge, the object's foot
 * is probably out of view, so the box height and a typical height for the class stand in.
 */
class GroundPlaneObstacleLocator(
    private val heightPriors: ObstacleHeightPriors = ObstacleHeightPriors(),
) : ObstacleLocator {

    override fun locate(detections: List<Detection>, frame: Frame): List<Obstacle> {
        val halfTanHorizontal = tan(frame.intrinsics.horizontalFovRadians / 2.0)
        val halfTanVertical = halfTanHorizontal * frame.height / frame.width
        return detections.mapNotNull { detection ->
            val bearingInCamera = atan((detection.box.centerX - 0.5) * 2.0 * halfTanHorizontal)
            val range = rangeFromGroundContact(detection.box, frame.pose, halfTanVertical)
                ?: rangeFromHeightPrior(detection, halfTanVertical)
                ?: return@mapNotNull null
            Obstacle(
                detection = detection,
                bearingRadians = bearingInCamera + frame.pose.yawRadians,
                rangeMeters = range,
                closingSpeedMetersPerSecond = null,
            )
        }
    }

    private fun rangeFromGroundContact(box: NormalizedBox, pose: CameraPose, halfTanVertical: Double): Double? {
        // A box touching the bottom edge has its foot out of frame, so the contact point is unknown.
        if (box.bottom >= BOTTOM_EDGE_CUTOFF) return null
        val angleBelowAxis = atan((box.bottom - 0.5) * 2.0 * halfTanVertical)
        val angleBelowHorizon = pose.pitchRadians + angleBelowAxis
        // At or above the horizon the ray never reaches the ground.
        if (angleBelowHorizon <= MIN_ANGLE_BELOW_HORIZON_RADIANS) return null
        return pose.heightMeters / tan(angleBelowHorizon)
    }

    private fun rangeFromHeightPrior(detection: Detection, halfTanVertical: Double): Double? {
        val realHeightMeters = heightPriors.heightMeters(detection.obstacleClass) ?: return null
        if (detection.box.height <= 0.0) return null
        // Pinhole camera: an object of real height H filling a fraction f of the frame height
        // sits at H / (f * 2 * tan(vfov / 2)).
        return realHeightMeters / (detection.box.height * 2.0 * halfTanVertical)
    }

    private companion object {
        const val BOTTOM_EDGE_CUTOFF = 0.98

        // Below this the range blows up into hundreds of meters, which is noise, not a reading.
        const val MIN_ANGLE_BELOW_HORIZON_RADIANS = 0.01
    }
}
