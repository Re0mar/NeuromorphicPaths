package com.neuromorphicpaths.core

/**
 * Where the camera sits and points, relative to the walker.
 *
 * All angles are radians. Pitch is positive when the camera looks down toward the ground, since
 * that is the direction the ground-plane range estimate needs. Yaw is the camera heading relative
 * to the walker's direction of travel, positive to the right. Roll is positive clockwise as seen
 * from behind the camera. Height is the lens above the ground, in meters.
 */
data class CameraPose(
    val pitchRadians: Double,
    val yawRadians: Double,
    val rollRadians: Double,
    val heightMeters: Double,
)
