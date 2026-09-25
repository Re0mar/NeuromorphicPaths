package com.neuromorphicpaths.core

/**
 * A detection placed in the world around the walker.
 *
 * Bearing is radians from the walker's direction of travel, positive to the right. Range is
 * meters along the ground. Closing speed is meters per second toward the walker, and null until
 * something tracks the object across frames.
 */
data class Obstacle(
    val detection: Detection,
    val bearingRadians: Double,
    val rangeMeters: Double,
    val closingSpeedMetersPerSecond: Double?,
) {
    val obstacleClass: ObstacleClass get() = detection.obstacleClass
}
