package com.neuromorphicpaths.core

/**
 * One obstacle's push on the walker, in the walker's frame. Lateral is positive to the right,
 * forward is positive ahead. Unitless. Only the direction and the relative sizes matter.
 */
data class Push(
    val lateral: Double,
    val forward: Double,
)

/** What one obstacle contributed: its own surprise and the push that came out of it. */
data class ObstacleSurprise(
    val obstacle: Obstacle,
    val surpriseBits: Double,
    val push: Push,
)

/**
 * The field's answer for one frame.
 *
 * Desired heading is radians relative to the camera's forward axis, positive to the right.
 * Overall surprise is how improbable the walker's actual heading is given the field, in bits.
 * The per-obstacle list is for the display and the log, not for adding up.
 */
data class Guidance(
    val desiredHeadingRadians: Double,
    val overallSurpriseBits: Double,
    val perObstacle: List<ObstacleSurprise>,
)
