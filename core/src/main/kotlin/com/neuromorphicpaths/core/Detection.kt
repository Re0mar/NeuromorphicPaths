package com.neuromorphicpaths.core

/**
 * A box in frame coordinates, every edge as a fraction of the frame size from the top-left.
 *
 * Fractions rather than pixels, so a detector that resizes its input does not have to know the
 * size of the frame it was handed.
 */
data class NormalizedBox(
    val left: Double,
    val top: Double,
    val right: Double,
    val bottom: Double,
) {
    init {
        require(left in 0.0..1.0 && right in 0.0..1.0 && top in 0.0..1.0 && bottom in 0.0..1.0) {
            "Box edges must lie in 0..1, got ($left, $top, $right, $bottom)"
        }
        require(left <= right && top <= bottom) { "Box is inside out: ($left, $top, $right, $bottom)" }
    }

    val centerX: Double get() = (left + right) / 2.0
    val width: Double get() = right - left
    val height: Double get() = bottom - top
}

/**
 * One object a detector found in one frame.
 *
 * The track id is the same integer across consecutive frames when the detector tracks, and
 * null when it does not. Closing speed needs it later, nothing else does.
 */
data class Detection(
    val obstacleClass: ObstacleClass,
    val confidence: Double,
    val box: NormalizedBox,
    val trackId: Int? = null,
)
