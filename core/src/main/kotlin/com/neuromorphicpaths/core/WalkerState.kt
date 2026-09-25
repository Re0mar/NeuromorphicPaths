package com.neuromorphicpaths.core

/**
 * What the walker is doing right now, as far as the pipeline knows.
 *
 * Heading is radians relative to the camera's forward axis, positive to the right, so a walker
 * who keeps their head forward has heading zero. Speed is null until something measures it.
 */
data class WalkerState(
    val headingRadians: Double,
    val speedMetersPerSecond: Double?,
) {
    companion object {
        /** The head-forward assumption: walking where the camera points, speed unknown. */
        val ALIGNED_WITH_CAMERA = WalkerState(headingRadians = 0.0, speedMetersPerSecond = null)
    }
}
