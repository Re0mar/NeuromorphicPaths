package com.neuromorphicpaths.core

/**
 * What the walker is doing right now, as far as the pipeline knows.
 *
 * Heading is radians relative to the camera's forward axis, positive to the right, so a walker
 * who keeps their head forward has heading zero. Speed is null until something measures it.
 * Azimuth is where the camera points over the ground, in radians from whatever reference the
 * sensor uses, null when nothing measures it. Only its changes over time are used, so the
 * reference and the sign convention do not matter.
 */
data class WalkerState(
    val headingRadians: Double,
    val speedMetersPerSecond: Double?,
    val azimuthRadians: Double? = null,
) {
    companion object {
        /** The head-forward assumption: walking where the camera points, speed unknown. */
        val ALIGNED_WITH_CAMERA = WalkerState(headingRadians = 0.0, speedMetersPerSecond = null)
    }
}
