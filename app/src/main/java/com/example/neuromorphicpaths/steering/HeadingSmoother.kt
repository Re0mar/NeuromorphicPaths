package com.example.neuromorphicpaths.steering

class HeadingSmoother(private val alpha: Float = 0.15f) {
    private var smoothedHeading: Float? = null

    fun smooth(heading: Float): Float {
        val current = smoothedHeading
        return if (current == null) {
            smoothedHeading = heading
            heading
        } else {
            val next = alpha * heading + (1f - alpha) * current
            smoothedHeading = next
            next
        }
    }

    fun reset() {
        smoothedHeading = null
    }
}
