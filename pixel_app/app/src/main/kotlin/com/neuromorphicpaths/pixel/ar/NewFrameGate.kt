package com.neuromorphicpaths.pixel.ar

/**
 * Lets a frame through only when its timestamp differs from the last one that passed.
 *
 * The GL surface draws at the display rate and ARCore's update with LATEST_CAMERA_IMAGE returns
 * at once with whatever frame it has, so most draws see the frame the previous draw already
 * sent. On the first phone run that was 60 messages a second of which every one repeated its
 * neighbour, byte for byte, and the laptop recorded all of them.
 */
class NewFrameGate {
    private var lastTimestampNanos: Long? = null

    /** True when this timestamp has not been seen as the most recent one. Remembers it either way. */
    fun isNew(timestampNanos: Long): Boolean {
        if (timestampNanos == lastTimestampNanos) return false
        lastTimestampNanos = timestampNanos
        return true
    }
}
