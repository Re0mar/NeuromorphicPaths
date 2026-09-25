package com.neuromorphicpaths.math

import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Running spread of the walker's heading over the ground, from a stream of azimuth samples.
 *
 * Exponentially weighted mean and variance with weight 1 - exp(-dt / window), so the estimate
 * comes out the same whether samples arrive every frame or every few. Each sample is unwrapped
 * against the running mean before it is used, so a heading that crosses the compass wrap reads
 * as a small step and not as a full turn.
 *
 * The number this produces is the walker's own sway, which is what a turn tolerance would be
 * measured against. On the first outdoor recording, a 1 s window gives a median of 4 degrees,
 * rising past 20 degrees only through corners.
 */
class HeadingWobbleEstimator(
    private val windowSeconds: Double = DEFAULT_WINDOW_SECONDS,
) {
    private var meanRadians: Double? = null
    private var variance = 0.0
    private var previousTimestampNanos: Long? = null
    private var sampleCount = 0

    init {
        require(windowSeconds > 0.0) { "windowSeconds must be positive" }
    }

    /** Standard deviation of the heading in radians. Null until two samples with time between them. */
    val wobbleRadians: Double?
        get() = if (sampleCount >= 2) sqrt(variance) else null

    /** The running mean heading, wrapped to -pi..pi. Null before the first sample. */
    val meanHeadingRadians: Double?
        get() = meanRadians

    fun add(timestampNanos: Long, azimuthRadians: Double) {
        val mean = meanRadians
        val previous = previousTimestampNanos
        if (mean == null || previous == null) {
            meanRadians = wrapToPi(azimuthRadians)
            variance = 0.0
            previousTimestampNanos = timestampNanos
            sampleCount = 1
            return
        }
        val elapsedSeconds = (timestampNanos - previous) / NANOS_PER_SECOND
        // A repeated or out-of-order sample carries no new time, so it cannot move the estimate.
        if (elapsedSeconds <= 0.0) return

        val weight = 1.0 - exp(-elapsedSeconds / windowSeconds)
        val error = wrapToPi(azimuthRadians - mean)
        meanRadians = wrapToPi(mean + weight * error)
        variance = (1.0 - weight) * (variance + weight * error * error)
        previousTimestampNanos = timestampNanos
        sampleCount += 1
    }

    fun reset() {
        meanRadians = null
        variance = 0.0
        previousTimestampNanos = null
        sampleCount = 0
    }

    companion object {
        /** Vertegaal's spread was measured over 1 s windows, and one walking stride is about that long. */
        const val DEFAULT_WINDOW_SECONDS = 1.0
        private const val NANOS_PER_SECOND = 1e9
        private const val TWO_PI = 2.0 * PI

        /** Brings any angle into -pi..pi. */
        fun wrapToPi(radians: Double): Double {
            var wrapped = radians % TWO_PI
            if (wrapped > PI) wrapped -= TWO_PI
            if (wrapped < -PI) wrapped += TWO_PI
            return wrapped
        }
    }
}
