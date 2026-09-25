package com.neuromorphicpaths.input

import kotlin.math.exp

/**
 * One speed reading from a GPS fix, in meters per second, or null when there is nothing to go on.
 *
 * Prefers the receiver's own speed, which Android derives from the Doppler shift of the satellite
 * signals. Distance between two fixes over the time between them is the fallback, because phone
 * positions wobble by a few meters from fix to fix, more than a walker covers in one second.
 */
fun rawSpeedMetersPerSecond(
    dopplerMetersPerSecond: Double?,
    distanceMeters: Double?,
    elapsedSeconds: Double?,
): Double? {
    if (dopplerMetersPerSecond != null) return dopplerMetersPerSecond
    if (distanceMeters == null || elapsedSeconds == null || elapsedSeconds <= 0.0) return null
    return distanceMeters / elapsedSeconds
}

/**
 * Running average of speed readings with weight 1 - exp(-dt / window), so the result is the same
 * whether fixes arrive every second or every few. Plain Kotlin, so it is tested on the laptop.
 */
class SpeedSmoother(private val windowSeconds: Double = DEFAULT_WINDOW_SECONDS) {
    var metersPerSecond: Double? = null
        private set
    private var previousElapsedNanos: Long? = null

    fun add(readingMetersPerSecond: Double, elapsedNanos: Long): Double {
        val current = metersPerSecond
        val previousNanos = previousElapsedNanos
        if (current == null || previousNanos == null) {
            metersPerSecond = readingMetersPerSecond
            previousElapsedNanos = elapsedNanos
            return readingMetersPerSecond
        }
        val elapsedSeconds = (elapsedNanos - previousNanos) / NANOS_PER_SECOND
        // A repeated or out-of-order fix carries no new time, so it can't move the average.
        if (elapsedSeconds <= 0.0) return current

        val weight = 1.0 - exp(-elapsedSeconds / windowSeconds)
        val smoothed = current + weight * (readingMetersPerSecond - current)
        metersPerSecond = smoothed
        previousElapsedNanos = elapsedNanos
        return smoothed
    }

    fun reset() {
        metersPerSecond = null
        previousElapsedNanos = null
    }

    companion object {
        /** The same one-second window the heading wobble uses, so the two numbers describe the same moment. */
        const val DEFAULT_WINDOW_SECONDS = 1.0
        private const val NANOS_PER_SECOND = 1e9
    }
}
