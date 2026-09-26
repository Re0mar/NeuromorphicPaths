package com.neuromorphicpaths.input

import kotlin.math.exp

/**
 * The walker's speed from the rhythm of their steps, read off the accelerometer.
 *
 * Each step lands as a bump in the acceleration magnitude, so steps are counted as peaks above
 * a threshold with a minimum spacing, the cadence is one over the mean of the last few step
 * intervals, and the speed is cadence times a step length. That estimate is smoothed with the
 * same one-second window the other walker numbers use. When no step has landed for a while the
 * walker has stopped, and the speed decays to zero, which is what makes standing objects fade
 * out of the field while a person walking toward the walker does not.
 *
 * GPS speed lags by seconds and is useless indoors, which is why the steps are the source and
 * GPS is only a cross-check outdoors. The step length is a constant here. A taller or shorter
 * walker, or one hurrying, walks a different length per step, and the GPS cross-check is what
 * would calibrate it.
 *
 * The input is the acceleration magnitude minus gravity, in meters per second squared, which
 * needs no knowledge of how the phone is held. Plain Kotlin, tested on the laptop.
 */
class StepCadenceSpeedEstimator(
    private val stepLengthMeters: Double = DEFAULT_STEP_LENGTH_METERS,
    private val peakThresholdMetersPerSecondSquared: Double = DEFAULT_PEAK_THRESHOLD,
    private val minimumStepIntervalSeconds: Double = DEFAULT_MINIMUM_STEP_INTERVAL_SECONDS,
    private val stopAfterSeconds: Double = DEFAULT_STOP_AFTER_SECONDS,
    private val smoothingWindowSeconds: Double = DEFAULT_SMOOTHING_WINDOW_SECONDS,
    private val filterSeconds: Double = DEFAULT_FILTER_SECONDS,
) {
    init {
        require(stepLengthMeters > 0.0 && peakThresholdMetersPerSecondSquared > 0.0) { "step length and threshold must be positive" }
        require(minimumStepIntervalSeconds > 0.0 && stopAfterSeconds > minimumStepIntervalSeconds) { "stop time must exceed the step interval" }
        require(smoothingWindowSeconds > 0.0 && filterSeconds > 0.0) { "windows must be positive" }
    }

    /** Smoothed speed in meters per second. Null until the estimator has seen enough to say anything, zero once the walker has stood still. */
    var metersPerSecond: Double? = null
        private set

    private var filtered = 0.0
    private var previousNanos: Long? = null
    private var firstNanos: Long? = null
    private var armed = false
    private var peakValue = 0.0
    private var peakNanos = 0L
    private var lastStepNanos: Long? = null
    private val recentIntervalsSeconds = ArrayDeque<Double>()

    /** Feeds one accelerometer sample and returns the current speed estimate. */
    fun add(timestampNanos: Long, accelerationMinusGravity: Double): Double? {
        val previous = previousNanos
        if (previous == null) {
            previousNanos = timestampNanos
            firstNanos = timestampNanos
            filtered = accelerationMinusGravity
            return metersPerSecond
        }
        val elapsedSeconds = (timestampNanos - previous) / NANOS_PER_SECOND
        if (elapsedSeconds <= 0.0) return metersPerSecond
        previousNanos = timestampNanos

        // A light low-pass takes the sensor's jitter out before the peaks are counted.
        filtered += (1.0 - exp(-elapsedSeconds / filterSeconds)) * (accelerationMinusGravity - filtered)
        detectStep(timestampNanos)

        // No step for a while means the walker is standing, and the estimate drifts to zero.
        val sinceLastStep = ((lastStepNanos ?: firstNanos ?: timestampNanos).let { timestampNanos - it }) / NANOS_PER_SECOND
        if (sinceLastStep > stopAfterSeconds) {
            recentIntervalsSeconds.clear()
            smoothToward(0.0, elapsedSeconds)
        }
        return metersPerSecond
    }

    fun reset() {
        metersPerSecond = null
        filtered = 0.0
        previousNanos = null
        firstNanos = null
        armed = false
        lastStepNanos = null
        recentIntervalsSeconds.clear()
    }

    private fun detectStep(timestampNanos: Long) {
        if (!armed) {
            if (filtered > peakThresholdMetersPerSecondSquared) {
                armed = true
                peakValue = filtered
                peakNanos = timestampNanos
            }
            return
        }
        if (filtered > peakValue) {
            peakValue = filtered
            peakNanos = timestampNanos
            return
        }
        // The bump is over once the signal falls back through zero. The step is timed at its peak.
        if (filtered < 0.0) {
            armed = false
            val lastStep = lastStepNanos
            val intervalSeconds = lastStep?.let { (peakNanos - it) / NANOS_PER_SECOND }
            if (intervalSeconds != null && intervalSeconds < minimumStepIntervalSeconds) return
            lastStepNanos = peakNanos
            if (intervalSeconds != null && intervalSeconds <= stopAfterSeconds) {
                recentIntervalsSeconds.addLast(intervalSeconds)
                while (recentIntervalsSeconds.size > INTERVALS_KEPT) recentIntervalsSeconds.removeFirst()
                val cadenceStepsPerSecond = 1.0 / recentIntervalsSeconds.average()
                smoothToward(cadenceStepsPerSecond * stepLengthMeters, intervalSeconds)
            }
        }
    }

    private fun smoothToward(readingMetersPerSecond: Double, elapsedSeconds: Double) {
        val current = metersPerSecond
        if (current == null) {
            metersPerSecond = readingMetersPerSecond
            return
        }
        val weight = 1.0 - exp(-elapsedSeconds / smoothingWindowSeconds)
        metersPerSecond = current + weight * (readingMetersPerSecond - current)
    }

    companion object {
        /** An adult's ordinary step, heel to heel, which at 1.8 steps a second gives about 1.3 m/s. */
        const val DEFAULT_STEP_LENGTH_METERS = 0.7

        /** Standing still on the outdoor recording peaks at about 0.5 m/s squared, walking at 3 to 4. */
        const val DEFAULT_PEAK_THRESHOLD = 1.0
        const val DEFAULT_MINIMUM_STEP_INTERVAL_SECONDS = 0.3
        const val DEFAULT_STOP_AFTER_SECONDS = 1.5
        const val DEFAULT_SMOOTHING_WINDOW_SECONDS = 1.0
        const val DEFAULT_FILTER_SECONDS = 0.05
        private const val INTERVALS_KEPT = 3
        private const val NANOS_PER_SECOND = 1e9
    }
}
