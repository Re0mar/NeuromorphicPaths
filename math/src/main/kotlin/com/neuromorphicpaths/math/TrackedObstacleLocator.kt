package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.ObstacleLocator

/**
 * Wraps a locator so each tracked obstacle also carries a closing speed.
 *
 * The range of a track over the last [windowSeconds] is fitted with a straight line, and the
 * closing speed is minus its slope: positive when the object and the walker are getting nearer,
 * whether because the walker walks toward a standing thing or a person walks toward the walker.
 * Fewer than [minimumSamples] fresh ranges in the window give null, so the field falls back to
 * the walker's own speed, and a coasting box is never a sample, since its range is a repeat.
 */
class TrackedObstacleLocator(
    private val inner: ObstacleLocator,
    private val tracker: DetectionTracker,
    private val windowSeconds: Double = 2.0,
    private val minimumSamples: Int = 3,
) : ObstacleLocator {

    private class Sample(val seconds: Double, val rangeMeters: Double)

    private val history = mutableMapOf<Int, MutableList<Sample>>()

    override fun locate(detections: List<Detection>, frame: Frame): List<Obstacle> {
        val nowSeconds = frame.timestampNanos / NANOS_PER_SECOND
        val obstacles = inner.locate(detections, frame)
        val liveIds = mutableSetOf<Int>()
        val result = obstacles.map { obstacle ->
            val trackId = obstacle.detection.trackId ?: return@map obstacle
            liveIds += trackId
            val samples = history.getOrPut(trackId) { mutableListOf() }
            if (!tracker.isCoasting(trackId)) samples += Sample(nowSeconds, obstacle.rangeMeters)
            samples.removeAll { nowSeconds - it.seconds > windowSeconds }
            obstacle.copy(closingSpeedMetersPerSecond = closingSpeed(samples))
        }
        // A track the tracker dropped takes its history with it.
        history.keys.retainAll(liveIds)
        return result
    }

    private fun closingSpeed(samples: List<Sample>): Double? {
        if (samples.size < minimumSamples) return null
        val meanSeconds = samples.sumOf { it.seconds } / samples.size
        val meanRange = samples.sumOf { it.rangeMeters } / samples.size
        var covariance = 0.0
        var variance = 0.0
        for (sample in samples) {
            val dt = sample.seconds - meanSeconds
            covariance += dt * (sample.rangeMeters - meanRange)
            variance += dt * dt
        }
        if (variance <= 0.0) return null
        return -covariance / variance
    }

    private companion object {
        const val NANOS_PER_SECOND = 1e9
    }
}
