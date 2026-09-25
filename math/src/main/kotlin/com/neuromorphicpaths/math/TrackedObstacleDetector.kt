package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.ObstacleDetector

/**
 * Wraps any detector so its detections carry track ids and briefly outlive a missed frame.
 *
 * The detector underneath knows nothing about it, so the same tracking serves the scripted
 * detector in tests and the real one on the phone. Share the [tracker] with a
 * [TrackedObstacleLocator] so the closing speed knows which boxes are coasting.
 */
class TrackedObstacleDetector(
    private val inner: ObstacleDetector,
    private val tracker: DetectionTracker,
) : ObstacleDetector {

    override val name: String = "${inner.name}+tracks"

    override suspend fun detect(frame: Frame): List<Detection> = tracker.update(inner.detect(frame))

    override fun close() = inner.close()
}
