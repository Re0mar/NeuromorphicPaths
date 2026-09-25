package com.neuromorphicpaths.model

import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.ObstacleDetector

/**
 * Returns the same detections for every frame.
 *
 * For exercising the locator, the field and the displays with known input, on a phone or in a
 * test, without a model in the loop.
 */
class ScriptedObstacleDetector(
    private val detections: List<Detection>,
) : ObstacleDetector {
    override val name: String = "scripted"

    override suspend fun detect(frame: Frame): List<Detection> = detections

    override fun close() = Unit
}
