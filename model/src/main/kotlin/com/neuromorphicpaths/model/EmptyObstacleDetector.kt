package com.neuromorphicpaths.model

import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.ObstacleDetector

/** Finds nothing. Lets the pipeline run with a real source and real displays before a model is wired in. */
class EmptyObstacleDetector : ObstacleDetector {
    override val name: String = "none"

    override suspend fun detect(frame: Frame): List<Detection> = emptyList()

    override fun close() = Unit
}
