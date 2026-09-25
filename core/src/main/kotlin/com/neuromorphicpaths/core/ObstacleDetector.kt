package com.neuromorphicpaths.core

/**
 * The model contract. One frame in, the objects found in it out.
 *
 * Suspending, because a real detector runs on its own thread and the pipeline waits for it.
 * A detector maps its model's labels onto [ObstacleClass] before returning.
 */
interface ObstacleDetector : AutoCloseable {
    val name: String

    suspend fun detect(frame: Frame): List<Detection>
}
