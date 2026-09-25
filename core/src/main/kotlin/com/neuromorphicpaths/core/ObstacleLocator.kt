package com.neuromorphicpaths.core

/**
 * First math contract. Detections in a frame become obstacles around the walker.
 *
 * Pure. Reads the frame's pose and intrinsics, never its pixels. A detection the locator cannot
 * place is left out rather than given a made-up range.
 */
interface ObstacleLocator {
    fun locate(detections: List<Detection>, frame: Frame): List<Obstacle>
}
