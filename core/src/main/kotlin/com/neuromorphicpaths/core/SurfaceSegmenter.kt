package com.neuromorphicpaths.core

/**
 * The second model contract. One frame in, a coarse map of what every part of it is out.
 *
 * Where [ObstacleDetector] finds things with an outline, this finds regions: the pavement the
 * walker is on, the grass beside it, the wall along it. It runs on a fraction of the frames and
 * at a fraction of the detector's resolution, since regions move slowly in the frame, and the
 * pipeline holds its last answer between calls. A segmenter maps its model's labels onto
 * [SceneClass] before returning.
 */
interface SurfaceSegmenter : AutoCloseable {
    val name: String

    suspend fun segment(frame: Frame): SceneClassMap
}
