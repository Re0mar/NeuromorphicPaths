package com.neuromorphicpaths.core

/** Everything one pass of the pipeline produced for one frame, handed to every display. */
data class GuidanceUpdate(
    val frame: Frame,
    val detections: List<Detection>,
    val obstacles: List<Obstacle>,
    val guidance: Guidance,
    val walker: WalkerState,
)
