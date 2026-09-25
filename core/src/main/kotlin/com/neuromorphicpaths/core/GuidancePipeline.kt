package com.neuromorphicpaths.core

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.conflate

/**
 * Chains the four contracts: frames in, then detections, obstacles, guidance, displays.
 *
 * [run] suspends until the source ends or the caller cancels. It catches nothing. A failing
 * detector or display fails the run, and whoever launched it decides what that means.
 */
class GuidancePipeline(
    private val source: FrameSource,
    private val detector: ObstacleDetector,
    private val locator: ObstacleLocator,
    private val field: GuidanceField,
    private val displays: List<GuidanceDisplay>,
    private val walkerState: (Frame) -> WalkerState = { WalkerState.ALIGNED_WITH_CAMERA },
    private val dropStaleFrames: Boolean = true,
) {
    suspend fun run() {
        // A detector slower than the source should see the newest frame, not a growing backlog.
        // Tests turn this off so every frame is accounted for.
        val frames: Flow<Frame> = if (dropStaleFrames) source.frames().conflate() else source.frames()
        frames.collect { frame ->
            val detectStartNanos = System.nanoTime()
            val detections = detector.detect(frame)
            val detectorNanos = System.nanoTime() - detectStartNanos
            val obstacles = locator.locate(detections, frame)
            val walker = walkerState(frame)
            val guidance = field.evaluate(obstacles, walker, frame.timestampNanos)
            val update = GuidanceUpdate(frame, detections, obstacles, guidance, walker, detectorNanos)
            for (display in displays) {
                display.show(update)
            }
        }
    }
}
