package com.neuromorphicpaths.core

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.conflate

/**
 * Chains the contracts: frames in, then detections, obstacles, guidance, displays.
 *
 * With a segmenter, every [segmentEveryNthFrame]th frame is also segmented and located, and the
 * scene that comes out, the ground surfaces and the structure samples, is held and used on every
 * frame until the next map replaces it. Walls and verges move slowly in the frame, so a scene a
 * frame or two old is close enough and the segmenter's cost is paid a third as often.
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
    private val segmenter: SurfaceSegmenter? = null,
    private val sceneLocator: SceneLocator? = null,
    private val segmentEveryNthFrame: Int = DEFAULT_SEGMENT_EVERY_NTH_FRAME,
) {
    init {
        require((segmenter == null) == (sceneLocator == null)) { "A segmenter and a scene locator come together or not at all" }
        require(segmentEveryNthFrame >= 1) { "segmentEveryNthFrame must be at least 1, got $segmentEveryNthFrame" }
    }

    suspend fun run() {
        // A detector slower than the source should see the newest frame, not a growing backlog.
        // Tests turn this off so every frame is accounted for.
        val frames: Flow<Frame> = if (dropStaleFrames) source.frames().conflate() else source.frames()
        var scene = LocatedScene.NOTHING
        var sceneMap: SceneClassMap? = null
        var frameIndex = 0
        frames.collect { frame ->
            val detectStartNanos = System.nanoTime()
            val detections = detector.detect(frame)
            val detectorNanos = System.nanoTime() - detectStartNanos
            var segmenterNanos: Long? = null
            if (segmenter != null && sceneLocator != null && frameIndex % segmentEveryNthFrame == 0) {
                val segmentStartNanos = System.nanoTime()
                val map = segmenter.segment(frame)
                segmenterNanos = System.nanoTime() - segmentStartNanos
                sceneMap = map
                scene = sceneLocator.locate(map, frame)
            }
            frameIndex += 1
            val obstacles = locator.locate(detections, frame) + scene.structures
            val walker = walkerState(frame)
            val guidance = field.evaluate(obstacles, walker, frame.timestampNanos, scene.surfaces)
            val update = GuidanceUpdate(frame, detections, obstacles, guidance, walker, detectorNanos, sceneMap, segmenterNanos)
            for (display in displays) {
                display.show(update)
            }
        }
    }

    companion object {
        /** Every third frame: at the detector's two frames a second that is a map every second and a half. */
        const val DEFAULT_SEGMENT_EVERY_NTH_FRAME = 3
    }
}
