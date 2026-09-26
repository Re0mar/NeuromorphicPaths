package com.neuromorphicpaths.core

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.asFlow
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Test

class GuidancePipelineTest {

    private val pose = CameraPose(pitchRadians = 0.0, yawRadians = 0.0, rollRadians = 0.0, heightMeters = 1.5)
    private val intrinsics = CameraIntrinsics(horizontalFovRadians = Math.toRadians(60.0))

    private fun frameAt(timestampNanos: Long) = Frame(
        timestampNanos = timestampNanos,
        width = 2,
        height = 2,
        rgba = ByteArray(2 * 2 * Frame.BYTES_PER_PIXEL),
        pose = pose,
        intrinsics = intrinsics,
    )

    private class ListFrameSource(private val frames: List<Frame>) : FrameSource {
        override val name = "list"
        override fun frames(): Flow<Frame> = frames.asFlow()
        override fun close() = Unit
    }

    private class OneDetectionDetector : ObstacleDetector {
        override val name = "one"
        var calls = 0
        override suspend fun detect(frame: Frame): List<Detection> {
            calls += 1
            return listOf(Detection(ObstacleClass.CHAIR, 0.9, NormalizedBox(0.4, 0.4, 0.6, 0.8)))
        }
        override fun close() = Unit
    }

    private object PassThroughLocator : ObstacleLocator {
        override fun locate(detections: List<Detection>, frame: Frame): List<Obstacle> =
            detections.map { Obstacle(it, bearingRadians = 0.0, rangeMeters = 3.0, closingSpeedMetersPerSecond = null) }
    }

    private object StraightAheadField : GuidanceField {
        override fun evaluate(obstacles: List<Obstacle>, walker: WalkerState, timestampNanos: Long, surfaces: GroundSurfaceMap): Guidance =
            Guidance(0.0, 0.0, obstacles.map { ObstacleSurprise(it, 0.0, Push(0.0, 0.0)) })
    }

    private class RecordingDisplay : GuidanceDisplay {
        override val name = "recording"
        val updates = mutableListOf<GuidanceUpdate>()
        override suspend fun show(update: GuidanceUpdate) {
            updates += update
        }
        override fun close() = Unit
    }

    @Test
    fun everyFrameReachesEveryDisplayWhenNothingIsDropped() = runBlocking {
        val detector = OneDetectionDetector()
        val first = RecordingDisplay()
        val second = RecordingDisplay()
        val pipeline = GuidancePipeline(
            source = ListFrameSource(listOf(frameAt(1), frameAt(2), frameAt(3))),
            detector = detector,
            locator = PassThroughLocator,
            field = StraightAheadField,
            displays = listOf(first, second),
            dropStaleFrames = false,
        )

        pipeline.run()

        assertEquals(3, detector.calls)
        assertEquals(listOf(1L, 2L, 3L), first.updates.map { it.frame.timestampNanos })
        assertEquals(listOf(1L, 2L, 3L), second.updates.map { it.frame.timestampNanos })
        assertEquals(1, first.updates.last().obstacles.size)
        assertEquals(1, first.updates.last().guidance.perObstacle.size)
    }

    @Test
    fun droppingStaleFramesStillDeliversTheNewest() = runBlocking {
        val display = RecordingDisplay()
        val pipeline = GuidancePipeline(
            source = ListFrameSource(listOf(frameAt(1), frameAt(2), frameAt(3))),
            detector = OneDetectionDetector(),
            locator = PassThroughLocator,
            field = StraightAheadField,
            displays = listOf(display),
        )

        pipeline.run()

        assertEquals(3L, display.updates.last().frame.timestampNanos)
    }
}
