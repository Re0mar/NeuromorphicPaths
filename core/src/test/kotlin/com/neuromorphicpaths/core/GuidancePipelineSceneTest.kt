package com.neuromorphicpaths.core

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.asFlow
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Test

/** The segmenter runs on every nth frame and its scene is held on the frames between. */
class GuidancePipelineSceneTest {

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

    private object NothingDetector : ObstacleDetector {
        override val name = "nothing"
        override suspend fun detect(frame: Frame): List<Detection> = emptyList()
        override fun close() = Unit
    }

    private object NothingLocator : ObstacleLocator {
        override fun locate(detections: List<Detection>, frame: Frame): List<Obstacle> = emptyList()
    }

    /** Hands each call a fresh map and counts the calls. */
    private class CountingSegmenter : SurfaceSegmenter {
        override val name = "counting"
        var calls = 0
        override suspend fun segment(frame: Frame): SceneClassMap {
            calls += 1
            return SceneClassMap.build(2, 2) { _, _ -> SceneClass.GRASS }
        }
        override fun close() = Unit
    }

    /** One wall sample per map, tagged with the map it came from through the range. */
    private class OneWallLocator : SceneLocator {
        var calls = 0
        override fun locate(map: SceneClassMap, frame: Frame): LocatedScene {
            calls += 1
            val wall = Obstacle(
                detection = Detection(ObstacleClass.WALL, 1.0, NormalizedBox(0.4, 0.4, 0.6, 0.8)),
                bearingRadians = 0.0,
                rangeMeters = calls.toDouble(),
                closingSpeedMetersPerSecond = null,
            )
            return LocatedScene(surfaces = { _, _ -> SurfaceClass.GRASS }, structures = listOf(wall))
        }
    }

    /** Records the surface map it was handed, so the test can see the scene reached the field. */
    private class RecordingField : GuidanceField {
        val surfacesSeen = mutableListOf<GroundSurfaceMap>()
        override fun evaluate(obstacles: List<Obstacle>, walker: WalkerState, timestampNanos: Long, surfaces: GroundSurfaceMap): Guidance {
            surfacesSeen += surfaces
            return Guidance(0.0, 0.0, obstacles.map { ObstacleSurprise(it, 0.0, Push(0.0, 0.0)) })
        }
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
    fun theSceneIsRefreshedEveryNthFrameAndHeldBetween() = runBlocking {
        val segmenter = CountingSegmenter()
        val sceneLocator = OneWallLocator()
        val field = RecordingField()
        val display = RecordingDisplay()
        GuidancePipeline(
            source = ListFrameSource((1L..5L).map(::frameAt)),
            detector = NothingDetector,
            locator = NothingLocator,
            field = field,
            displays = listOf(display),
            dropStaleFrames = false,
            segmenter = segmenter,
            sceneLocator = sceneLocator,
            segmentEveryNthFrame = 2,
        ).run()

        // Frames 1, 3 and 5 segment. Frames 2 and 4 reuse the scene before them.
        assertEquals(3, segmenter.calls)
        assertEquals(3, sceneLocator.calls)
        assertEquals(listOf(1.0, 1.0, 2.0, 2.0, 3.0), display.updates.map { it.obstacles.single().rangeMeters })
        assertEquals(listOf(true, false, true, false, true), display.updates.map { it.segmenterNanos != null })
        assertNotNull(display.updates[1].sceneMap)
        assertSame(display.updates[0].sceneMap, display.updates[1].sceneMap)
        // Every frame's field call saw a real surface map, not the unknown default.
        assertEquals(5, field.surfacesSeen.size)
        assertEquals(SurfaceClass.GRASS, field.surfacesSeen.last().surfaceAt(1.0, 0.0))
    }

    @Test
    fun withoutASegmenterNothingChanges() = runBlocking {
        val field = RecordingField()
        val display = RecordingDisplay()
        GuidancePipeline(
            source = ListFrameSource(listOf(frameAt(1))),
            detector = NothingDetector,
            locator = NothingLocator,
            field = field,
            displays = listOf(display),
            dropStaleFrames = false,
        ).run()

        assertNull(display.updates.single().sceneMap)
        assertNull(display.updates.single().segmenterNanos)
        assertSame(GroundSurfaceMap.UNKNOWN_EVERYWHERE, field.surfacesSeen.single())
    }

    @Test
    fun aSegmenterWithoutALocatorIsRefused() {
        assertThrows(IllegalArgumentException::class.java) {
            GuidancePipeline(
                source = ListFrameSource(emptyList()),
                detector = NothingDetector,
                locator = NothingLocator,
                field = RecordingField(),
                displays = emptyList(),
                segmenter = CountingSegmenter(),
            )
        }
    }
}
