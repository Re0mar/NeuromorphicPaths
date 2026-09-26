package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.CameraIntrinsics
import com.neuromorphicpaths.core.CameraPose
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.ObstacleClass
import com.neuromorphicpaths.core.SceneClass
import com.neuromorphicpaths.core.SceneClassMap
import com.neuromorphicpaths.core.SurfaceClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.hypot

class GroundPlaneSceneLocatorTest {

    private val width = 1080
    private val height = 1920
    private val intrinsics = CameraIntrinsics(horizontalFovRadians = Math.toRadians(66.0))
    private val pose = CameraPose(pitchRadians = Math.toRadians(18.0), yawRadians = 0.0, rollRadians = 0.0, heightMeters = 1.4)
    private val frame = Frame(0L, width, height, ByteArray(width * height * Frame.BYTES_PER_PIXEL), pose, intrinsics)
    private val projection = GroundPlaneProjection(pose, intrinsics, width, height)
    private val locator = GroundPlaneSceneLocator()

    private val gridSize = 64

    /** The row the horizon falls in, so tests can put ground below it and sky above. */
    private val horizonRow: Int = (0 until gridSize).first { row -> projection.groundAt(FramePoint(0.5, (row + 0.5) / gridSize)) != null }

    /** Sky above the horizon, pavement below, and grass on the left third of the ground. */
    private fun pavementWithGrassOnTheLeft() = SceneClassMap.build(gridSize, gridSize) { column, row ->
        when {
            row < horizonRow -> SceneClass.SKY
            column < gridSize / 3 -> SceneClass.GRASS
            else -> SceneClass.PAVEMENT
        }
    }

    /** Pavement, with a wall filling the right quarter of the frame from the horizon down. */
    private fun pavementWithAWallOnTheRight() = SceneClassMap.build(gridSize, gridSize) { column, row ->
        when {
            row < horizonRow -> SceneClass.SKY
            column >= gridSize * 3 / 4 -> SceneClass.WALL
            else -> SceneClass.PAVEMENT
        }
    }

    @Test
    fun theGroundIsAnsweredByTheCellItProjectsInto() {
        val scene = locator.locate(pavementWithGrassOnTheLeft(), frame)

        assertEquals(SurfaceClass.PAVEMENT, scene.surfaces.surfaceAt(3.0, 0.0))
        assertEquals(SurfaceClass.GRASS, scene.surfaces.surfaceAt(3.0, -1.5))
        assertEquals(SurfaceClass.PAVEMENT, scene.surfaces.surfaceAt(3.0, 1.0))
        assertTrue(scene.structures.isEmpty())
    }

    @Test
    fun groundTheFrameDoesNotCoverIsUnknown() {
        val scene = locator.locate(pavementWithGrassOnTheLeft(), frame)

        // Behind the camera, and far enough to the side to fall out of a 66 degree view.
        assertEquals(SurfaceClass.UNKNOWN, scene.surfaces.surfaceAt(-1.0, 0.0))
        assertEquals(SurfaceClass.UNKNOWN, scene.surfaces.surfaceAt(2.0, 8.0))
    }

    @Test
    fun aWallBecomesSamplesAlongItsFootOnTheRight() {
        val scene = locator.locate(pavementWithAWallOnTheRight(), frame)

        assertTrue("expected wall samples, got none", scene.structures.isNotEmpty())
        assertTrue(scene.structures.all { it.obstacleClass == ObstacleClass.WALL })
        assertTrue("every sample should be to the right", scene.structures.all { it.bearingRadians > 0.0 })
        assertTrue(scene.structures.all { it.detection.confidence == 1.0 && it.detection.trackId == null && it.closingSpeedMetersPerSecond == null })
        // The ground under the wall is not a surface the field charges for on top of the obstacle.
        val underTheWall = scene.structures.minByOrNull { it.rangeMeters }!!
        val forward = underTheWall.rangeMeters * kotlin.math.cos(underTheWall.bearingRadians)
        val right = underTheWall.rangeMeters * kotlin.math.sin(underTheWall.bearingRadians)
        assertEquals(SurfaceClass.UNKNOWN, scene.surfaces.surfaceAt(forward, right + 0.3))
    }

    @Test
    fun samplesOfOneStructureAreAtLeastAClearanceApart() {
        val spacing = 0.5
        val scene = GroundPlaneSceneLocator(sampleSpacingMeters = spacing).locate(pavementWithAWallOnTheRight(), frame)

        val points = scene.structures.map { it.rangeMeters * kotlin.math.cos(it.bearingRadians) to it.rangeMeters * kotlin.math.sin(it.bearingRadians) }
        for (first in points.indices) {
            for (second in first + 1 until points.size) {
                val distance = hypot(points[first].first - points[second].first, points[first].second - points[second].second)
                assertTrue("samples $first and $second are $distance m apart", distance >= spacing - 1e-9)
            }
        }
        // A wall spanning a quarter of the frame from the horizon down is many cells but a handful of samples.
        assertTrue("expected fewer than 60 samples, got ${scene.structures.size}", scene.structures.size < 60)
    }

    @Test
    fun aStructureWithNoGroundBesideItHasNoFoot() {
        // A wall that fills the frame below the horizon touches no ground cell, so nothing is placed.
        val allWall = SceneClassMap.build(gridSize, gridSize) { _, row -> if (row < horizonRow) SceneClass.SKY else SceneClass.WALL }
        val scene = locator.locate(allWall, frame)

        assertTrue(scene.structures.isEmpty())
    }

    @Test
    fun feetBeyondTheRangeCapAreDropped() {
        val scene = GroundPlaneSceneLocator(maxRangeMeters = 3.0).locate(pavementWithAWallOnTheRight(), frame)

        assertTrue(scene.structures.all { it.rangeMeters <= 3.0 })
    }
}
