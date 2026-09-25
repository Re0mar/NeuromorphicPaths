package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.CameraIntrinsics
import com.neuromorphicpaths.core.CameraPose
import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.NormalizedBox
import com.neuromorphicpaths.core.ObstacleClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.atan
import kotlin.math.tan

class GroundPlaneObstacleLocatorTest {

    private val locator = GroundPlaneObstacleLocator()

    // A 4:3 frame with a 90 degree horizontal view: tan(hfov/2) = 1 and tan(vfov/2) = 0.75.
    private fun frame(pitchRadians: Double = 0.0, yawRadians: Double = 0.0) = Frame(
        timestampNanos = 0L,
        width = 400,
        height = 300,
        rgba = ByteArray(400 * 300 * Frame.BYTES_PER_PIXEL),
        pose = CameraPose(pitchRadians, yawRadians, rollRadians = 0.0, heightMeters = 1.6),
        intrinsics = CameraIntrinsics(horizontalFovRadians = Math.toRadians(90.0)),
    )

    private fun detection(obstacleClass: ObstacleClass, left: Double, top: Double, right: Double, bottom: Double) =
        Detection(obstacleClass, confidence = 0.9, box = NormalizedBox(left, top, right, bottom))

    @Test
    fun rangeComesFromWhereTheBoxMeetsTheGround() {
        val obstacles = locator.locate(listOf(detection(ObstacleClass.CHAIR, 0.4, 0.5, 0.6, 0.75)), frame())

        // Bottom edge a quarter of the frame below center is 0.25 * 2 * 0.75 = 0.375 of the way
        // down the half-view, so range = 1.6 / 0.375.
        assertEquals(1, obstacles.size)
        assertEquals(1.6 / 0.375, obstacles.single().rangeMeters, 1e-9)
        assertEquals(0.0, obstacles.single().bearingRadians, 1e-9)
    }

    @Test
    fun bearingComesFromTheBoxCenterAndAddsCameraYaw() {
        val yaw = Math.toRadians(5.0)
        val obstacles = locator.locate(listOf(detection(ObstacleClass.POLE, 0.7, 0.5, 0.8, 0.75)), frame(yawRadians = yaw))

        // Center at 0.75 is halfway to the right edge, and tan(hfov/2) = 1, so atan(0.5).
        assertEquals(atan(0.5) + yaw, obstacles.single().bearingRadians, 1e-9)
    }

    @Test
    fun pitchingDownBringsTheGroundUpTheFrame() {
        val pitch = Math.toRadians(10.0)
        val obstacles = locator.locate(listOf(detection(ObstacleClass.CHAIR, 0.4, 0.3, 0.6, 0.5)), frame(pitchRadians = pitch))

        // Box bottom on the optical axis with the camera tilted 10 degrees down: the axis meets
        // the ground at height / tan(10 degrees).
        assertEquals(1.6 / tan(pitch), obstacles.single().rangeMeters, 1e-9)
    }

    @Test
    fun boxCutOffAtTheBottomFallsBackToTheClassHeight() {
        val obstacles = locator.locate(listOf(detection(ObstacleClass.PERSON, 0.4, 0.2, 0.6, 1.0)), frame())

        // A 1.7 m person filling 0.8 of the frame height: 1.7 / (0.8 * 2 * 0.75).
        assertEquals(1.7 / 1.2, obstacles.single().rangeMeters, 1e-9)
    }

    @Test
    fun objectsWithNoGroundContactAndNoHeightAreLeftOut() {
        val aboveHorizon = detection(ObstacleClass.UNKNOWN, 0.4, 0.1, 0.6, 0.3)
        val cutOffHole = detection(ObstacleClass.HOLE, 0.4, 0.8, 0.6, 1.0)

        val obstacles = locator.locate(listOf(aboveHorizon, cutOffHole), frame())

        assertTrue(obstacles.isEmpty())
    }
}
