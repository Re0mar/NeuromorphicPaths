package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.CameraIntrinsics
import com.neuromorphicpaths.core.CameraPose
import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.FramePoint
import com.neuromorphicpaths.core.GroundPlaneProjection
import com.neuromorphicpaths.core.GroundPoint
import com.neuromorphicpaths.core.NormalizedBox
import com.neuromorphicpaths.core.ObstacleClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class GroundPlaneProjectionTest {

    private val intrinsics = CameraIntrinsics(horizontalFovRadians = Math.toRadians(66.0))
    private val width = 1080
    private val height = 1920

    private fun projection(pitchDegrees: Double = 18.0, yawDegrees: Double = 0.0, heightMeters: Double = 1.4) =
        GroundPlaneProjection(
            CameraPose(Math.toRadians(pitchDegrees), Math.toRadians(yawDegrees), 0.0, heightMeters),
            intrinsics,
            width,
            height,
        )

    @Test
    fun straightDownTheCenterColumnMatchesTheBoxLocator() {
        val pose = CameraPose(Math.toRadians(18.0), 0.0, 0.0, 1.4)
        val frame = Frame(0L, width, height, ByteArray(width * height * Frame.BYTES_PER_PIXEL), pose, intrinsics)
        // A box whose bottom edge is at 80 percent down the frame, centered.
        val box = NormalizedBox(0.45, 0.5, 0.55, 0.8)
        val fromLocator = GroundPlaneObstacleLocator().locate(listOf(Detection(ObstacleClass.POLE, 1.0, box)), frame).single()

        val ground = projection().groundAt(FramePoint(0.5, 0.8))

        assertNotNull(ground)
        assertEquals(fromLocator.rangeMeters, ground!!.forwardMeters, 1e-9)
        assertEquals(0.0, ground.rightMeters, 1e-9)
    }

    @Test
    fun groundAndFrameAreInverses() {
        val projection = projection(pitchDegrees = 20.0, yawDegrees = 7.0)
        for (x in listOf(0.1, 0.5, 0.9)) {
            for (y in listOf(0.6, 0.8, 0.95)) {
                val ground = projection.groundAt(FramePoint(x, y))
                assertNotNull("($x, $y) should be below the horizon", ground)
                val back = projection.frameAt(ground!!)
                assertNotNull(back)
                assertEquals(x, back!!.x, 1e-9)
                assertEquals(y, back.y, 1e-9)
            }
        }
    }

    @Test
    fun aPointOnTheRightOfTheFrameIsOnTheRightOfTheWalker() {
        val ground = projection().groundAt(FramePoint(0.9, 0.9))!!
        assert(ground.rightMeters > 0.0) { "expected a positive right offset, got ${ground.rightMeters}" }
        assert(ground.forwardMeters > 0.0)
    }

    @Test
    fun yawTurnsTheCameraFrameIntoTheWalkerFrame() {
        // The camera turned 30 degrees right, so the center column is 30 degrees right of the walker's line.
        val ground = projection(yawDegrees = 30.0).groundAt(FramePoint(0.5, 0.8))!!
        val bearing = Math.toDegrees(kotlin.math.atan2(ground.rightMeters, ground.forwardMeters))
        assertEquals(30.0, bearing, 1e-9)
    }

    @Test
    fun theHorizonAndAboveNeverReachTheGround() {
        val projection = projection(pitchDegrees = 18.0)
        // At 18 degrees of pitch and a 66 degree horizontal field of view, the horizon sits
        // above the frame center, so the very top of the frame looks at the sky.
        assertNull(projection.groundAt(FramePoint(0.5, 0.0)))
    }

    @Test
    fun groundBehindTheCameraHasNoPixel() {
        assertNull(projection().frameAt(GroundPoint(forwardMeters = -1.0, rightMeters = 0.0)))
    }

    @Test
    fun groundFarToTheSideFallsOutsideTheFrame() {
        assertNull(projection().frameAt(GroundPoint(forwardMeters = 2.0, rightMeters = 10.0)))
    }
}
