package com.example.sidewalkvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

// A simulated camera looking along a straight path, the same one the analysis/ tests use.
private const val FRAME_WIDTH = 1920
private const val FRAME_HEIGHT = 1080
private const val FOCAL_LENGTH_PX = 1450.0
private const val CELL_FRACTION = 1.0 / 80
private val INTRINSICS = CameraIntrinsics(FOCAL_LENGTH_PX / FRAME_WIDTH)

private data class Camera(
    val heightMeters: Double,
    val pitchDegrees: Double,
    val headingDegrees: Double,
    val positionAcross: Double,
    val pathWidthMeters: Double = DEFAULT_PATH_WIDTH_METERS,
)

/** Level frame (x right, y down, z forward, camera at the origin) to frame pixels. */
private fun project(x: Double, y: Double, z: Double, pitch: Double): Pair<Double, Double> {
    val cameraY = y * cos(pitch) - z * sin(pitch)
    val cameraZ = y * sin(pitch) + z * cos(pitch)
    return (FRAME_WIDTH / 2.0 + FOCAL_LENGTH_PX * x / cameraZ) to (FRAME_HEIGHT / 2.0 + FOCAL_LENGTH_PX * cameraY / cameraZ)
}

/** The image line x = slope * y + offset of a path edge at a lateral offset from the camera. */
private fun edgeLine(camera: Camera, lateral: Double): Pair<Double, Double> {
    val pitch = Math.toRadians(camera.pitchDegrees)
    val heading = Math.toRadians(camera.headingDegrees)
    fun ground(distance: Double) = project(
        lateral * cos(heading) + distance * sin(heading),
        camera.heightMeters,
        -lateral * sin(heading) + distance * cos(heading),
        pitch,
    )
    val (nearX, nearY) = ground(3.0)
    val (farX, farY) = ground(30.0)
    val slope = (farX - nearX) / (farY - nearY)
    return slope to (nearX - slope * nearY)
}

/** Edge points a perfect detector would report for the camera, two per row over the lower half. */
private fun edgePoints(camera: Camera): List<SidewalkEdgePoint> {
    val (leftSlope, leftOffset) = edgeLine(camera, -camera.positionAcross * camera.pathWidthMeters)
    val (rightSlope, rightOffset) = edgeLine(camera, (1 - camera.positionAcross) * camera.pathWidthMeters)
    return (FRAME_HEIGHT - 12 downTo FRAME_HEIGHT / 2 + 1 step 24).flatMap { rowPx ->
        val y = (rowPx.toDouble() / FRAME_HEIGHT).toFloat()
        listOf(
            SidewalkEdgePoint(Point2D(((leftSlope * rowPx + leftOffset) / FRAME_WIDTH).toFloat(), y), SidewalkEdgeSide.LEFT, null),
            SidewalkEdgePoint(Point2D(((rightSlope * rowPx + rightOffset) / FRAME_WIDTH).toFloat(), y), SidewalkEdgeSide.RIGHT, null),
        )
    }
}

private fun estimate(points: List<SidewalkEdgePoint>, intrinsics: CameraIntrinsics? = INTRINSICS, maskBounds: FloatArray? = null) =
    estimatePose(points, FRAME_WIDTH, FRAME_HEIGHT, CELL_FRACTION, maskBounds, intrinsics)

class PoseEstimateTest {
    @Test
    fun recoversKnownCameras() {
        for (camera in listOf(Camera(0.4, 8.0, 0.0, 0.5), Camera(0.3, 4.0, 3.0, 0.3), Camera(1.6, 15.0, -5.0, 0.7))) {
            val pose = estimate(edgePoints(camera))

            assertEquals(PoseStatus.OK, pose.status)
            assertTrue("$camera should be reliable", pose.reliable)
            assertEquals(camera.pitchDegrees, pose.pitchDegrees!!, 1e-3)
            assertEquals(camera.headingDegrees, pose.headingDegrees!!, 1e-3)
            assertEquals(camera.positionAcross, pose.positionAcross!!, 1e-4)
            assertEquals(camera.heightMeters, pose.cameraHeightMeters!!, camera.heightMeters * 1e-4)
        }
    }

    @Test
    fun withoutAFocalLengthOnlyPositionIsReported() {
        val pose = estimate(edgePoints(Camera(1.6, 12.0, 0.0, 0.4)), intrinsics = null)

        assertEquals(PoseStatus.OK, pose.status)
        assertNull(pose.pitchDegrees)
        assertNull(pose.cameraHeightMeters)
        assertEquals(0.4, pose.positionAcross!!, 1e-4)
    }

    @Test
    fun clippedPointsAreNotFitted() {
        val clipped = edgePoints(Camera(0.4, 8.0, 0.0, 0.5)).map { point ->
            if (point.side == SidewalkEdgeSide.LEFT) point.copy(clipped = true) else point
        }

        assertEquals(PoseStatus.TOO_FEW_POINTS, estimate(clipped).status)
    }

    @Test
    fun anEdgeAlongTheMaskRegionsSideIsTheBoxNotThePath() {
        // The left edge runs straight down at 30% of the width, exactly where the mask region ends.
        val boxed = edgePoints(Camera(0.4, 8.0, 0.0, 0.5)).map { point ->
            if (point.side == SidewalkEdgeSide.LEFT) point.copy(position = point.position.copy(x = 0.3f)) else point
        }

        val pose = estimate(boxed, maskBounds = floatArrayOf(0.3f, 0f, 1f, 1f))

        assertTrue(pose.left!!.onBoxSide)
        assertFalse(pose.reliable)
        assertEquals("an edge is the detection box's side", pose.unreliableReason)
    }

    @Test
    fun groundDistanceMatchesTheSimulatedGround() {
        val camera = Camera(0.4, 8.0, 0.0, 0.5)
        val pose = estimate(edgePoints(camera))
        // The row where ground 5 m ahead of the camera appears.
        val (_, rowPx) = project(0.0, camera.heightMeters, 5.0, Math.toRadians(camera.pitchDegrees))

        val distance = groundDistanceMeters(pose, rowPx, FRAME_HEIGHT)

        assertNotNull(distance)
        assertEquals(5.0, distance!!, 5.0 * 1e-3)
    }

    @Test
    fun aThreeByTwoPhotoMatchesTheFilmWidthConvention() {
        // On a 3:2 image the diagonal and the 36 mm width agree: a 36 mm equivalent is one image width.
        val intrinsics = intrinsicsFrom35mmEquivalent(36.0, 6000, 4000)!!
        assertEquals(6000.0, intrinsics.focalLengthPx(6000, 4000), 6000 * 1e-3)
    }

    @Test
    fun aFourByThreePhotoIsMeasuredAgainstTheDiagonal() {
        // The iPhone 12 ultra-wide: 14 mm equivalent on 4032 x 3024, whose diagonal is 5040 px.
        val intrinsics = intrinsicsFrom35mmEquivalent(14.0, 4032, 3024)!!
        assertEquals(14.0 * 5040 / 43.27, intrinsics.focalLengthPx(4032, 3024), 1e-6)
        // The film-width shortcut would give 14 / 36 of the width, about 4% less.
        assertTrue(intrinsics.focalLengthPx(4032, 3024) > 1.03 * 14.0 / 36 * 4032)
    }

    @Test
    fun noIntrinsicsWithoutAFocalLength() {
        assertNull(intrinsicsFrom35mmEquivalent(0.0, 4032, 3024))
    }

    @Test
    fun noGroundDistanceAboveTheHorizon() {
        val pose = estimate(edgePoints(Camera(0.4, 8.0, 0.0, 0.5)))
        assertNull(groundDistanceMeters(pose, 0.0, FRAME_HEIGHT))
    }
}
