package com.neuromorphicpaths.pixel.ar

import com.google.ar.core.Pose
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The two conversions that would fail silently, against values worked out by hand.
 *
 * A rotation or a plane that is wrong by a consistent transform agrees with itself everywhere, so
 * nothing here is checked against another part of the system. The expected values are literals.
 */
class ArCoreToWireTest {
    private val tolerance = 1e-6

    private fun assertClose(expected: DoubleArray, actual: DoubleArray, message: String = "") {
        assertEquals(expected.size, actual.size, "$message size")
        for (index in expected.indices) {
            assertTrue(abs(expected[index] - actual[index]) < tolerance, "$message index $index: expected ${expected[index]}, got ${actual[index]}")
        }
    }

    /** ARCore poses take rotation as (x, y, z, w). */
    private fun pose(tx: Float, ty: Float, tz: Float, qx: Float, qy: Float, qz: Float, qw: Float): Pose =
        Pose(floatArrayOf(tx, ty, tz), floatArrayOf(qx, qy, qz, qw))

    /** Rotate a vector by a (w, x, y, z) quaternion, the laptop's convention, independently of the code under test. */
    private fun rotate(q: DoubleArray, v: DoubleArray): DoubleArray {
        val (w, x, y, z) = q
        val m = arrayOf(
            doubleArrayOf(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            doubleArrayOf(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            doubleArrayOf(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
        return DoubleArray(3) { row -> m[row][0] * v[0] + m[row][1] * v[1] + m[row][2] * v[2] }
    }

    @Test
    fun anIdentityArCorePoseBecomesTheHalfTurnAboutX() {
        val q = ArCoreToWire.cameraToWorldWxyz(pose(0f, 0f, 0f, 0f, 0f, 0f, 1f))

        assertClose(doubleArrayOf(0.0, 1.0, 0.0, 0.0), q)
        // The laptop's camera down, +y, is the world's down, -y, in a y-up world.
        assertClose(doubleArrayOf(0.0, -1.0, 0.0), rotate(q, doubleArrayOf(0.0, 1.0, 0.0)), "down")
        // And the laptop's forward, +z, is ARCore's forward, -z.
        assertClose(doubleArrayOf(0.0, 0.0, -1.0), rotate(q, doubleArrayOf(0.0, 0.0, 1.0)), "forward")
    }

    @Test
    fun aQuarterYawTurnsTheLaptopsForwardToWorldMinusX() {
        // ARCore camera yawed 90 degrees about world y: ARCore's forward (0, 0, -1) goes to (-1, 0, 0).
        val half = (Math.PI / 4).toFloat()
        val q = ArCoreToWire.cameraToWorldWxyz(pose(0f, 0f, 0f, 0f, sin(half), 0f, cos(half)))

        assertClose(doubleArrayOf(-1.0, 0.0, 0.0), rotate(q, doubleArrayOf(0.0, 0.0, 1.0)), "forward after yaw")
        assertTrue(abs(sqrt(q.sumOf { it * it }) - 1.0) < tolerance, "unit quaternion")
    }

    @Test
    fun theOutputQuaternionIsAlwaysUnitLength() {
        // ARCore's quaternion arrives as floats, so it is not exactly unit. Ours must be.
        val q = ArCoreToWire.cameraToWorldWxyz(pose(1f, 2f, 3f, 0.1f, 0.2f, 0.3f, 0.9f))

        assertTrue(abs(sqrt(q.sumOf { it * it }) - 1.0) < tolerance)
    }

    @Test
    fun aLevelFloorOneMeterBelowALevelCameraIsNormalUpAndOffsetOne() {
        val camera = pose(0f, 0f, 0f, 0f, 0f, 0f, 1f)
        val floorCenter = pose(0f, -1f, 0f, 0f, 0f, 0f, 1f)

        val plane = ArCoreToWire.planeInCameraFrame(floorCenter, camera)

        // Up in the laptop's camera axes is -y.
        assertClose(doubleArrayOf(0.0, -1.0, 0.0), plane.normal)
        assertEquals(1.0, plane.offsetMeters, tolerance)
    }

    @Test
    fun theOffsetIsTheCamerasHeightWhereverTheCameraStandsAndFaces() {
        // Camera at (3, 1.6, -5) in the world, yawed, over the world floor at y = 0.
        val half = (Math.PI / 6).toFloat()
        val camera = pose(3f, 1.6f, -5f, 0f, sin(half), 0f, cos(half))
        val floorCenter = pose(10f, 0f, 10f, 0f, 0f, 0f, 1f)

        val plane = ArCoreToWire.planeInCameraFrame(floorCenter, camera)

        assertEquals(1.6, plane.offsetMeters, 1e-5)
        // The camera is at the origin of its own frame, so its height is the offset. A point on
        // the floor must sit at height zero: take the floor center in the camera frame.
        val worldToCamera = camera.inverse()
        val center = worldToCamera.transformPoint(floorCenter.translation)
        val centerOurs = doubleArrayOf(center[0].toDouble(), -center[1].toDouble(), -center[2].toDouble())
        val height = plane.normal[0] * centerOurs[0] + plane.normal[1] * centerOurs[1] + plane.normal[2] * centerOurs[2] + plane.offsetMeters
        assertEquals(0.0, height, 1e-5)
    }

    @Test
    fun aTiltedCameraStillSeesTheFloorNormalAsUnitLength() {
        // Pitched down 40 degrees about x, like a hand-held phone.
        val half = (-Math.PI * 40 / 360).toFloat()
        val camera = pose(0f, 1.6f, 0f, sin(half), 0f, 0f, cos(half))
        val floorCenter = pose(0f, 0f, -3f, 0f, 0f, 0f, 1f)

        val plane = ArCoreToWire.planeInCameraFrame(floorCenter, camera)

        assertTrue(abs(sqrt(plane.normal.sumOf { it * it }) - 1.0) < 1e-5)
        assertEquals(1.6, plane.offsetMeters, 1e-5)
    }

    @Test
    fun repackDropsRowPaddingAndMasksTheReservedBits() {
        // Two rows of two pixels, each row padded to eight bytes. One value has the top three
        // bits set: 0xE5DC must come out as 0x05DC, which is 1500 millimeters.
        val buffer = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putShort(0, 0xE5DC.toShort())
        buffer.putShort(2, 2000)
        buffer.putShort(8, 2500)
        buffer.putShort(10, 0x1FFF)

        val values = ArCoreToWire.DepthImageReader.repack(buffer, rows = 2, columns = 2, rowStride = 8, pixelStride = 2)

        assertEquals(listOf<Short>(1500, 2000, 2500, 0x1FFF.toShort()), values.toList())
    }

    @Test
    fun repackHonorsAPixelStrideWiderThanTwo() {
        val buffer = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putShort(0, 100)
        buffer.putShort(4, 200)

        val values = ArCoreToWire.DepthImageReader.repack(buffer, rows = 1, columns = 2, rowStride = 8, pixelStride = 4)

        assertEquals(listOf<Short>(100, 200), values.toList())
    }

    @Test
    fun repackRefusesStridesThatCannotHoldTheImage() {
        val buffer = ByteBuffer.allocate(16)

        // Each case breaks one rule with everything else valid, and each asserts which rule it
        // broke. The exception type alone is satisfied by any of the three, so a merged or
        // renamed check would have left all of this green.
        val tooNarrowARow = assertFailsWith<IllegalArgumentException> {
            ArCoreToWire.DepthImageReader.repack(buffer, 2, 2, rowStride = 2, pixelStride = 2)
        }
        assertTrue(tooNarrowARow.message!!.contains("row stride 2 cannot hold 2 pixels"), tooNarrowARow.message)

        val pixelTooSmall = assertFailsWith<IllegalArgumentException> {
            ArCoreToWire.DepthImageReader.repack(buffer, 2, 2, rowStride = 8, pixelStride = 1)
        }
        assertTrue(pixelTooSmall.message!!.contains("DEPTH16 pixels are two bytes"), pixelTooSmall.message)

        val noRows = assertFailsWith<IllegalArgumentException> {
            ArCoreToWire.DepthImageReader.repack(buffer, 0, 2, rowStride = 8, pixelStride = 2)
        }
        assertTrue(noRows.message!!.contains("positive size, got 0x2"), noRows.message)
    }

    @Test
    fun aZeroDepthValueStaysZeroWhichTheLaptopTreatsAsUnknown() {
        val buffer = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)

        val values = ArCoreToWire.DepthImageReader.repack(buffer, 1, 2, rowStride = 4, pixelStride = 2)

        assertEquals(listOf<Short>(0, 0), values.toList())
    }
}
