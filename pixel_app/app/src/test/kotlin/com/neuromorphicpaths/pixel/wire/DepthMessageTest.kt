package com.neuromorphicpaths.pixel.wire

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** The message refuses to exist in a shape the laptop would refuse, so the refusal happens here. */
class DepthMessageTest {
    private fun valid(
        depth: ShortArray = ShortArray(6),
        rows: Int = 2,
        columns: Int = 3,
        intrinsics: DoubleArray = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        orientation: DoubleArray = doubleArrayOf(1.0, 0.0, 0.0, 0.0),
        position: DoubleArray? = doubleArrayOf(0.0, 0.0, 0.0),
    ) = DepthMessage(1.0, depth, rows, columns, intrinsics, orientation, position, null)

    @Test
    fun aWellFormedMessageIsAccepted() {
        val message = valid()
        assertTrue(message.hasPosition)
    }

    @Test
    fun hasPositionFollowsThePositionAndNothingElse() {
        assertFalse(valid(position = null).hasPosition)
        assertTrue(valid(position = doubleArrayOf(1.0, 2.0, 3.0)).hasPosition)
    }

    @Test
    fun depthMustMatchRowsTimesColumns() {
        assertFailsWith<IllegalArgumentException> { valid(depth = ShortArray(5)) }
        assertFailsWith<IllegalArgumentException> { valid(depth = ShortArray(7)) }
    }

    @Test
    fun zeroOrNegativeSizesAreRefused() {
        assertFailsWith<IllegalArgumentException> { valid(depth = ShortArray(0), rows = 0, columns = 3) }
        assertFailsWith<IllegalArgumentException> { valid(depth = ShortArray(0), rows = 2, columns = 0) }
        assertFailsWith<IllegalArgumentException> { valid(depth = ShortArray(6), rows = -2, columns = -3) }
    }

    @Test
    fun intrinsicsMustBeNineValues() {
        assertFailsWith<IllegalArgumentException> { valid(intrinsics = DoubleArray(4)) }
        assertFailsWith<IllegalArgumentException> { valid(intrinsics = DoubleArray(16)) }
    }

    @Test
    fun orientationMustBeAQuaternion() {
        assertFailsWith<IllegalArgumentException> { valid(orientation = DoubleArray(3)) }
    }

    @Test
    fun positionMustBeThreeValuesWhenPresent() {
        assertFailsWith<IllegalArgumentException> { valid(position = DoubleArray(2)) }
    }

    @Test
    fun aGroundPlaneNormalMustBeThreeValues() {
        assertFailsWith<IllegalArgumentException> { GroundPlane(DoubleArray(2), 1.0) }
        assertEquals(3, GroundPlane(doubleArrayOf(0.0, -1.0, 0.0), 1.6).normal.size)
    }

    @Test
    fun theLargestUint16DepthSurvivesAsAShort() {
        // 65535 does not fit a signed Short as a positive number. It is stored as -1 and the
        // encoder writes its two bytes unchanged, which the laptop reads back as 65535.
        val message = valid(depth = shortArrayOf(0xFFFF.toShort(), 0, 0, 0, 0, 0))
        val encoded = FrameEncoder.encode(message)
        val depthStart = encoded.size - 12
        assertEquals(0xFF.toByte(), encoded[depthStart])
        assertEquals(0xFF.toByte(), encoded[depthStart + 1])
    }
}
