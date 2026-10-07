package com.neuromorphicpaths.pixel.wire

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * The message refuses to exist in a shape the display could not draw.
 *
 * The finite rule lives here and not in the decoder's tests because neither JSON library hands
 * the decoder a non-finite Double from text the same way. The test library makes a String of an
 * Infinity token and the string rule catches it first, which would prove nothing about this one.
 */
class PathMessageTest {
    private fun valid(
        timestamp: Double = 1.0,
        times: DoubleArray = doubleArrayOf(0.0, 0.1),
        offsets: DoubleArray = doubleArrayOf(0.0, 0.05),
        heading: Double = 0.1,
        cost: Double = 2.5,
    ) = PathMessage(timestamp, times, offsets, heading, false, cost)

    @Test
    fun aWellFormedMessageIsAccepted() {
        assertEquals(0.1, valid().lookaheadHeadingRadians)
    }

    @Test
    fun mismatchedArrayLengthsAreRefusedWithBothCounts() {
        val refusal = assertFailsWith<IllegalArgumentException> { valid(offsets = doubleArrayOf(0.0)) }
        assertTrue(refusal.message!!.contains("2 entries") && refusal.message!!.contains("has 1"), refusal.message)
    }

    @Test
    fun anEmptyPathIsRefused() {
        val refusal = assertFailsWith<IllegalArgumentException> { valid(times = doubleArrayOf(), offsets = doubleArrayOf()) }
        assertTrue(refusal.message!!.contains("at least one entry"), refusal.message)
    }

    @Test
    fun aNonFiniteScalarIsRefusedByName() {
        assertTrue(assertFailsWith<IllegalArgumentException> { valid(timestamp = Double.NaN) }.message!!.contains("timestamp_seconds"))
        assertTrue(assertFailsWith<IllegalArgumentException> { valid(heading = Double.POSITIVE_INFINITY) }.message!!.contains("lookahead_heading_radians"))
        assertTrue(assertFailsWith<IllegalArgumentException> { valid(cost = Double.NEGATIVE_INFINITY) }.message!!.contains("cumulative_cost_bits"))
    }

    @Test
    fun aNonFiniteArrayEntryIsRefusedWithItsIndex() {
        val refusal = assertFailsWith<IllegalArgumentException> { valid(offsets = doubleArrayOf(0.0, Double.NaN)) }
        assertTrue(refusal.message!!.contains("lateral_offsets_meters[1]"), refusal.message)
    }

    @Test
    fun equalityIsByContentNotByArrayReference() {
        assertEquals(valid(), valid())
        assertEquals(valid().hashCode(), valid().hashCode())
        assertNotEquals(valid(), valid(heading = 0.2))
    }
}
