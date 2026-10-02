package com.neuromorphicpaths.pixel.ar

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The gate between the display's draw rate and ARCore's frame rate. Without it the first phone
 * run sent sixty messages a second, every one a copy of its neighbour.
 */
class NewFrameGateTest {
    @Test
    fun theFirstFrameIsNew() {
        assertTrue(NewFrameGate().isNew(1_000L))
    }

    @Test
    fun theSameTimestampAgainIsNotNew() {
        val gate = NewFrameGate()
        gate.isNew(1_000L)

        assertFalse(gate.isNew(1_000L))
        assertFalse(gate.isNew(1_000L))
    }

    @Test
    fun aLaterTimestampIsNewAndTheOldOneIsForgotten() {
        val gate = NewFrameGate()
        gate.isNew(1_000L)

        assertTrue(gate.isNew(2_000L))
        // Change detection, not monotonicity. A session restart can hand back an older clock.
        assertTrue(gate.isNew(1_000L))
    }

    @Test
    fun sixtyDrawsOfTwoFramesSendTwo() {
        val gate = NewFrameGate()
        val draws = List(30) { 1_000L } + List(30) { 2_000L }

        assertEquals(2, draws.count { gate.isNew(it) })
    }
}
