package com.neuromorphicpaths.pixel.ui

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The part of "when was this path first drawn" that a JVM test can reach. Whether the overlay
 * really calls the gate on a real draw is device only, and the walk's log shows it: a drawn row for
 * every received one.
 */
class FirstDrawGateTest {
    private data class Item(val name: String)

    @Test
    fun theFirstSightOfAnItemIsAFirstDraw() {
        val gate = FirstDrawGate<Item>()

        assertTrue(gate.isFirstDraw(Item("a")))
    }

    @Test
    fun theSameItemDrawnAgainIsNot() {
        val gate = FirstDrawGate<Item>()
        val item = Item("a")
        gate.isFirstDraw(item)

        assertFalse(gate.isFirstDraw(item))
        assertFalse(gate.isFirstDraw(item))
    }

    @Test
    fun anEqualButSeparateItemIsAFirstDraw() {
        // Two paths can carry identical numbers and still be two arrivals. Identity, not equality.
        val gate = FirstDrawGate<Item>()
        gate.isFirstDraw(Item("a"))

        assertTrue(gate.isFirstDraw(Item("a")))
    }

    @Test
    fun aNewItemAfterAnOldOneIsAFirstDraw() {
        val gate = FirstDrawGate<Item>()
        gate.isFirstDraw(Item("a"))

        assertTrue(gate.isFirstDraw(Item("b")))
    }

    @Test
    fun nothingToDrawIsNeverAFirstDraw() {
        val gate = FirstDrawGate<Item>()

        assertFalse(gate.isFirstDraw(null))
    }
}
