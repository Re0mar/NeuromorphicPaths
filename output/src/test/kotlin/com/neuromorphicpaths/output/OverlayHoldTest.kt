package com.neuromorphicpaths.output

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OverlayHoldTest {

    @Test
    fun detectionsAreAlwaysShown() {
        val hold = OverlayHold(holdMillis = 400)
        assertTrue(hold.shouldShow(hasDetections = true, nowMillis = 0))
        assertTrue(hold.shouldShow(hasDetections = true, nowMillis = 10))
    }

    @Test
    fun anEmptyResultWaitsOutTheHoldAfterDetections() {
        val hold = OverlayHold(holdMillis = 400)
        hold.shouldShow(hasDetections = true, nowMillis = 1_000)

        assertFalse(hold.shouldShow(hasDetections = false, nowMillis = 1_100))
        assertFalse(hold.shouldShow(hasDetections = false, nowMillis = 1_399))
        assertTrue(hold.shouldShow(hasDetections = false, nowMillis = 1_400))
    }

    @Test
    fun anEmptyResultIsShownWhenNothingWasSeenYet() {
        assertTrue(OverlayHold(holdMillis = 400).shouldShow(hasDetections = false, nowMillis = 0))
    }

    @Test
    fun resetForgetsTheLastDetections() {
        val hold = OverlayHold(holdMillis = 400)
        hold.shouldShow(hasDetections = true, nowMillis = 1_000)
        hold.reset()
        assertTrue(hold.shouldShow(hasDetections = false, nowMillis = 1_010))
    }
}
