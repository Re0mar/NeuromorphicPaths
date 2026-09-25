package com.example.sidewalkvision

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OverlayHoldTest {
    @Test
    fun aPathIsAlwaysShown() {
        val hold = OverlayHold(holdMillis = 400)
        assertTrue(hold.shouldShow(hasPath = true, nowMillis = 0))
        assertTrue(hold.shouldShow(hasPath = true, nowMillis = 10))
    }

    @Test
    fun anEmptyResultWaitsOutTheHoldAfterAPath() {
        val hold = OverlayHold(holdMillis = 400)
        hold.shouldShow(hasPath = true, nowMillis = 1_000)

        assertFalse(hold.shouldShow(hasPath = false, nowMillis = 1_100))
        assertFalse(hold.shouldShow(hasPath = false, nowMillis = 1_399))
        assertTrue(hold.shouldShow(hasPath = false, nowMillis = 1_400))
    }

    @Test
    fun anEmptyResultIsShownWhenNoPathWasSeenYet() {
        assertTrue(OverlayHold(holdMillis = 400).shouldShow(hasPath = false, nowMillis = 0))
    }

    @Test
    fun resetForgetsTheLastPath() {
        val hold = OverlayHold(holdMillis = 400)
        hold.shouldShow(hasPath = true, nowMillis = 1_000)
        hold.reset()
        assertTrue(hold.shouldShow(hasPath = false, nowMillis = 1_010))
    }
}
