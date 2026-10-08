package com.neuromorphicpaths.pixel.ui

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The overlay's arithmetic. The composable and the GL picture are device only and are read by
 * eye on the phone. What can be wrong by a sign or a unit is here.
 */
class ArrowOverlayTest {
    private val baseX = 200f
    private val baseY = 600f
    private val length = 300f

    // The widest heading the planner asks for, atan(max lateral speed / walking speed) on the laptop.
    private val sidestepLimitRadians = atan2(1.0, 1.4)

    @Test
    fun aZeroHeadingPointsStraightUp() {
        val tip = ArrowGeometry.headingToArrowTip(0.0, length, baseX, baseY)

        assertEquals(baseX, tip.x, 1e-3f)
        assertEquals(baseY - length, tip.y, 1e-3f)
    }

    @Test
    fun aPositiveHeadingPointsRight() {
        // The web page and the OpenCV window draw a positive heading to the right. This must too.
        val tip = ArrowGeometry.headingToArrowTip(0.3, length, baseX, baseY)

        assertTrue(tip.x > baseX, "tip x ${tip.x} is not right of the base")
        assertTrue(tip.y < baseY, "the arrow still points up")
    }

    @Test
    fun aNegativeHeadingPointsLeft() {
        val tip = ArrowGeometry.headingToArrowTip(-0.3, length, baseX, baseY)

        assertTrue(tip.x < baseX, "tip x ${tip.x} is not left of the base")
        assertTrue(tip.y < baseY)
    }

    @Test
    fun theTipStaysInsideTheCanvasAtTheHeadingLimit() {
        // A phone-shaped canvas, 400 by 800, base at the overlay's fractions. At plus and minus
        // the sidestep limit the tip must not leave it, or the widest step is the one not shown.
        val width = 400f
        val height = 800f
        val base = height * 0.8f
        val arrowLength = height * 0.35f
        for (heading in doubleArrayOf(-sidestepLimitRadians, sidestepLimitRadians)) {
            val tip = ArrowGeometry.headingToArrowTip(heading, arrowLength, width / 2f, base)

            assertTrue(tip.x in 0f..width && tip.y in 0f..height, "tip $tip left the canvas at heading $heading")
        }
    }

    @Test
    fun theHeadingTextAlwaysCarriesASign() {
        assertEquals("+12 deg", ArrowGeometry.headingText(Math.toRadians(12.4)))
        // The limit is 35.54 degrees, so it reads as 36. The sign is the point here.
        assertEquals("-36 deg", ArrowGeometry.headingText(-sidestepLimitRadians))
        assertEquals("+0 deg", ArrowGeometry.headingText(0.0))
    }

    @Test
    fun theAgeTextRoundsToATenthOfASecond() {
        assertEquals("0.0 s ago", ArrowGeometry.ageText(0L))
        assertEquals("0.1 s ago", ArrowGeometry.ageText(120L))
        assertEquals("0.9 s ago", ArrowGeometry.ageText(949L))
    }

    @Test
    fun anAgeOverASecondIsMarkedStale() {
        assertEquals("1.4 s ago, stale", ArrowGeometry.ageText(1_400L))
        assertEquals("1.0 s ago", ArrowGeometry.ageText(1_000L), "exactly a second is the boundary, not past it")
    }

    @Test
    fun aClockThatRanBackwardsReadsAsZeroNotNegative() {
        assertEquals("0.0 s ago", ArrowGeometry.ageText(-50L))
    }

    private fun assertEquals(expected: Float, actual: Float, tolerance: Float) {
        assertTrue(abs(expected - actual) <= tolerance, "expected $expected, got $actual")
    }
}
