package com.example.sidewalkvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import kotlin.math.exp

private const val SECOND_NANOS = 1_000_000_000L
private const val TOLERANCE = 1e-9

class GroundSpeedTest {
    @Test
    fun dopplerSpeedIsPreferredOverDistanceOverTime() {
        assertEquals(1.4, rawSpeedMetersPerSecond(1.4f, 5.0f, 1.0)!!, 1e-6)
    }

    @Test
    fun distanceOverTimeIsTheFallback() {
        assertEquals(1.5, rawSpeedMetersPerSecond(null, 3.0f, 2.0)!!, TOLERANCE)
    }

    @Test
    fun noReadingWithoutDopplerOrAPreviousFix() {
        assertNull(rawSpeedMetersPerSecond(null, null, null))
        assertNull(rawSpeedMetersPerSecond(null, 3.0f, 0.0))
    }

    @Test
    fun firstReadingIsTakenAsIs() {
        val smoother = SpeedSmoother()
        assertEquals(1.2, smoother.add(1.2, 0L), TOLERANCE)
    }

    @Test
    fun laterReadingsMoveTheAverageByTheWindowWeight() {
        val smoother = SpeedSmoother(windowSeconds = 1.0)
        smoother.add(1.0, 0L)

        val smoothed = smoother.add(2.0, SECOND_NANOS)

        // One second into a one-second window moves the average 1 - e^-1 of the way.
        assertEquals(1.0 + (1.0 - exp(-1.0)), smoothed, TOLERANCE)
    }

    @Test
    fun theSameGapGivesTheSameResultWhateverTheFixRate() {
        val everySecond = SpeedSmoother()
        everySecond.add(0.0, 0L)
        everySecond.add(2.0, SECOND_NANOS)
        val twoSteps = everySecond.add(2.0, 2 * SECOND_NANOS)

        val everyTwoSeconds = SpeedSmoother()
        everyTwoSeconds.add(0.0, 0L)
        val oneStep = everyTwoSeconds.add(2.0, 2 * SECOND_NANOS)

        assertEquals(oneStep, twoSteps, TOLERANCE)
    }

    @Test
    fun aFixWithNoNewTimeLeavesTheAverageAlone() {
        val smoother = SpeedSmoother()
        smoother.add(1.0, SECOND_NANOS)
        assertEquals(1.0, smoother.add(9.0, SECOND_NANOS), TOLERANCE)
        assertEquals(1.0, smoother.add(9.0, 0L), TOLERANCE)
    }

    @Test
    fun resetForgetsEverything() {
        val smoother = SpeedSmoother()
        smoother.add(3.0, 0L)
        smoother.reset()
        assertNull(smoother.metersPerSecond)
        assertEquals(0.5, smoother.add(0.5, SECOND_NANOS), TOLERANCE)
    }
}
