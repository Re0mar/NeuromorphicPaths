package com.example.sidewalkvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt

// The expected values are the worked scenarios in docs/math/surprise_and_tau.md, in meters with
// a 0.7 m usable half-width and samples every 0.1 s. The formulas don't care about the unit.
private const val DOC_HALF_WIDTH = 0.7
private const val DOC_TOLERANCE = 0.338
private val DOC_WEIGHT = smoothingWeight(0.1)
private const val FRAME_NANOS = 100_000_000L

class SurpriseMonitorTest {
    @Test
    fun theSmoothingWeightMatchesTheDoc() {
        assertEquals(0.0952, DOC_WEIGHT, 1e-4)
    }

    @Test
    fun runningPositionSnapshot() {
        val next = RunningPosition(mean = 0.010, variance = 0.00125).update(0.045, DOC_WEIGHT)
        assertEquals(0.0133, next.mean, 1e-4)
        assertEquals(0.001237, next.variance, 2e-6)
    }

    @Test
    fun runningPositionFollowsASteadyWobble() {
        // x = 0.05 sin(2 pi t / 1.2), from mean and variance zero.
        var running = RunningPosition(0.0, 0.0)
        val checkpoints = mapOf(1 to (0.0024 to 0.0073), 6 to (0.0133 to 0.0189), 30 to (0.0090 to 0.0322))
        for (step in 1..30) {
            running = running.update(0.05 * sin(2 * PI * step * 0.1 / 1.2), DOC_WEIGHT)
            checkpoints[step]?.let { (mean, wobble) ->
                assertEquals("mean at step $step", mean, running.mean, 2e-4)
                assertEquals("wobble at step $step", wobble, sqrt(running.variance), 2e-4)
            }
        }
    }

    @Test
    fun runningPositionLagsASuddenPush() {
        // From the settled wobble, pushed right at 0.5 m/s for 0.6 s, then held at 0.30 m.
        var running = RunningPosition(0.0, 0.0354 * 0.0354)
        for (step in 1..30) {
            running = running.update(minOf(0.05 * step, 0.30), DOC_WEIGHT)
            if (step == 6) {
                assertEquals(0.0855, running.mean, 2e-4)
                assertEquals(0.1132, sqrt(running.variance), 2e-4)
            }
        }
        assertEquals(0.2805, running.mean, 2e-4)
        assertEquals(0.0704, sqrt(running.variance), 2e-4)
    }

    @Test
    fun lineSurpriseMatchesTheDoc() {
        assertEquals(0.0009, lineSurpriseBits(0.012 / DOC_TOLERANCE), 1e-4)
        assertEquals(0.252, lineSurpriseBits(0.20 / DOC_TOLERANCE), 1e-3)
        // The lookahead example: x = 0.045 m, v = 0.15 m/s, T = 0.5 s.
        assertEquals(0.091, lineSurpriseBits((0.045 + 0.15 * LOOKAHEAD_SECONDS) / DOC_TOLERANCE), 1e-3)
    }

    @Test
    fun lineIntensityRunsFromTheLineToTheTolerance() {
        assertEquals(0.0, lineIntensity(0.0), 1e-12)
        assertEquals(1.0, lineIntensity(TOLERANCE_Z), 1e-12)
        assertEquals(1.0, lineIntensity(-5.0), 1e-12)
    }

    @Test
    fun edgeWhileStayingClear() {
        val inverseTau = inverseTimeToEdge(0.0433, 0.131, DOC_HALF_WIDTH)
        assertEquals(0.199, inverseTau, 1e-3)
        assertEquals(0.029, edgeSurpriseBits(inverseTau), 1e-3)
        assertEquals(0.262, edgeDifficultyBits(inverseTau), 1e-3)
    }

    @Test
    fun edgeWhileHeadingForIt() {
        val inverseTau = inverseTimeToEdge(0.25, 0.5, DOC_HALF_WIDTH)
        assertEquals(1.11, inverseTau, 1e-2)
        assertEquals(0.891, edgeSurpriseBits(inverseTau), 1e-3)
        assertEquals(1.078, edgeDifficultyBits(inverseTau), 1e-3)
    }

    @Test
    fun movingParallelOrPastTheEdge() {
        assertEquals(0.0, inverseTimeToEdge(0.3, 0.0, DOC_HALF_WIDTH), 0.0)
        assertEquals(Double.POSITIVE_INFINITY, inverseTimeToEdge(0.8, 0.1, DOC_HALF_WIDTH), 0.0)
    }

    @Test
    fun theAlarmFollowsThePushTowardTheEdge() {
        // The doc's streaming table: 0.5 m/s toward the right edge from the middle.
        val expectedSurprise = listOf(0.427, 0.501, 0.596, 0.721, 0.891, 1.127)
        val alarm = AlarmHysteresis()
        val states = expectedSurprise.indices.map { step ->
            val surprise = edgeSurpriseBits(inverseTimeToEdge(0.05 * (step + 1), 0.5, DOC_HALF_WIDTH))
            assertEquals(expectedSurprise[step], surprise, 1e-3)
            alarm.update(surprise)
        }
        // The doc labels 0.721 "at threshold". It is 0.7213, just over 0.72, so the alarm is on.
        assertEquals(listOf(false, false, false, true, true, true), states)
    }

    @Test
    fun theAlarmHoldsUntilSurpriseDropsBelowTheLowerThreshold() {
        val alarm = AlarmHysteresis()
        assertTrue(alarm.update(0.9))
        assertTrue(alarm.update(0.5))
        assertTrue(alarm.update(0.33))
        assertFalse(alarm.update(0.31))
        assertFalse(alarm.update(0.7))
    }

    @Test
    fun aRiderHoldingTheMiddleRaisesNoAlarm() {
        val monitor = SurpriseMonitor()
        var reading: SurpriseReading? = null
        for (frame in 0..30) {
            reading = monitor.update(frame * FRAME_NANOS, 0.5)
        }
        assertNotNull(reading)
        assertFalse(reading!!.alarmOn)
        assertNull(reading.secondsToEdge)
        assertEquals(0.0, reading.lineSurpriseBits, 1e-12)
    }

    @Test
    fun aRiderDriftingRightRaisesTheAlarmBeforeTheEdge() {
        // A tenth of the path width per second toward the right, starting from the middle.
        val monitor = SurpriseMonitor()
        var firstAlarmPosition: Double? = null
        for (frame in 0..60) {
            val reading = monitor.update(frame * FRAME_NANOS, 0.5 + 0.1 * frame * 0.1)!!
            if (reading.alarmOn && firstAlarmPosition == null) firstAlarmPosition = reading.positionFromCenter
        }
        assertNotNull("the alarm should come on", firstAlarmPosition)
        assertTrue("alarm at x = $firstAlarmPosition should come before the edge", firstAlarmPosition!! < USABLE_HALF_WIDTH)
    }

    @Test
    fun framesWithoutAPoseAgeTheReadingUntilItIsDropped() {
        val monitor = SurpriseMonitor()
        monitor.update(0L, 0.5)
        assertNotNull("half a second without a pose keeps the reading", monitor.update(500_000_000L, null))
        assertNull("over a second without a pose drops it", monitor.update(1_500_000_000L, null))
    }

    @Test
    fun aLongGapDoesNotTurnIntoASpeed() {
        val monitor = SurpriseMonitor()
        monitor.update(0L, 0.5)
        // Two seconds later the rider stands elsewhere. How they got there is unknown.
        val reading = monitor.update(2_000_000_000L, 0.7)!!
        assertEquals(0.0, reading.sidewaysSpeed, 0.0)
        assertFalse(reading.alarmOn)
    }
}
