package com.neuromorphicpaths.input

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The estimator on a real phone: an excerpt of the outdoor recording's TotalAcceleration.csv,
 * the first six seconds standing on the spot and twenty seconds of walking from 60 s on.
 */
class StepCadenceOnRecordingTest {

    private val log: AccelerationLog by lazy {
        val text = checkNotNull(javaClass.getResourceAsStream("/total_acceleration_outdoor_excerpt.csv")) { "excerpt missing" }
            .bufferedReader().use { it.readText() }
        AccelerationLog.parse(text)
    }

    private fun speedsOver(fromSeconds: Double, toSeconds: Double): List<Double> {
        val estimator = StepCadenceSpeedEstimator()
        val start = log.samples.first().epochNanos
        val speeds = mutableListOf<Double>()
        for (sample in log.samples) {
            val seconds = (sample.epochNanos - start) / 1e9
            if (seconds > toSeconds) break
            val speed = estimator.add(sample.epochNanos, sample.magnitudeMinusGravity)
            if (seconds >= fromSeconds && speed != null) speeds += speed
        }
        return speeds
    }

    @Test
    fun standingAtTheStartReadsZero() {
        val speeds = speedsOver(fromSeconds = 2.0, toSeconds = 6.0)

        assertTrue(speeds.isNotEmpty())
        assertEquals(0.0, speeds.max(), 1e-9)
    }

    @Test
    fun walkingReadsAWalkingPace() {
        // The excerpt jumps from 6 s to 60 s, so the estimator sees a long gap and then steps.
        // By 65 s it has three intervals, and the pace holds through 80 s.
        val speeds = speedsOver(fromSeconds = 65.0, toSeconds = 80.0)
        val median = speeds.sorted()[speeds.size / 2]
        println("cadence pace on the excerpt, 65 to 80 s: median %.2f m/s, min %.2f, max %.2f".format(median, speeds.min(), speeds.max()))

        assertNotNull(median)
        assertTrue("median walking pace should be about 1 to 1.6 m/s, was $median", median in 1.0..1.6)
        assertTrue("the pace should never drop out mid-walk, min was ${speeds.min()}", speeds.min() > 0.6)
    }
}
