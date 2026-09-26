package com.neuromorphicpaths.input

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class StepCadenceSpeedEstimatorTest {

    private val sampleSeconds = 0.01

    /** Feeds [seconds] of a walk at [stepsPerSecond] with bumps of [amplitude], starting at [startSeconds]. */
    private fun walk(estimator: StepCadenceSpeedEstimator, startSeconds: Double, seconds: Double, stepsPerSecond: Double, amplitude: Double = 3.0): Double? {
        var latest: Double? = null
        var t = startSeconds
        while (t < startSeconds + seconds) {
            latest = estimator.add((t * 1e9).toLong(), amplitude * sin(2 * PI * stepsPerSecond * t))
            t += sampleSeconds
        }
        return latest
    }

    /** Feeds [seconds] of standing still: small noise well under the threshold. */
    private fun stand(estimator: StepCadenceSpeedEstimator, startSeconds: Double, seconds: Double): Double? {
        var latest: Double? = null
        var t = startSeconds
        var sign = 1.0
        while (t < startSeconds + seconds) {
            latest = estimator.add((t * 1e9).toLong(), 0.2 * sign)
            sign = -sign
            t += sampleSeconds
        }
        return latest
    }

    @Test
    fun aSteadyWalkReadsCadenceTimesStepLength() {
        val estimator = StepCadenceSpeedEstimator()
        val speed = walk(estimator, startSeconds = 0.0, seconds = 10.0, stepsPerSecond = 1.8)

        assertNotNull(speed)
        assertEquals(1.8 * StepCadenceSpeedEstimator.DEFAULT_STEP_LENGTH_METERS, speed!!, 0.05)
    }

    @Test
    fun aFasterCadenceReadsFaster() {
        val slow = walk(StepCadenceSpeedEstimator(), 0.0, 10.0, stepsPerSecond = 1.5)!!
        val quick = walk(StepCadenceSpeedEstimator(), 0.0, 10.0, stepsPerSecond = 2.2)!!

        assertTrue(quick > slow)
    }

    @Test
    fun standingStillReadsZeroAfterTheStopTime() {
        val estimator = StepCadenceSpeedEstimator()
        assertNull(stand(estimator, 0.0, 1.0))
        val speed = stand(estimator, 1.0, 2.0)

        assertEquals(0.0, speed!!, 1e-9)
    }

    @Test
    fun stoppingAfterAWalkDecaysTheSpeedToZero() {
        val estimator = StepCadenceSpeedEstimator()
        walk(estimator, 0.0, 10.0, stepsPerSecond = 1.8)
        val justStopped = stand(estimator, 10.0, 1.6)!!
        val stoppedForAWhile = stand(estimator, 11.6, 3.0)!!

        assertTrue("speed should still be coming down, was $justStopped", justStopped > 0.3)
        assertTrue("speed should be near zero after standing, was $stoppedForAWhile", stoppedForAWhile < 0.1)
    }

    @Test
    fun resetForgetsEverything() {
        val estimator = StepCadenceSpeedEstimator()
        walk(estimator, 0.0, 5.0, stepsPerSecond = 1.8)
        estimator.reset()

        assertNull(estimator.metersPerSecond)
        assertNull(stand(estimator, 100.0, 1.0))
    }
}
