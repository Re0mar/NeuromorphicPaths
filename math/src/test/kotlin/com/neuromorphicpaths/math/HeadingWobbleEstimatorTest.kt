package com.neuromorphicpaths.math

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin

private const val SECOND_NANOS = 1_000_000_000L
private const val TOLERANCE = 1e-9

class HeadingWobbleEstimatorTest {

    private fun degrees(radians: Double) = Math.toDegrees(radians)

    @Test
    fun nothingToSayBeforeTwoSamples() {
        val estimator = HeadingWobbleEstimator()
        assertNull(estimator.wobbleRadians)
        estimator.add(0L, 0.3)
        assertNull(estimator.wobbleRadians)
        assertEquals(0.3, estimator.meanHeadingRadians!!, TOLERANCE)
    }

    @Test
    fun aSteadyHeadingHasNoWobble() {
        val estimator = HeadingWobbleEstimator()
        for (step in 0..20) estimator.add(step * SECOND_NANOS / 10, 1.0)
        assertEquals(0.0, estimator.wobbleRadians!!, TOLERANCE)
        assertEquals(1.0, estimator.meanHeadingRadians!!, TOLERANCE)
    }

    @Test
    fun theSecondSampleMovesTheMeanByTheWindowWeight() {
        val estimator = HeadingWobbleEstimator(windowSeconds = 1.0)
        estimator.add(0L, 0.0)
        estimator.add(SECOND_NANOS, 0.5)

        // One second into a one-second window moves the mean 1 - e^-1 of the way.
        assertEquals(0.5 * (1.0 - exp(-1.0)), estimator.meanHeadingRadians!!, TOLERANCE)
    }

    @Test
    fun theSameGapGivesTheSameResultWhateverTheSampleRate() {
        val everySecond = HeadingWobbleEstimator()
        everySecond.add(0L, 0.0)
        everySecond.add(SECOND_NANOS, 0.4)
        everySecond.add(2 * SECOND_NANOS, 0.4)

        val everyTwoSeconds = HeadingWobbleEstimator()
        everyTwoSeconds.add(0L, 0.0)
        everyTwoSeconds.add(2 * SECOND_NANOS, 0.4)

        assertEquals(everyTwoSeconds.meanHeadingRadians!!, everySecond.meanHeadingRadians!!, TOLERANCE)
    }

    @Test
    fun aSwayOfFiveDegreesReadsAsAFewDegreesOfWobble() {
        // Heading swings plus and minus 5 degrees with a 1.2 s stride, sampled at 30 Hz for 20 s.
        val estimator = HeadingWobbleEstimator()
        val amplitude = Math.toRadians(5.0)
        for (step in 0..600) {
            val seconds = step / 30.0
            estimator.add((seconds * SECOND_NANOS).toLong(), amplitude * sin(2 * PI * seconds / 1.2))
        }
        val wobble = degrees(estimator.wobbleRadians!!)
        assertTrue("wobble should be a few degrees, was $wobble", wobble > 2.0 && wobble < 5.0)
        assertTrue("mean should stay near zero, was ${degrees(estimator.meanHeadingRadians!!)}", degrees(estimator.meanHeadingRadians!!) < 2.0)
    }

    @Test
    fun crossingTheCompassWrapIsASmallStepNotAFullTurn() {
        val estimator = HeadingWobbleEstimator()
        val justBelowWrap = PI - Math.toRadians(1.0)
        val justAboveWrap = -PI + Math.toRadians(1.0)
        for (step in 0..40) {
            estimator.add(step * SECOND_NANOS / 10, if (step % 2 == 0) justBelowWrap else justAboveWrap)
        }
        val wobble = degrees(estimator.wobbleRadians!!)
        assertTrue("wobble should be about a degree, was $wobble", wobble < 2.0)
    }

    @Test
    fun aSampleWithNoNewTimeIsIgnored() {
        val estimator = HeadingWobbleEstimator()
        estimator.add(SECOND_NANOS, 0.0)
        estimator.add(SECOND_NANOS, 2.0)
        estimator.add(0L, 2.0)
        assertEquals(0.0, estimator.meanHeadingRadians!!, TOLERANCE)
        assertNull(estimator.wobbleRadians)
    }

    @Test
    fun resetForgetsEverything() {
        val estimator = HeadingWobbleEstimator()
        estimator.add(0L, 1.0)
        estimator.add(SECOND_NANOS, 2.0)
        estimator.reset()
        assertNull(estimator.wobbleRadians)
        assertNull(estimator.meanHeadingRadians)
    }

    @Test
    fun wrapToPiBringsAnyAngleIntoRange() {
        assertEquals(0.0, HeadingWobbleEstimator.wrapToPi(2 * PI), TOLERANCE)
        assertEquals(-PI / 2, HeadingWobbleEstimator.wrapToPi(3 * PI / 2), TOLERANCE)
        assertEquals(PI / 2, HeadingWobbleEstimator.wrapToPi(-3 * PI / 2), TOLERANCE)
    }
}
