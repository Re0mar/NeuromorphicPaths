package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.WalkerState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

private const val SECOND_NANOS = 1_000_000_000L

class PushFieldTurnToleranceTest {

    /** Twenty seconds of a walker swaying plus and minus [amplitudeDegrees], at 30 Hz. */
    private fun feedSway(field: PushFieldGuidance, amplitudeDegrees: Double) {
        val amplitude = Math.toRadians(amplitudeDegrees)
        for (step in 0..600) {
            val seconds = step / 30.0
            val walker = WalkerState(headingRadians = 0.0, speedMetersPerSecond = null, azimuthRadians = amplitude * sin(2 * PI * seconds / 1.2))
            field.evaluate(emptyList(), walker, (seconds * SECOND_NANOS).toLong())
        }
    }

    @Test
    fun withoutAnAzimuthNothingIsEstimatedAndTheToleranceIsTheParameter() {
        val field = PushFieldGuidance()
        val guidance = field.evaluate(emptyList(), WalkerState.ALIGNED_WITH_CAMERA, 0L)

        assertNull(guidance.walkerWobbleRadians)
        assertEquals(Math.toRadians(30.0), guidance.turnToleranceRadians!!, 1e-12)
    }

    @Test
    fun byDefaultTheWobbleIsReportedButTheToleranceStaysPut() {
        val field = PushFieldGuidance()
        feedSway(field, amplitudeDegrees = 5.0)
        val guidance = field.evaluate(emptyList(), WalkerState.ALIGNED_WITH_CAMERA, 21 * SECOND_NANOS)

        assertNotNull(guidance.walkerWobbleRadians)
        assertEquals(Math.toRadians(30.0), guidance.turnToleranceRadians!!, 1e-12)
        assertEquals(Math.toRadians(30.0), field.turnToleranceRadians, 1e-12)
    }

    @Test
    fun whenSwitchedOnTheToleranceIsTheWobbleTimesTheRatio() {
        val parameters = PushFieldParameters(turnToleranceFromWobble = true, minimumTurnToleranceRadians = Math.toRadians(1.0))
        val field = PushFieldGuidance(parameters)
        feedSway(field, amplitudeDegrees = 5.0)
        val guidance = field.evaluate(emptyList(), WalkerState.ALIGNED_WITH_CAMERA, 21 * SECOND_NANOS)

        val wobble = guidance.walkerWobbleRadians!!
        assertEquals(wobble * PushFieldParameters.WELFORD_BAND_Z, guidance.turnToleranceRadians!!, 1e-12)
    }

    @Test
    fun theFloorAndTheSearchRangeBoundTheTolerance() {
        val stiff = PushFieldGuidance(PushFieldParameters(turnToleranceFromWobble = true))
        feedSway(stiff, amplitudeDegrees = 0.5)
        val stiffGuidance = stiff.evaluate(emptyList(), WalkerState.ALIGNED_WITH_CAMERA, 21 * SECOND_NANOS)
        assertEquals(Math.toRadians(5.0), stiffGuidance.turnToleranceRadians!!, 1e-12)

        val loose = PushFieldGuidance(PushFieldParameters(turnToleranceFromWobble = true))
        feedSway(loose, amplitudeDegrees = 80.0)
        val looseGuidance = loose.evaluate(emptyList(), WalkerState.ALIGNED_WITH_CAMERA, 21 * SECOND_NANOS)
        assertEquals(Math.toRadians(60.0), looseGuidance.turnToleranceRadians!!, 1e-12)
    }
}
