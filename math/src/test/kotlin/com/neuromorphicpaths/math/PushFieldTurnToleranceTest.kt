package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.WalkerState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

private const val SECOND_NANOS = 1_000_000_000L

/** The wobble is measured and reported. The tolerance is a parameter and nothing moves it. */
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
    fun theWobbleIsReportedAndTheToleranceStaysPut() {
        val field = PushFieldGuidance()
        feedSway(field, amplitudeDegrees = 5.0)
        val guidance = field.evaluate(emptyList(), WalkerState.ALIGNED_WITH_CAMERA, 21 * SECOND_NANOS)

        val wobble = guidance.walkerWobbleRadians
        assertNotNull(wobble)
        assertTrue("a 5 degree sway should read a few degrees of wobble, was ${Math.toDegrees(wobble!!)}", wobble > Math.toRadians(2.0) && wobble < Math.toRadians(5.0))
        assertEquals(Math.toRadians(30.0), guidance.turnToleranceRadians!!, 1e-12)
        assertEquals(Math.toRadians(30.0), field.turnToleranceRadians, 1e-12)
    }

    @Test
    fun aDifferentToleranceIsAParameterAndChangesTheTurnCost() {
        val narrow = PushFieldGuidance(PushFieldParameters(turnToleranceRadians = Math.toRadians(15.0)))
        val wide = PushFieldGuidance(PushFieldParameters(turnToleranceRadians = Math.toRadians(45.0)))

        assertTrue(narrow.turnCostBits(Math.toRadians(20.0)) > wide.turnCostBits(Math.toRadians(20.0)))
        assertEquals(Math.toRadians(15.0), narrow.evaluate(emptyList(), WalkerState.ALIGNED_WITH_CAMERA, 0L).turnToleranceRadians!!, 1e-12)
    }
}
