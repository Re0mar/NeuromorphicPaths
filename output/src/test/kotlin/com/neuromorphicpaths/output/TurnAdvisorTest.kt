package com.neuromorphicpaths.output

import com.neuromorphicpaths.core.Guidance
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Symbol picking, its hysteresis, and the STOP and U-turn latch. */
class TurnAdvisorTest {

    private fun guidance(headingDegrees: Double = 0.0, surpriseBits: Double = 0.0, aheadBits: Double = 0.0) = Guidance(
        desiredHeadingRadians = Math.toRadians(headingDegrees),
        overallSurpriseBits = surpriseBits,
        perObstacle = emptyList(),
        lowestSurpriseAheadBits = aheadBits,
    )

    private fun seconds(value: Double) = (value * 1e9).toLong()

    @Test
    fun headingsSnapToStraightBearAndHard() {
        val cases = listOf(0.0 to TurnSymbol.FORWARD, 30.0 to TurnSymbol.BEAR_RIGHT, -30.0 to TurnSymbol.BEAR_LEFT, 55.0 to TurnSymbol.HARD_RIGHT, -60.0 to TurnSymbol.HARD_LEFT)
        for ((heading, expected) in cases) {
            // A fresh advisor each time, so hysteresis from an earlier case does not carry over.
            assertEquals("heading $heading", expected, TurnAdvisor().advise(guidance(heading), 1.2, 0L).symbol)
        }
    }

    @Test
    fun aHeadingOnACutoffDoesNotFlickerTheSymbol() {
        val advisor = TurnAdvisor()
        assertEquals(TurnSymbol.FORWARD, advisor.advise(guidance(18.0), 1.2, 0L).symbol)
        // Past the 20 degree cutoff but inside the 5 degree margin: still straight.
        assertEquals(TurnSymbol.FORWARD, advisor.advise(guidance(23.0), 1.2, 1L).symbol)
        assertEquals(TurnSymbol.BEAR_RIGHT, advisor.advise(guidance(26.0), 1.2, 2L).symbol)
        // Back under the cutoff, but not by the margin: still bearing right.
        assertEquals(TurnSymbol.BEAR_RIGHT, advisor.advise(guidance(17.0), 1.2, 3L).symbol)
        assertEquals(TurnSymbol.FORWARD, advisor.advise(guidance(14.0), 1.2, 4L).symbol)
    }

    @Test
    fun colorReachesRedFromEitherSurpriseAndAWayThroughIsNotAnEmergency() {
        val advisor = TurnAdvisor()
        assertEquals(0f, advisor.advise(guidance(), 1.2, 0L).alertFraction, 1e-6f)
        assertEquals(0.5f, advisor.advise(guidance(surpriseBits = 1.5), 1.2, 1L).alertFraction, 1e-6f)
        val aheadRed = advisor.advise(guidance(aheadBits = 3.2), 1.2, 2L)
        assertEquals(1f, aheadRed.alertFraction, 1e-6f)
        assertTrue(aheadRed.emergency)
        assertFalse(advisor.advise(guidance(surpriseBits = 2.0, aheadBits = 1.0), 1.2, 3L).emergency)
    }

    @Test
    fun stopNeedsTheSurpriseAheadHeldForASecond() {
        val advisor = TurnAdvisor()
        assertEquals(TurnSymbol.FORWARD, advisor.advise(guidance(aheadBits = 5.0), 1.2, seconds(10.0)).symbol)
        assertEquals(TurnSymbol.FORWARD, advisor.advise(guidance(aheadBits = 5.0), 1.2, seconds(10.5)).symbol)
        // A one-frame dip resets the hold, like the half-second spikes on the outdoor walk.
        advisor.advise(guidance(aheadBits = 2.0), 1.2, seconds(10.75))
        assertEquals(TurnSymbol.FORWARD, advisor.advise(guidance(aheadBits = 5.0), 1.2, seconds(11.25)).symbol)
        val stop = advisor.advise(guidance(aheadBits = 5.0), 1.2, seconds(12.25))
        assertEquals(TurnSymbol.STOP, stop.symbol)
        assertEquals(1f, stop.alertFraction, 1e-6f)
    }

    @Test
    fun stoppingTurnsStopIntoAUTurnThatHoldsUntilTheWalkerMovesAgain() {
        val advisor = TurnAdvisor()
        advisor.advise(guidance(aheadBits = 5.0), 1.2, seconds(0.0))
        assertEquals(TurnSymbol.STOP, advisor.advise(guidance(aheadBits = 5.0), 1.2, seconds(1.0)).symbol)
        // Standing still, the field's surprise ahead drops, and the U-turn holds anyway.
        assertEquals(TurnSymbol.U_TURN, advisor.advise(guidance(aheadBits = 0.5), 0.0, seconds(2.0)).symbol)
        assertEquals(TurnSymbol.U_TURN, advisor.advise(guidance(aheadBits = 0.2), 0.0, seconds(3.0)).symbol)
        // Walking again, now facing open ground: the way through is back.
        assertEquals(TurnSymbol.FORWARD, advisor.advise(guidance(aheadBits = 0.2), 1.1, seconds(4.0)).symbol)
    }

    @Test
    fun anUnknownSpeedNeverShowsAUTurn() {
        val advisor = TurnAdvisor()
        advisor.advise(guidance(aheadBits = 5.0), null, seconds(0.0))
        assertEquals(TurnSymbol.STOP, advisor.advise(guidance(aheadBits = 5.0), null, seconds(1.0)).symbol)
    }
}
