package com.neuromorphicpaths.output

import com.neuromorphicpaths.core.Guidance
import kotlin.math.abs

/** The small overlay's symbols, ordered left to right for the turns, then the two that say stop. */
enum class TurnSymbol {
    HARD_LEFT,
    BEAR_LEFT,
    FORWARD,
    BEAR_RIGHT,
    HARD_RIGHT,
    STOP,
    U_TURN,
}

/** What the floating overlay shows for one frame: which symbol, how red, and whether this is an emergency. */
data class TurnAdvice(
    val symbol: TurnSymbol,
    /** 0 for calm blue, 1 for full red. */
    val alertFraction: Float,
    val emergency: Boolean,
) {
    companion object {
        val CALM = TurnAdvice(TurnSymbol.FORWARD, 0f, emergency = false)
    }
}

/**
 * Turns the field's answer into one of a handful of symbols, remembering enough between frames
 * that the symbol holds still.
 *
 * The turn symbols come from the desired heading, snapped to straight, a 45 degree bend or a 90
 * degree bend, with a margin so a heading sitting on a cutoff does not flip the symbol back and
 * forth. STOP comes from the lowest surprise ahead: when even the least surprising heading stays
 * at [GuidanceDisplayTuning.stopSurpriseBits] for [GuidanceDisplayTuning.stopHoldSeconds],
 * there is no way through. Once the walker has stopped, STOP becomes a U-turn and stays one
 * until they walk again. It has to latch, because the field floors a standing walker's speed and
 * standing things fade at that speed, so the surprise ahead drops the moment they stop. When the
 * walker sets off again the field is asked afresh, and STOP clears once the surprise ahead is
 * back under the red line.
 *
 * Not thread safe. The pipeline calls it once per frame from one coroutine.
 */
class TurnAdvisor(private val tuning: GuidanceDisplayTuning = GuidanceDisplayTuning()) {

    private var headingSymbol = TurnSymbol.FORWARD
    private var noWayThroughSinceNanos: Long? = null
    private var stopActive = false

    fun advise(guidance: Guidance, walkerSpeedMetersPerSecond: Double?, timestampNanos: Long): TurnAdvice {
        val ahead = guidance.lowestSurpriseAheadBits ?: 0.0
        noWayThroughSinceNanos = if (ahead >= tuning.stopSurpriseBits) noWayThroughSinceNanos ?: timestampNanos else null
        val holdNanos = (tuning.stopHoldSeconds * NANOS_PER_SECOND).toLong()
        val heldSince = noWayThroughSinceNanos
        if (heldSince != null && timestampNanos - heldSince >= holdNanos) stopActive = true

        // An unknown speed counts as walking, so a missing estimator never shows a U-turn.
        val stopped = walkerSpeedMetersPerSecond != null && walkerSpeedMetersPerSecond <= tuning.stoppedSpeedMetersPerSecond
        if (stopActive && !stopped && ahead < tuning.redSurpriseBits) stopActive = false

        headingSymbol = nextHeadingSymbol(Math.toDegrees(guidance.desiredHeadingRadians))
        val symbol = when {
            stopActive && stopped -> TurnSymbol.U_TURN
            stopActive -> TurnSymbol.STOP
            else -> headingSymbol
        }
        val alert = if (stopActive) 1f else tuning.alertFraction(guidance)
        val emergency = stopActive || guidance.overallSurpriseBits >= tuning.redSurpriseBits || ahead >= tuning.redSurpriseBits
        return TurnAdvice(symbol, alert, emergency)
    }

    // Moves off the current symbol only once the heading is past the cutoff by the hysteresis margin.
    private fun nextHeadingSymbol(headingDegrees: Double): TurnSymbol {
        val current = TURNS.indexOf(headingSymbol)
        val raw = turnIndex(headingDegrees)
        val next = when {
            raw > current -> maxOf(current, turnIndex(headingDegrees - tuning.symbolHysteresisDegrees))
            raw < current -> minOf(current, turnIndex(headingDegrees + tuning.symbolHysteresisDegrees))
            else -> current
        }
        return TURNS[next]
    }

    private fun turnIndex(headingDegrees: Double): Int {
        val magnitude = abs(headingDegrees)
        val steps = when {
            magnitude >= tuning.hardTurnDegrees -> 2
            magnitude >= tuning.bearTurnDegrees -> 1
            else -> 0
        }
        return FORWARD_INDEX + if (headingDegrees < 0.0) -steps else steps
    }

    private companion object {
        val TURNS = listOf(TurnSymbol.HARD_LEFT, TurnSymbol.BEAR_LEFT, TurnSymbol.FORWARD, TurnSymbol.BEAR_RIGHT, TurnSymbol.HARD_RIGHT)
        const val FORWARD_INDEX = 2
        const val NANOS_PER_SECOND = 1_000_000_000.0
    }
}
