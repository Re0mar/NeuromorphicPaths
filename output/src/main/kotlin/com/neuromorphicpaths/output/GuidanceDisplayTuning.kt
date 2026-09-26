package com.neuromorphicpaths.output

import com.neuromorphicpaths.core.Guidance
import kotlin.math.max
import kotlin.math.min

/**
 * Every number the displays decide with, in one place so they can be tuned without reading the
 * drawing code. Each one is marked TUNABLE with its unit and why it has the value it has.
 *
 * The field's own numbers, horizons, clearances and how far ahead the no-way-through question
 * looks, are in `PushFieldParameters` in the math module. These are only about what the screen
 * shows for a given field answer.
 */
data class GuidanceDisplayTuning(
    /** TUNABLE, degrees. Below this the small overlay shows a straight arrow, above it a 45 degree bend. */
    val bearTurnDegrees: Double = 20.0,
    /**
     * TUNABLE, degrees. Above this the small overlay shows a 90 degree bend. The field searches
     * no further than 60 degrees, so the plain halfway point between 45 and 90 would never show.
     */
    val hardTurnDegrees: Double = 45.0,
    /** TUNABLE, degrees. How far past a cutoff the heading has to go before the symbol changes, so it does not flicker at two updates a second. */
    val symbolHysteresisDegrees: Double = 5.0,
    /**
     * TUNABLE, bits. Where the color reaches full red. Applies to the frame's surprise and to the
     * lowest surprise ahead alike, so red always comes before a STOP. Three bits was the cone at
     * its worst on the outdoor walk.
     */
    val redSurpriseBits: Double = 3.0,
    /**
     * TUNABLE, bits. The lowest surprise ahead at which there is no way through. A wall straight
     * across the path reaches it about 2.8 m out. No frame of the outdoor walk held it for a second.
     */
    val stopSurpriseBits: Double = 4.0,
    /** TUNABLE, seconds. How long the lowest surprise ahead has to stay at [stopSurpriseBits] before STOP shows. Filters one-frame spikes. */
    val stopHoldSeconds: Double = 1.0,
    /** TUNABLE, meters per second. At or below this measured speed the walker counts as stopped, and a STOP becomes a U-turn. */
    val stoppedSpeedMetersPerSecond: Double = 0.05,
    /** TUNABLE, 0 to 1. How strong the full-screen ribbon is at the walker's feet, normally. */
    val fullScreenNormalOpacity: Float = 0.5f,
    /**
     * TUNABLE, 0 to 1. How strong the full-screen ribbon is at the walker's feet in an emergency:
     * red surprise, or STOP. At Android's touch-through limit, so it cannot go higher.
     */
    val fullScreenEmergencyOpacity: Float = 0.8f,
    /** TUNABLE, dp. Width and height of the small overlay's symbol. */
    val cornerSymbolSizeDp: Int = 96,
) {
    init {
        require(0.0 < bearTurnDegrees && bearTurnDegrees < hardTurnDegrees) { "bearTurnDegrees must be positive and below hardTurnDegrees" }
        require(symbolHysteresisDegrees >= 0.0) { "symbolHysteresisDegrees cannot be negative" }
        require(redSurpriseBits > 0.0 && stopSurpriseBits > 0.0 && stopHoldSeconds >= 0.0) { "surprise thresholds must be positive" }
        // Android lets taps through an overlay window only up to this combined opacity. Above it
        // the app underneath stops taking touches while the ribbon is up.
        require(fullScreenNormalOpacity <= MAXIMUM_TOUCH_THROUGH_OPACITY && fullScreenEmergencyOpacity <= MAXIMUM_TOUCH_THROUGH_OPACITY) {
            "full-screen opacity above $MAXIMUM_TOUCH_THROUGH_OPACITY blocks touches to the app underneath"
        }
    }

    /**
     * How red to draw, 0 for calm blue and 1 for full red. Whichever is higher of the frame's
     * surprise and the lowest surprise ahead, so the color warns before a STOP does.
     */
    fun alertFraction(guidance: Guidance): Float {
        val surprise = max(guidance.overallSurpriseBits, guidance.lowestSurpriseAheadBits ?: 0.0)
        return min(1.0, surprise / redSurpriseBits).toFloat()
    }

    companion object {
        /** Android's default maximum obscuring opacity for touches, since Android 12. */
        const val MAXIMUM_TOUCH_THROUGH_OPACITY = 0.8f
    }
}
