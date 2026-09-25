package com.example.sidewalkvision

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log2
import kotlin.math.sign
import kotlin.math.sqrt

// Kotlin form of docs/math/surprise_and_tau.md. The monitor counts lengths in path widths: every
// quantity the surprise channels read is a ratio of two lengths, so the unit cancels.

// Welford's convention: 96% of hits land inside the target, about ±2.07 standard deviations.
const val TOLERANCE_Z = 2.07

// Δt, one unit of time. Vertegaal et al. measured spread over 1-second windows.
const val WINDOW_SECONDS = 1.0

// T, how far ahead the display looks. The math doc's example value, to be measured from gaze.
const val LOOKAHEAD_SECONDS = 0.5

// θ₁ and θ₀: on when time to the edge drops below one window, off only once it is back above
// one and a half, so the alarm doesn't flicker at the boundary.
const val ALARM_ON_BITS = 0.72
const val ALARM_OFF_BITS = 0.32

// Half the path, for a rider treated as a point. The path is one unit wide.
const val USABLE_HALF_WIDTH = 0.5

// Past this gap between reliable frames, two positions are too far apart in time to difference
// into a speed, and the last reading is too old to show.
const val MAX_FRAME_GAP_SECONDS = 1.0

private val TWO_LN_2 = 2 * ln(2.0)

fun smoothingWeight(elapsedSeconds: Double, windowSeconds: Double = WINDOW_SECONDS): Double =
    1 - exp(-elapsedSeconds / windowSeconds)

/** Running mean and variance of the rider's position, the math doc's μₚ and σₚ². */
data class RunningPosition(val mean: Double, val variance: Double) {
    fun update(position: Double, weight: Double): RunningPosition {
        val error = position - mean
        return RunningPosition(mean + weight * error, (1 - weight) * (variance + weight * error * error))
    }
}

/** Line surprise from its signal-to-noise ratio, Vertegaal Eq 15, in bits. */
fun lineSurpriseBits(signalToNoise: Double): Double = signalToNoise * signalToNoise / TWO_LN_2

/** 0 on the line, 1 at the tolerance, logarithmic in between because perception is. */
fun lineIntensity(signalToNoise: Double, z: Double = TOLERANCE_Z): Double =
    (log2(abs(signalToNoise) + 1) / log2(z + 1)).coerceIn(0.0, 1.0)

/** Gap to the edge the rider is moving toward. Zero or less means already past the tolerance. */
fun gapToEdge(position: Double, sidewaysSpeed: Double, usableHalfWidth: Double): Double =
    usableHalfWidth - position * sign(sidewaysSpeed)

/**
 * 1/τ, the closing speed over the gap. Zero when moving parallel to the edge, where τ itself would
 * be infinite, and infinite once past the tolerance.
 */
fun inverseTimeToEdge(position: Double, sidewaysSpeed: Double, usableHalfWidth: Double): Double {
    if (sidewaysSpeed == 0.0) return 0.0
    val gap = gapToEdge(position, sidewaysSpeed, usableHalfWidth)
    if (gap <= 0.0) return Double.POSITIVE_INFINITY
    return abs(sidewaysSpeed) / gap
}

/** Edge surprise, Vertegaal Eq 32 read as a time-to-contact, in bits. */
fun edgeSurpriseBits(inverseTau: Double, windowSeconds: Double = WINDOW_SECONDS): Double {
    val closing = windowSeconds * inverseTau
    return closing * closing / TWO_LN_2
}

/** Index of difficulty for the edge, the Eq 38 logarithmic form, in bits. */
fun edgeDifficultyBits(inverseTau: Double, windowSeconds: Double = WINDOW_SECONDS): Double =
    log2(windowSeconds * inverseTau + 1)

/** Switches on above one threshold and off below a lower one. */
class AlarmHysteresis(private val onBits: Double = ALARM_ON_BITS, private val offBits: Double = ALARM_OFF_BITS) {
    var isOn = false
        private set

    fun update(edgeSurprise: Double): Boolean {
        if (!isOn && edgeSurprise > onBits) isOn = true
        else if (isOn && edgeSurprise < offBits) isOn = false
        return isOn
    }

    fun reset() {
        isOn = false
    }
}

/** Both surprise channels for one frame. Lengths in path widths, times in seconds. */
data class SurpriseReading(
    // x, 0 in the middle of the path, positive to the right.
    val positionFromCenter: Double,
    // v, positive toward the right edge.
    val sidewaysSpeed: Double,
    val meanPosition: Double,
    val wobble: Double,
    val lineSignalToNoise: Double,
    val lineSurpriseBits: Double,
    // From the lookahead position x + v·T, for what the rider sees.
    val displayIntensity: Double,
    val gapToEdge: Double,
    // τ. Null while moving parallel to the edge, zero once past the tolerance.
    val secondsToEdge: Double?,
    val edgeSurpriseBits: Double,
    val edgeDifficultyBits: Double,
    val alarmOn: Boolean,
)

/**
 * Turns the camera's position across the path, frame by frame, into both surprise channels.
 *
 * Only frames with a reliable pose update it. The rest leave the running values as they were,
 * while time moves on. Sideways speed is the position differenced between reliable frames and
 * smoothed with the running average's weight, so it needs no GPS and no path width in meters.
 */
class SurpriseMonitor(
    private val usableHalfWidth: Double = USABLE_HALF_WIDTH,
    private val lineTarget: Double = 0.0,
) {
    private val lineTolerance = usableHalfWidth / TOLERANCE_Z
    private val alarm = AlarmHysteresis()
    private var running: RunningPosition? = null
    private var smoothedSpeed = 0.0
    private var previousPosition: Double? = null
    private var previousTimeNanos: Long? = null

    var latest: SurpriseReading? = null
        private set

    /**
     * Feeds one frame. Returns the current reading, or null when none is recent enough to trust.
     *
     * @param positionAcross 0 at the left edge, 1 at the right, from a reliable pose. Null for a
     * frame without one, which only ages the last reading.
     */
    fun update(timeNanos: Long, positionAcross: Double?): SurpriseReading? {
        val previousTime = previousTimeNanos
        val elapsedSeconds = previousTime?.let { (timeNanos - it) / 1e9 }

        if (positionAcross == null) {
            if (elapsedSeconds != null && elapsedSeconds > MAX_FRAME_GAP_SECONDS) {
                latest = null
                alarm.reset()
            }
            return latest
        }
        // A repeated or out-of-order frame carries no new time.
        if (elapsedSeconds != null && elapsedSeconds <= 0.0) return latest

        val position = positionAcross - 0.5
        val weight = elapsedSeconds?.let { smoothingWeight(it) } ?: 1.0
        running = running?.update(position, weight) ?: RunningPosition(position, 0.0)

        val previous = previousPosition
        smoothedSpeed = when {
            elapsedSeconds == null || previous == null -> 0.0
            // Too long since the last reliable frame to know how the position got here.
            elapsedSeconds > MAX_FRAME_GAP_SECONDS -> 0.0
            else -> smoothedSpeed + weight * ((position - previous) / elapsedSeconds - smoothedSpeed)
        }
        previousPosition = position
        previousTimeNanos = timeNanos

        val current = running!!
        val lineSignalToNoise = (current.mean - lineTarget) / lineTolerance
        val lookaheadSignalToNoise = (position + smoothedSpeed * LOOKAHEAD_SECONDS - lineTarget) / lineTolerance
        val inverseTau = inverseTimeToEdge(position, smoothedSpeed, usableHalfWidth)
        val edgeSurprise = edgeSurpriseBits(inverseTau)

        return SurpriseReading(
            positionFromCenter = position,
            sidewaysSpeed = smoothedSpeed,
            meanPosition = current.mean,
            wobble = sqrt(current.variance),
            lineSignalToNoise = lineSignalToNoise,
            lineSurpriseBits = lineSurpriseBits(lineSignalToNoise),
            displayIntensity = lineIntensity(lookaheadSignalToNoise),
            gapToEdge = gapToEdge(position, smoothedSpeed, usableHalfWidth),
            secondsToEdge = if (inverseTau == 0.0) null else 1 / inverseTau,
            edgeSurpriseBits = edgeSurprise,
            edgeDifficultyBits = edgeDifficultyBits(inverseTau),
            alarmOn = alarm.update(edgeSurprise),
        ).also { latest = it }
    }

    fun reset() {
        running = null
        smoothedSpeed = 0.0
        previousPosition = null
        previousTimeNanos = null
        latest = null
        alarm.reset()
    }
}
