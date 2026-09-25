package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.Guidance
import com.neuromorphicpaths.core.GuidanceField
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.ObstacleSurprise
import com.neuromorphicpaths.core.Push
import com.neuromorphicpaths.core.WalkerState
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sin

/**
 * The push field. Every obstacle carries a surprise that depends on the walker's heading, the
 * surprises add up, and the desired heading is where the sum is lowest.
 *
 * For one obstacle and one candidate heading: how far the object sits off that line (the miss
 * distance) and how soon the walker would draw level with it (the time to contact) together
 * give its surprise. An object well off the line, or far away, contributes almost nothing. An
 * object on the line and close contributes a lot. A turn away from straight ahead costs a little
 * on its own, so with nothing in view the answer is straight ahead.
 *
 * The push each obstacle reports is the slope of its surprise against heading at the walker's
 * current heading. Positive lateral means turning right would lower it. The forward component
 * is the obstacle's surprise with the sign flipped, since an obstacle only ever argues for
 * slowing down.
 *
 * The overall surprise is the sum at the walker's actual heading minus the sum at the desired
 * heading. It is zero when the walker is already doing the best thing, whatever is in view.
 *
 * Searching the headings rather than adding force vectors is what handles the awkward cases.
 * Two objects flanking the line cancel as forces and leave the walker aimed between them even
 * when the gap is too narrow. The search sees that straight ahead costs more than either side.
 * A single object dead ahead has no sideways force at all. The search finds that a turn either
 * way is cheaper, and a fixed tie-break picks the right-hand side when both are equal.
 */
class PushFieldGuidance(
    private val parameters: PushFieldParameters = PushFieldParameters(),
    private val wobble: HeadingWobbleEstimator = HeadingWobbleEstimator(),
) : GuidanceField {

    /** The tolerance turns are charged against right now. Follows the wobble only when the parameters say so. */
    var turnToleranceRadians: Double = parameters.turnToleranceRadians
        private set

    override fun evaluate(obstacles: List<Obstacle>, walker: WalkerState, timestampNanos: Long): Guidance {
        walker.azimuthRadians?.let { wobble.add(timestampNanos, it) }
        turnToleranceRadians = currentTurnTolerance()
        val speed = walker.speedMetersPerSecond ?: parameters.defaultWalkerSpeedMetersPerSecond
        val desiredHeading = lowestCostHeading(obstacles, speed)
        val excess = totalCostBits(obstacles, walker.headingRadians, speed) - totalCostBits(obstacles, desiredHeading, speed)
        val perObstacle = obstacles.map { obstacle ->
            val surprise = obstacleSurpriseBits(obstacle, walker.headingRadians, speed)
            ObstacleSurprise(
                obstacle = obstacle,
                surpriseBits = surprise,
                push = Push(
                    lateral = -slopeAgainstHeading(obstacle, walker.headingRadians, speed),
                    forward = -surprise,
                ),
            )
        }
        return Guidance(
            desiredHeadingRadians = desiredHeading,
            // Rounding in the search can leave the difference a hair below zero.
            overallSurpriseBits = max(0.0, excess),
            perObstacle = perObstacle,
            walkerWobbleRadians = wobble.wobbleRadians,
            turnToleranceRadians = turnToleranceRadians,
        )
    }

    private fun currentTurnTolerance(): Double {
        if (!parameters.turnToleranceFromWobble) return parameters.turnToleranceRadians
        val measured = wobble.wobbleRadians ?: return parameters.turnToleranceRadians
        return (measured * parameters.wobbleToToleranceRatio)
            .coerceIn(parameters.minimumTurnToleranceRadians, parameters.maxHeadingRadians)
    }

    /**
     * Surprise one obstacle carries if the walker holds [headingRadians], in bits.
     *
     * Half the square of a signal-to-noise ratio, converted to bits, the same shape as every
     * other surprise in this project. The ratio here is the reference time over the time to
     * contact, scaled down by how far off the line the object sits.
     */
    fun obstacleSurpriseBits(obstacle: Obstacle, headingRadians: Double, walkerSpeed: Double): Double {
        val relativeBearing = obstacle.bearingRadians - headingRadians
        val alongMeters = obstacle.rangeMeters * cos(relativeBearing)
        // Behind the walker, or level with them: no longer in the way.
        if (alongMeters <= 0.0) return 0.0
        val missMeters = obstacle.rangeMeters * sin(relativeBearing)
        val closingSpeed = obstacle.closingSpeedMetersPerSecond?.takeIf { it > 0.0 } ?: walkerSpeed
        val timeToContact = max(alongMeters / closingSpeed, parameters.minimumTimeToContactSeconds)
        val urgency = parameters.referenceTimeSeconds / timeToContact
        val clearance = parameters.clearanceMeters
        val onTheLine = exp(-(missMeters * missMeters) / (2.0 * clearance * clearance))
        return HALF * urgency * urgency * onTheLine / LN_2
    }

    /** What deviating from straight ahead costs on its own, in bits. */
    fun turnCostBits(headingRadians: Double): Double {
        val ratio = headingRadians / turnToleranceRadians
        return HALF * ratio * ratio / LN_2
    }

    fun totalCostBits(obstacles: List<Obstacle>, headingRadians: Double, walkerSpeed: Double): Double =
        turnCostBits(headingRadians) + obstacles.sumOf { obstacleSurpriseBits(it, headingRadians, walkerSpeed) }

    private fun lowestCostHeading(obstacles: List<Obstacle>, walkerSpeed: Double): Double {
        var bestHeading = 0.0
        var bestCost = totalCostBits(obstacles, 0.0, walkerSpeed)
        // Straight ahead first, then outward in steps, right before left at each step. Only a
        // strictly lower cost replaces the best, so a symmetric scene resolves to the smallest
        // turn, and to the right when both sides tie.
        var offset = parameters.headingStepRadians
        while (offset <= parameters.maxHeadingRadians + HALF_STEP_TOLERANCE * parameters.headingStepRadians) {
            for (heading in doubleArrayOf(offset, -offset)) {
                val cost = totalCostBits(obstacles, heading, walkerSpeed)
                if (cost < bestCost) {
                    bestCost = cost
                    bestHeading = heading
                }
            }
            offset += parameters.headingStepRadians
        }
        return bestHeading
    }

    private fun slopeAgainstHeading(obstacle: Obstacle, headingRadians: Double, walkerSpeed: Double): Double {
        val step = SLOPE_STEP_RADIANS
        val ahead = obstacleSurpriseBits(obstacle, headingRadians + step, walkerSpeed)
        val behind = obstacleSurpriseBits(obstacle, headingRadians - step, walkerSpeed)
        return (ahead - behind) / (2.0 * step)
    }

    private companion object {
        const val HALF = 0.5
        val LN_2 = ln(2.0)
        const val SLOPE_STEP_RADIANS = 1e-4
        const val HALF_STEP_TOLERANCE = 0.5
    }
}
