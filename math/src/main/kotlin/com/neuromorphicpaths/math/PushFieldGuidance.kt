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
import kotlin.math.log2
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin

/**
 * The push field. Every obstacle carries a probability of being hit that depends on the
 * walker's heading, the surprises those probabilities imply add up, and the desired heading
 * is where the sum is lowest.
 *
 * For one obstacle and one candidate heading, four probabilities multiply into the chance of a
 * collision: that the object is in the path, from how far off that line it sits; that contact
 * comes within the class's horizon, from the time to contact; that the object exists at all,
 * the detector's confidence calibrated so a middling score already counts as certain; and that
 * touching it would matter, one minus the class's contact acceptability. The surprise is
 * minus log2 of the chance of no collision. Independent objects multiply their no-collision
 * chances, so their surprises add exactly.
 *
 * The turn away from straight ahead has a Gaussian prior, so a turn costs a little on its own
 * and with nothing in view the answer is straight ahead. The cost of a heading is the prior
 * cost plus every object's surprise, and two things come out of the grid of costs. The lowest
 * one is the desired heading. Two to the minus cost, normalized over the grid, is a posterior
 * over headings, and its entropy says how spread the field's belief is: near zero when one
 * heading is clearly best, near log2 of the grid size when every heading is about as good.
 * Picking the heading with the lowest expected surprise is one step of active inference.
 *
 * The overall surprise is the cost at the walker's actual heading minus the cost at the desired
 * heading. It is zero when the walker is already doing the best thing, whatever is in view. It
 * is not minus log of the posterior at the walker's heading: in an empty scene that would read
 * several bits, since the prior spreads over a hundred-odd candidates, with nothing there.
 *
 * The push each obstacle reports is the slope of its surprise against heading at the walker's
 * current heading. Positive lateral means turning right would lower it. The forward component
 * is the obstacle's surprise with the sign flipped, since an obstacle only ever argues for
 * slowing down.
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

    // Straight ahead first, then outward in steps, right before left at each step, so a scan
    // that only accepts a strictly lower cost resolves a symmetric scene to the smallest turn,
    // and to the right when both sides tie.
    private val candidateHeadings: DoubleArray = buildList {
        add(0.0)
        var offset = parameters.headingStepRadians
        while (offset <= parameters.maxHeadingRadians + HALF_STEP_TOLERANCE * parameters.headingStepRadians) {
            add(offset)
            add(-offset)
            offset += parameters.headingStepRadians
        }
    }.toDoubleArray()

    override fun evaluate(obstacles: List<Obstacle>, walker: WalkerState, timestampNanos: Long): Guidance {
        walker.azimuthRadians?.let { wobble.add(timestampNanos, it) }
        turnToleranceRadians = currentTurnTolerance()
        val speed = walker.speedMetersPerSecond ?: parameters.defaultWalkerSpeedMetersPerSecond
        val costs = DoubleArray(candidateHeadings.size) { totalCostBits(obstacles, candidateHeadings[it], speed) }
        val bestIndex = lowestCostIndex(costs)
        val desiredHeading = candidateHeadings[bestIndex]
        val excess = totalCostBits(obstacles, walker.headingRadians, speed) - costs[bestIndex]
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
            headingEntropyBits = posteriorEntropyBits(costs, costs[bestIndex]),
        )
    }

    private fun currentTurnTolerance(): Double {
        if (!parameters.turnToleranceFromWobble) return parameters.turnToleranceRadians
        val measured = wobble.wobbleRadians ?: return parameters.turnToleranceRadians
        return (measured * parameters.wobbleToToleranceRatio)
            .coerceIn(parameters.minimumTurnToleranceRadians, parameters.maxHeadingRadians)
    }

    /**
     * The chance the walker collides with [obstacle] if they hold [headingRadians].
     *
     * Four factors, each a probability: in the path, contact within the horizon, the object
     * exists, and contact would matter. An object behind the walker, or level with them, has
     * no chance at all.
     */
    fun collisionProbability(obstacle: Obstacle, headingRadians: Double, walkerSpeed: Double): Double {
        val relativeBearing = obstacle.bearingRadians - headingRadians
        val alongMeters = obstacle.rangeMeters * cos(relativeBearing)
        if (alongMeters <= 0.0) return 0.0
        val profile = parameters.profileOf(obstacle.obstacleClass)
        val missMeters = obstacle.rangeMeters * sin(relativeBearing)
        val clearance = profile.clearanceMeters
        val inPath = exp(-(missMeters * missMeters) / (2.0 * clearance * clearance))
        // A measured closing speed at or below zero means the gap is not shrinking, so contact
        // never comes. Only an unmeasured one falls back to the walker's own speed.
        val closingSpeed = obstacle.closingSpeedMetersPerSecond ?: walkerSpeed
        if (closingSpeed <= 0.0) return 0.0
        val timeToContact = max(alongMeters / closingSpeed, parameters.minimumTimeToContactSeconds)
        val urgency = profile.horizonSeconds / timeToContact
        // The complement of this term is a Gaussian in urgency, so for an object on the line
        // with certain existence and no acceptability the surprise is half urgency squared in
        // nats, the same closed form as the time-to-contact surprise elsewhere in the project.
        val contactSoon = 1.0 - exp(-HALF * urgency * urgency)
        // The detector's score, calibrated: certain from the saturation point up, linear below.
        val exists = (obstacle.detection.confidence / parameters.confidenceForCertainty).coerceIn(0.0, 1.0)
        val matters = 1.0 - profile.contactAcceptability
        return inPath * contactSoon * exists * matters
    }

    /** Surprise one obstacle carries if the walker holds [headingRadians], in bits: minus log2 of the chance of missing it. */
    fun obstacleSurpriseBits(obstacle: Obstacle, headingRadians: Double, walkerSpeed: Double): Double {
        val noCollision = 1.0 - collisionProbability(obstacle, headingRadians, walkerSpeed)
        // A collision probability rounded to exactly one would give infinite bits, which no
        // sum or search can use, so the floor turns it into a very large finite number. The
        // outer max turns the minus zero of a certain miss into a plain zero for the log.
        return max(0.0, -log2(max(noCollision, MINIMUM_NO_COLLISION_PROBABILITY)))
    }

    /** What deviating from straight ahead costs on its own, in bits: minus log2 of a Gaussian prior, relative to straight ahead. */
    fun turnCostBits(headingRadians: Double): Double {
        val ratio = headingRadians / turnToleranceRadians
        return HALF * ratio * ratio / LN_2
    }

    fun totalCostBits(obstacles: List<Obstacle>, headingRadians: Double, walkerSpeed: Double): Double =
        turnCostBits(headingRadians) + obstacles.sumOf { obstacleSurpriseBits(it, headingRadians, walkerSpeed) }

    /**
     * Entropy of the posterior over candidate headings, in bits.
     *
     * The posterior at each heading is two to the minus its cost, normalized over the grid.
     * Shifting every cost by the lowest one first keeps the powers inside double range.
     */
    fun posteriorEntropyBits(costs: DoubleArray, lowestCost: Double): Double {
        val weights = DoubleArray(costs.size) { 2.0.pow(-(costs[it] - lowestCost)) }
        val total = weights.sum()
        var entropy = 0.0
        for (weight in weights) {
            val probability = weight / total
            if (probability > 0.0) entropy -= probability * log2(probability)
        }
        return entropy
    }

    private fun lowestCostIndex(costs: DoubleArray): Int {
        var bestIndex = 0
        for (index in 1 until costs.size) {
            if (costs[index] < costs[bestIndex]) bestIndex = index
        }
        return bestIndex
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
        const val MINIMUM_NO_COLLISION_PROBABILITY = 1e-300
    }
}
