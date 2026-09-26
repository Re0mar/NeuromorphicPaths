package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.GroundSurfaceMap
import com.neuromorphicpaths.core.Guidance
import com.neuromorphicpaths.core.GuidanceField
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.ObstacleSurprise
import com.neuromorphicpaths.core.PathPoint
import com.neuromorphicpaths.core.Push
import com.neuromorphicpaths.core.WalkerState
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
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
 * collision: that the object is in the path (from how far off that line it sits), that contact
 * comes within the class's horizon (from the time to contact), that the object exists at all
 * (the detector's confidence, calibrated so a middling score already counts as certain), and
 * that touching it would matter (one minus the class's contact acceptability). The surprise is minus
 * log2 of the chance of no collision. Independent objects multiply their no-collision chances,
 * so their surprises add exactly.
 *
 * The turn away from straight ahead has a Gaussian prior, so a turn costs a little on its own
 * and with nothing in view the answer is straight ahead. The ground ahead on a heading costs a
 * little per meter for each surface the walker prefers not to be on, so grass is crossed to
 * get away from a wall and left alone otherwise. The cost of a heading is the prior cost plus
 * every object's surprise plus the surface cost, and two things come out of the grid of costs.
 * The lowest one is the desired heading. Two to the minus cost, normalized over the grid, is a
 * posterior over headings, and its entropy says how spread the field's belief is: near zero
 * when one heading is clearly best, near log2 of the grid size when every heading is about as
 * good. Picking the heading with the lowest expected surprise is one step of active inference.
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

    /** The spread of the prior on the heading, which turns are charged against. Fixed by the parameters. */
    val turnToleranceRadians: Double = parameters.turnToleranceRadians

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

    override fun evaluate(obstacles: List<Obstacle>, walker: WalkerState, timestampNanos: Long, surfaces: GroundSurfaceMap): Guidance {
        walker.azimuthRadians?.let { wobble.add(timestampNanos, it) }
        // A measured speed is floored so a standing walker still gets a heading on the first step.
        val speed = walker.speedMetersPerSecond?.coerceAtLeast(parameters.minimumWalkerSpeedMetersPerSecond)
            ?: parameters.defaultWalkerSpeedMetersPerSecond
        val costs = DoubleArray(candidateHeadings.size) { totalCostBits(obstacles, candidateHeadings[it], speed, surfaces) }
        val bestIndex = lowestCostIndex(costs)
        val desiredHeading = candidateHeadings[bestIndex]
        val excess = totalCostBits(obstacles, walker.headingRadians, speed, surfaces) - costs[bestIndex]
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
            headingInformationBits = informationGainBits(costs, costs[bestIndex]),
            lowestSurpriseAheadBits = lowestSurpriseAheadBits(obstacles, speed, surfaces),
            projectedPath = projectPath(obstacles, speed, surfaces),
        )
    }

    /**
     * The surprise of the least surprising heading on offer, with every horizon stretched by
     * [PushFieldParameters.noWayThroughHorizonStretch], in bits.
     *
     * Low when some heading leads clear. High when every heading from hard left to hard right
     * is expected to run into something within the stretched look-ahead, which is what "no way
     * through" means here. A display decides what counts as high.
     */
    fun lowestSurpriseAheadBits(obstacles: List<Obstacle>, walkerSpeed: Double, surfaces: GroundSurfaceMap): Double =
        candidateHeadings.minOf { heading ->
            totalCostBits(obstacles, heading, walkerSpeed, surfaces, horizonStretch = parameters.noWayThroughHorizonStretch)
        }

    /**
     * Where the field would send the walker over the next few meters.
     *
     * The field is rolled forward a step at a time: from the current position the cheapest
     * heading is taken, the virtual walker moves one step along it, every obstacle is
     * re-placed relative to the new position, and the field is asked again. The prior stays
     * centered on the walker's real heading throughout, since it says where the walker wants
     * to go, not where the last step pointed. Each point carries the information the scene
     * added at that step, so a display can fade the path where the scene stops shaping it.
     */
    fun projectPath(obstacles: List<Obstacle>, walkerSpeed: Double, surfaces: GroundSurfaceMap): List<PathPoint> {
        val points = ArrayList<PathPoint>(parameters.pathSteps)
        var forward = 0.0
        var right = 0.0
        var placed = obstacles
        repeat(parameters.pathSteps) {
            val costs = DoubleArray(candidateHeadings.size) { totalCostBits(placed, candidateHeadings[it], walkerSpeed, surfaces, forward, right) }
            val bestIndex = lowestCostIndex(costs)
            val heading = candidateHeadings[bestIndex]
            forward += parameters.pathStepMeters * cos(heading)
            right += parameters.pathStepMeters * sin(heading)
            points += PathPoint(forward, right, heading, informationGainBits(costs, costs[bestIndex]))
            placed = obstacles.map { seenFrom(it, forward, right) }
        }
        return points
    }

    /** The same obstacle as seen from a point ahead of the walker, its bearing still in the walker's frame. */
    private fun seenFrom(obstacle: Obstacle, forwardMeters: Double, rightMeters: Double): Obstacle {
        val deltaForward = obstacle.rangeMeters * cos(obstacle.bearingRadians) - forwardMeters
        val deltaRight = obstacle.rangeMeters * sin(obstacle.bearingRadians) - rightMeters
        return obstacle.copy(bearingRadians = atan2(deltaRight, deltaForward), rangeMeters = hypot(deltaForward, deltaRight))
    }

    /**
     * The chance the walker collides with [obstacle] if they hold [headingRadians].
     *
     * Four factors, each a probability: in the path, contact within the horizon, the object
     * exists, and contact would matter. An object behind the walker, or level with them, has
     * no chance at all. [horizonStretch] multiplies the class's horizon, for the question of
     * whether there is any way through, which looks further ahead than the arrow does.
     */
    fun collisionProbability(obstacle: Obstacle, headingRadians: Double, walkerSpeed: Double, horizonStretch: Double = 1.0): Double {
        val relativeBearing = obstacle.bearingRadians - headingRadians
        val alongMeters = obstacle.rangeMeters * cos(relativeBearing)
        if (alongMeters <= 0.0) return 0.0
        val profile = parameters.profileOf(obstacle.obstacleClass)
        val missMeters = obstacle.rangeMeters * sin(relativeBearing)
        val clearance = profile.clearanceMeters
        val inPath = exp(-(missMeters * missMeters) / (2.0 * clearance * clearance))
        // A measured closing speed may raise the urgency, never lower it. The estimate comes from
        // a range that moves with every degree of pitch, and on the outdoor walk a fifth of
        // standing objects read as not closing at all, so a low or negative reading is noise far
        // more often than a person walking away, and the walker's own speed is the floor.
        val closingSpeed = max(obstacle.closingSpeedMetersPerSecond ?: walkerSpeed, walkerSpeed)
        val timeToContact = max(alongMeters / closingSpeed, parameters.minimumTimeToContactSeconds)
        val urgency = profile.horizonSeconds * horizonStretch / timeToContact
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
    fun obstacleSurpriseBits(obstacle: Obstacle, headingRadians: Double, walkerSpeed: Double, horizonStretch: Double = 1.0): Double {
        val noCollision = 1.0 - collisionProbability(obstacle, headingRadians, walkerSpeed, horizonStretch)
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

    /**
     * What the ground along [headingRadians] costs over the next lookahead, in bits.
     *
     * The line is sampled every step out to the distance the walker covers in the lookahead
     * time, and each step is charged the per-meter cost of the surface it lands on. Ground
     * nobody has classified costs nothing, so without a segmenter this term is zero everywhere.
     */
    fun surfaceCostBits(
        headingRadians: Double,
        walkerSpeed: Double,
        surfaces: GroundSurfaceMap,
        fromForwardMeters: Double = 0.0,
        fromRightMeters: Double = 0.0,
    ): Double {
        val lookaheadMeters = walkerSpeed * parameters.surfaceLookaheadSeconds
        val step = parameters.surfaceStepMeters
        var cost = 0.0
        var along = step
        while (along <= lookaheadMeters) {
            val surface = surfaces.surfaceAt(fromForwardMeters + along * cos(headingRadians), fromRightMeters + along * sin(headingRadians))
            cost += step * parameters.surfaceCostOf(surface)
            along += step
        }
        return cost
    }

    /**
     * The cost of holding [headingRadians], in bits, with the obstacles as already placed
     * relative to the point the cost is asked from. The point only matters to the surface
     * term, which reads the map in the walker's frame. The projected path asks from points
     * ahead of the walker with the obstacles re-placed to match.
     */
    fun totalCostBits(
        obstacles: List<Obstacle>,
        headingRadians: Double,
        walkerSpeed: Double,
        surfaces: GroundSurfaceMap = GroundSurfaceMap.UNKNOWN_EVERYWHERE,
        fromForwardMeters: Double = 0.0,
        fromRightMeters: Double = 0.0,
        horizonStretch: Double = 1.0,
    ): Double =
        turnCostBits(headingRadians) +
            obstacles.sumOf { obstacleSurpriseBits(it, headingRadians, walkerSpeed, horizonStretch) } +
            surfaceCostBits(headingRadians, walkerSpeed, surfaces, fromForwardMeters, fromRightMeters)

    /**
     * How far the scene moved the field's belief away from the prior, in bits: the divergence
     * of the posterior over headings from the prior over the same headings.
     *
     * This is what the literature calls Bayesian surprise, and it is the information-gain
     * term of expected free energy, so it is the one measure here with a name in the course's
     * family. Nothing in view gives exactly zero however spread the belief is, and a scene
     * that pulls the walker hard off their line gives several bits.
     */
    fun informationGainBits(costs: DoubleArray, lowestCost: Double): Double {
        val posterior = DoubleArray(costs.size) { 2.0.pow(-(costs[it] - lowestCost)) }
        val prior = DoubleArray(costs.size) { 2.0.pow(-turnCostBits(candidateHeadings[it])) }
        val posteriorTotal = posterior.sum()
        val priorTotal = prior.sum()
        var divergence = 0.0
        for (index in costs.indices) {
            val p = posterior[index] / posteriorTotal
            if (p > 0.0) divergence += p * log2(p / (prior[index] / priorTotal))
        }
        // Rounding can leave a scene with nothing in it a hair below zero.
        return max(0.0, divergence)
    }

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
