package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.NormalizedBox
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.ObstacleClass
import com.neuromorphicpaths.core.WalkerState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log2

/** The probability form: each term is a probability first, the surprise follows, and the numbers behave. */
class PushFieldProbabilityTest {

    private val field = PushFieldGuidance()
    private val walker = WalkerState.ALIGNED_WITH_CAMERA
    private val walkerSpeed = 1.4

    private fun obstacleAt(
        bearingDegrees: Double,
        rangeMeters: Double,
        obstacleClass: ObstacleClass = ObstacleClass.TREE,
        confidence: Double = 1.0,
    ) = Obstacle(
        detection = Detection(obstacleClass, confidence = confidence, box = NormalizedBox(0.4, 0.4, 0.6, 0.8)),
        bearingRadians = Math.toRadians(bearingDegrees),
        rangeMeters = rangeMeters,
        closingSpeedMetersPerSecond = null,
    )

    /** Bearing and range that put an object [alongMeters] ahead and [missMeters] to the right of the line. */
    private fun obstacleOffset(alongMeters: Double, missMeters: Double, obstacleClass: ObstacleClass = ObstacleClass.TREE) =
        obstacleAt(
            bearingDegrees = Math.toDegrees(kotlin.math.atan2(missMeters, alongMeters)),
            rangeMeters = kotlin.math.hypot(alongMeters, missMeters),
            obstacleClass = obstacleClass,
        )

    @Test
    fun collisionProbabilityIsTheProductOfItsFourFactors() {
        // A person 2.8 m dead ahead at 1.4 m/s: 2 s to contact, the person's horizon. In the
        // path with probability 1, contact soon with 1 - exp(-1/2), exists with 0.3 / 0.5
        // after calibration, and matters with 1 - 0.3.
        val person = obstacleAt(0.0, 2.8, ObstacleClass.PERSON, confidence = 0.3)

        val probability = field.collisionProbability(person, headingRadians = 0.0, walkerSpeed = walkerSpeed)

        val expected = 1.0 * (1.0 - exp(-0.5)) * 0.6 * (1.0 - 0.3)
        assertEquals(expected, probability, 1e-12)
        assertEquals(-log2(1.0 - expected), field.obstacleSurpriseBits(person, 0.0, walkerSpeed), 1e-12)
    }

    @Test
    fun surprisesOfIndependentObjectsAddExactly() {
        val left = obstacleAt(-10.0, 2.0)
        val right = obstacleAt(12.0, 2.5, ObstacleClass.BIKE, confidence = 0.7)

        val total = field.totalCostBits(listOf(left, right), headingRadians = 0.0, walkerSpeed = walkerSpeed)

        val jointMiss = (1.0 - field.collisionProbability(left, 0.0, walkerSpeed)) * (1.0 - field.collisionProbability(right, 0.0, walkerSpeed))
        assertEquals(-log2(jointMiss), total, 1e-12)
    }

    @Test
    fun aCloseObjectAlreadyBeingClearedIsUnderOneBit() {
        // 0.8 m to the side and 0.3 m ahead, an arm's length pass. The old product form scored
        // this at about 22 bits and would have swerved from something already cleared.
        val pass = obstacleOffset(alongMeters = 0.3, missMeters = 0.8)

        val surprise = field.obstacleSurpriseBits(pass, headingRadians = 0.0, walkerSpeed = walkerSpeed)

        assertTrue("was $surprise bits", surprise < 1.0)
        assertTrue("still worth something", surprise > 0.1)
    }

    @Test
    fun aCarOutweighsABinAtTheSameSpot() {
        val car = field.obstacleSurpriseBits(obstacleAt(5.0, 2.5, ObstacleClass.CAR), 0.0, walkerSpeed)
        val bin = field.obstacleSurpriseBits(obstacleAt(5.0, 2.5, ObstacleClass.TRASH_CAN), 0.0, walkerSpeed)

        assertTrue("car $car should beat bin $bin", car > 2.0 * bin)
    }

    @Test
    fun lessConfidenceMeansLessSurpriseUntilTheScoreSaturates() {
        val certain = field.obstacleSurpriseBits(obstacleAt(0.0, 2.0, confidence = 1.0), 0.0, walkerSpeed)
        val middling = field.obstacleSurpriseBits(obstacleAt(0.0, 2.0, confidence = 0.5), 0.0, walkerSpeed)
        val unsure = field.obstacleSurpriseBits(obstacleAt(0.0, 2.0, confidence = 0.3), 0.0, walkerSpeed)

        // From the saturation point up the detector's score no longer matters.
        assertEquals(certain, middling, 1e-12)
        assertTrue(middling > unsure)
        assertTrue(unsure > 0.0)
    }

    @Test
    fun aCertainCollisionStaysFinite() {
        // Within the time floor, on the line, certain, and no acceptability: the probability of
        // collision rounds to one, and the surprise has to stay a number the search can add.
        val surprise = field.obstacleSurpriseBits(obstacleAt(0.0, 0.05), 0.0, walkerSpeed)

        assertTrue(surprise.isFinite())
        assertTrue(surprise > 20.0)
    }

    @Test
    fun turnCostIsMinusLogOfAGaussianPrior() {
        // At one tolerance from straight ahead the prior is down by exp(-1/2), which is half a
        // nat, or 0.72 bits.
        val cost = field.turnCostBits(Math.toRadians(30.0))

        assertEquals(0.5 / ln(2.0), cost, 1e-12)
    }

    @Test
    fun emptySceneHasThePriorsEntropyAndAWallOfObjectsSharpensIt() {
        val open = field.evaluate(emptyList(), walker, 0L).headingEntropyBits!!
        // A row of trees 3 m out from 40 degrees left to 40 degrees right, with one gap about
        // 1.5 m wide between 4 and 36 degrees, three clearances, so the gap beats going round.
        val row = (-40..40 step 4).filter { it !in 8..32 }.map { obstacleAt(it.toDouble(), 3.0) }
        val gap = field.evaluate(row, walker, 0L)

        assertTrue("entropy is bounded by the grid, was $open", open > 0.0 && open < log2(121.0))
        assertTrue("belief should sharpen on the gap, was ${gap.headingEntropyBits}", gap.headingEntropyBits!! < open)
        assertTrue("the heading should be the gap, was ${Math.toDegrees(gap.desiredHeadingRadians)}", Math.toDegrees(gap.desiredHeadingRadians) in 8.0..32.0)
    }

    @Test
    fun theAlertIsNotThePosteriorAtTheWalkersHeading() {
        // With nothing in view the posterior puts about a percent on straight ahead, which
        // would read as several bits. The alert is the cost gap, which is zero.
        val open = field.evaluate(emptyList(), walker, 0L)

        assertEquals(0.0, open.overallSurpriseBits, 0.0)
        assertTrue(open.headingEntropyBits!! > 3.0)
    }

    @Test
    fun anApproachingPersonIsMoreSurprisingThanAStandingOne() {
        val standing = obstacleAt(0.0, 4.0, ObstacleClass.PERSON).copy(closingSpeedMetersPerSecond = walkerSpeed)
        val approaching = standing.copy(closingSpeedMetersPerSecond = 2.0 * walkerSpeed)
        val unmeasured = standing.copy(closingSpeedMetersPerSecond = null)

        val standingBits = field.obstacleSurpriseBits(standing, 0.0, walkerSpeed)
        val approachingBits = field.obstacleSurpriseBits(approaching, 0.0, walkerSpeed)

        assertTrue("approaching $approachingBits should beat standing $standingBits", approachingBits > 2.0 * standingBits)
        // Nothing measured means the walker's own speed, which is what a standing object closes at.
        assertEquals(standingBits, field.obstacleSurpriseBits(unmeasured, 0.0, walkerSpeed), 1e-12)
    }

    @Test
    fun anObjectThatIsNotClosingCannotBeHit() {
        val walkingAway = obstacleAt(0.0, 2.0, ObstacleClass.PERSON).copy(closingSpeedMetersPerSecond = -0.5)
        val level = obstacleAt(0.0, 2.0, ObstacleClass.PERSON).copy(closingSpeedMetersPerSecond = 0.0)

        assertEquals(0.0, field.obstacleSurpriseBits(walkingAway, 0.0, walkerSpeed), 0.0)
        assertEquals(0.0, field.obstacleSurpriseBits(level, 0.0, walkerSpeed), 0.0)
    }

    @Test
    fun everyObstacleClassHasAProfile() {
        for (obstacleClass in ObstacleClass.entries) {
            val profile = PushFieldParameters().profileOf(obstacleClass)
            assertTrue(profile.clearanceMeters > 0.0)
        }
    }
}
