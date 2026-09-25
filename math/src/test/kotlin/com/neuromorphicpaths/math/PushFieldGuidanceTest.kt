package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.NormalizedBox
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.ObstacleClass
import com.neuromorphicpaths.core.WalkerState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class PushFieldGuidanceTest {

    private val field = PushFieldGuidance()
    private val walker = WalkerState.ALIGNED_WITH_CAMERA

    // A tree is the plainest obstacle: 0.5 m clearance, 2 s horizon, no acceptability of
    // contact. With full confidence the closed forms below come out exact.
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

    private fun degrees(radians: Double) = Math.toDegrees(radians)

    @Test
    fun nothingInViewMeansStraightAheadAndNoSurprise() {
        val guidance = field.evaluate(emptyList(), walker, timestampNanos = 0L)

        assertEquals(0.0, guidance.desiredHeadingRadians, 0.0)
        assertEquals(0.0, guidance.overallSurpriseBits, 0.0)
        assertTrue(guidance.perObstacle.isEmpty())
    }

    @Test
    fun objectBehindTheWalkerIsIgnored() {
        val guidance = field.evaluate(listOf(obstacleAt(170.0, 1.0)), walker, 0L)

        assertEquals(0.0, guidance.desiredHeadingRadians, 0.0)
        assertEquals(0.0, guidance.overallSurpriseBits, 1e-12)
        assertEquals(0.0, guidance.perObstacle.single().surpriseBits, 0.0)
    }

    @Test
    fun objectOnTheRightPushesLeft() {
        val guidance = field.evaluate(listOf(obstacleAt(15.0, 2.5)), walker, 0L)

        assertTrue("desired heading should turn left, was ${degrees(guidance.desiredHeadingRadians)}", guidance.desiredHeadingRadians < 0.0)
        assertTrue("lateral push should point left", guidance.perObstacle.single().push.lateral < 0.0)
        assertTrue("forward push should be a brake", guidance.perObstacle.single().push.forward < 0.0)
    }

    @Test
    fun objectDeadAheadBreaksTheTieToTheRight() {
        val guidance = field.evaluate(listOf(obstacleAt(0.0, 3.0)), walker, 0L)

        assertTrue("should turn, was ${degrees(guidance.desiredHeadingRadians)}", abs(guidance.desiredHeadingRadians) > Math.toRadians(5.0))
        assertTrue("should turn right on a tie", guidance.desiredHeadingRadians > 0.0)
        // 3 m at 1.4 m/s is 2.1 s, just past the horizon, so the object alone is about
        // 0.6 bits. The best turn still costs something, so the excess is smaller than that.
        assertTrue("the object itself should be surprising", guidance.perObstacle.single().surpriseBits > 0.5)
        assertTrue("heading straight at it should cost more than turning", guidance.overallSurpriseBits > 0.2)
    }

    @Test
    fun wideGapBetweenTwoObjectsIsWalkedThrough() {
        val guidance = field.evaluate(listOf(obstacleAt(-30.0, 3.0), obstacleAt(30.0, 3.0)), walker, 0L)

        // Each sits 1.5 m off the line, three clearances out, so the middle is fine.
        assertEquals(0.0, guidance.desiredHeadingRadians, 0.0)
        assertTrue(guidance.overallSurpriseBits < 0.05)
    }

    @Test
    fun narrowGapBetweenTwoObjectsIsNotWalkedThrough() {
        val guidance = field.evaluate(listOf(obstacleAt(-8.0, 2.0), obstacleAt(8.0, 2.0)), walker, 0L)

        // Each sits under 0.3 m off the line. As forces they cancel. The search does not.
        assertTrue("should deflect, was ${degrees(guidance.desiredHeadingRadians)}", abs(guidance.desiredHeadingRadians) > Math.toRadians(10.0))
        assertTrue(guidance.overallSurpriseBits > 0.5)
    }

    @Test
    fun closerIsMoreSurprisingThanFarther() {
        val far = field.obstacleSurpriseBits(obstacleAt(0.0, 6.0), headingRadians = 0.0, walkerSpeed = 1.4)
        val near = field.obstacleSurpriseBits(obstacleAt(0.0, 1.5), headingRadians = 0.0, walkerSpeed = 1.4)

        assertTrue(near > far)
    }

    @Test
    fun objectAtTheHorizonDeadAheadIsAboutSevenTenthsOfABit() {
        // Range 2.8 m at 1.4 m/s is 2 s, the tree's horizon, so the urgency is 1 and the
        // surprise is half a nat, which is 0.72 bits.
        val surprise = field.obstacleSurpriseBits(obstacleAt(0.0, 2.8), headingRadians = 0.0, walkerSpeed = 1.4)

        assertEquals(0.5 / Math.log(2.0), surprise, 1e-9)
    }

    @Test
    fun walkerAlreadyOnTheDesiredHeadingHasNoOverallSurprise() {
        val obstacles = listOf(obstacleAt(10.0, 2.0))
        val first = field.evaluate(obstacles, walker, 0L)
        val aligned = WalkerState(headingRadians = first.desiredHeadingRadians, speedMetersPerSecond = null)

        val second = field.evaluate(obstacles, aligned, 0L)

        assertEquals(first.desiredHeadingRadians, second.desiredHeadingRadians, 0.0)
        assertEquals(0.0, second.overallSurpriseBits, 1e-12)
    }
}
