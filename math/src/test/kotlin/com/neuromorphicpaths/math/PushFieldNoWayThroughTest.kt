package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.NormalizedBox
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.ObstacleClass
import com.neuromorphicpaths.core.WalkerState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.sin

/**
 * The lowest surprise ahead, the number a display reads as "no way through". The scale comes
 * from the laptop re-run of the outdoor walk: a wall straight across the path reads about 7 bits
 * at 2 m with the doubled look-ahead, while no frame of the walk held 4 bits for a second.
 */
class PushFieldNoWayThroughTest {

    private val field = PushFieldGuidance()
    private val walker = WalkerState.ALIGNED_WITH_CAMERA

    private fun wallAt(forwardMeters: Double, rightMeters: Double) = Obstacle(
        detection = Detection(ObstacleClass.WALL, 1.0, NormalizedBox(0.4, 0.4, 0.6, 0.8)),
        bearingRadians = atan2(rightMeters, forwardMeters),
        rangeMeters = hypot(forwardMeters, rightMeters),
        closingSpeedMetersPerSecond = null,
    )

    // One sample per half meter from 4 m left to 4 m right, the way the scene locator thins a wall's feet.
    private fun wallAcross(forwardMeters: Double) = (-8..8).map { wallAt(forwardMeters, it * 0.5) }

    @Test
    fun anEmptySceneHasAWayThroughAtNoSurpriseAtAll() {
        assertEquals(0.0, field.evaluate(emptyList(), walker, 0L).lowestSurpriseAheadBits!!, 1e-9)
    }

    @Test
    fun aWallStraightAcrossThePathTwoMetersOutHasNoWayThrough() {
        val bits = field.evaluate(wallAcross(2.0), walker, 0L).lowestSurpriseAheadBits!!
        assertTrue("expected well over 4 bits, got $bits", bits > 4.0)
    }

    @Test
    fun theSameWallWithAGapInItHasAWayThrough() {
        // A gap from 0.4 to 1.6 m right, a little over a door's width.
        val gapped = wallAcross(2.0).filterNot { it.rangeMeters * sin(it.bearingRadians) in 0.4..1.6 }
        val bits = field.evaluate(gapped, walker, 0L).lowestSurpriseAheadBits!!
        assertTrue("expected under the 3 bit red line, got $bits", bits < 3.0)
    }

    @Test
    fun aSqueezeBetweenTwoPostsIsNotADeadEnd() {
        val posts = listOf(wallAt(2.0, -0.6), wallAt(2.0, 0.6))
        val bits = field.evaluate(posts, walker, 0L).lowestSurpriseAheadBits!!
        assertTrue("expected under the 3 bit red line, got $bits", bits < 3.0)
    }

    @Test
    fun theStretchLooksFurtherAheadWithoutMovingTheArrow() {
        val wall = wallAcross(3.0)
        val stretched = field.evaluate(wall, walker, 0L)
        val unstretched = PushFieldGuidance(PushFieldParameters(noWayThroughHorizonStretch = 1.0)).evaluate(wall, walker, 0L)

        // Three meters out the doubled look-ahead already reads the wall at over 3 bits, where
        // the shipped horizons read it at under 2.
        assertTrue("stretched ${stretched.lowestSurpriseAheadBits}", stretched.lowestSurpriseAheadBits!! > 3.0)
        assertTrue("unstretched ${unstretched.lowestSurpriseAheadBits}", unstretched.lowestSurpriseAheadBits!! < 2.0)
        assertEquals(unstretched.desiredHeadingRadians, stretched.desiredHeadingRadians, 1e-12)
        assertEquals(unstretched.overallSurpriseBits, stretched.overallSurpriseBits, 1e-12)
    }
}
