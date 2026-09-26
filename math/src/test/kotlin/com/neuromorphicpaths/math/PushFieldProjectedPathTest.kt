package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.GroundSurfaceMap
import com.neuromorphicpaths.core.NormalizedBox
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.ObstacleClass
import com.neuromorphicpaths.core.SurfaceClass
import com.neuromorphicpaths.core.WalkerState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.atan2
import kotlin.math.hypot

/** The projected path and the information measure it fades by. */
class PushFieldProjectedPathTest {

    private val field = PushFieldGuidance()
    private val walker = WalkerState.ALIGNED_WITH_CAMERA

    private fun barrierAt(forwardMeters: Double, rightMeters: Double) = Obstacle(
        detection = Detection(ObstacleClass.BARRIER, 1.0, NormalizedBox(0.4, 0.4, 0.6, 0.8)),
        bearingRadians = atan2(rightMeters, forwardMeters),
        rangeMeters = hypot(forwardMeters, rightMeters),
        closingSpeedMetersPerSecond = null,
    )

    @Test
    fun nothingInViewAddsNoInformationHoweverSpreadTheBeliefIs() {
        val guidance = field.evaluate(emptyList(), walker, 0L)

        assertEquals(0.0, guidance.headingInformationBits!!, 1e-9)
        // The belief is as spread as the prior, several bits, and the information is still zero.
        assertTrue(guidance.headingEntropyBits!! > 6.0)
    }

    @Test
    fun anObstacleDeadAheadAddsInformationAndACorridorAddsMore() {
        // A barrier a meter ahead moves the belief by about half a bit. Two meters ahead it is
        // under a tenth: the dent it makes is shallow against a prior spread over 121 headings.
        val close = field.evaluate(listOf(barrierAt(1.0, 0.0)), walker, 0L).headingInformationBits!!
        val far = field.evaluate(listOf(barrierAt(2.0, 0.0)), walker, 0L).headingInformationBits!!
        assertTrue("expected about half a bit, got $close", close > 0.3)
        assertTrue("the farther barrier should move the belief less, $far against $close", far < close)

        // Walls on both sides leave only straight ahead, which is about two bits against the prior.
        val corridor = (2..8).flatMap { listOf(barrierAt(it * 0.5, 0.4), barrierAt(it * 0.5, -0.4)) }
        val narrowed = field.evaluate(corridor, walker, 0L).headingInformationBits!!
        assertTrue("expected about two bits, got $narrowed", narrowed > 1.5)
    }

    @Test
    fun anEmptySceneProjectsAStraightLine() {
        val path = field.evaluate(emptyList(), walker, 0L).projectedPath

        assertEquals(6, path.size)
        for ((index, point) in path.withIndex()) {
            assertEquals((index + 1) * 0.5, point.forwardMeters, 1e-9)
            assertEquals(0.0, point.rightMeters, 1e-9)
            assertEquals(0.0, point.headingRadians, 0.0)
            assertEquals(0.0, point.informationBits, 1e-9)
        }
    }

    @Test
    fun aWallOnTheRightBendsTheCurveLeft() {
        // A wall along the right edge, 0.4 m off the line from 1 m to 4 m ahead.
        val wall = (2..8).map { barrierAt(forwardMeters = it * 0.5, rightMeters = 0.4) }
        val path = field.evaluate(wall, walker, 0L).projectedPath

        assertTrue("first step should head left, was ${Math.toDegrees(path.first().headingRadians)}", path.first().headingRadians < 0.0)
        assertTrue("the path should end to the left, was ${path.last().rightMeters}", path.last().rightMeters < -0.3)
        // The wall shaped every step, so every step carries information.
        assertTrue(path.all { it.informationBits > 0.0 })
        // Each step is one step length from the last.
        var previousForward = 0.0
        var previousRight = 0.0
        for (point in path) {
            assertEquals(0.5, hypot(point.forwardMeters - previousForward, point.rightMeters - previousRight), 1e-9)
            previousForward = point.forwardMeters
            previousRight = point.rightMeters
        }
    }

    @Test
    fun theFirstStepTakesTheFrameDesiredHeading() {
        val guidance = field.evaluate(listOf(barrierAt(2.0, 0.3)), walker, 0L)

        assertEquals(guidance.desiredHeadingRadians, guidance.projectedPath.first().headingRadians, 0.0)
        assertEquals(guidance.headingInformationBits!!, guidance.projectedPath.first().informationBits, 1e-9)
    }

    @Test
    fun theSurfaceTermIsReadFromWhereTheVirtualWalkerStands() {
        // Road on the right of the line, but only from 1.5 m ahead. From the walker's own
        // position the first step of the surface lookahead sees pavement either way, so the
        // road has to be seen from a point further along for the path to bend away from it.
        val roadAheadRight = GroundSurfaceMap { forward, right ->
            if (forward > 1.5 && right > 0.1) SurfaceClass.ROAD else SurfaceClass.PAVEMENT
        }
        val blocked = listOf(barrierAt(2.5, 0.0))
        val path = field.evaluate(blocked, walker, 0L, roadAheadRight).projectedPath

        assertTrue("the path should end on the pavement side, was ${path.last().rightMeters}", path.last().rightMeters < 0.0)
    }
}
