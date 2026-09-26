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

/** The ground term: a small charge per meter on a surface the walker would rather not be on. */
class PushFieldSurfaceTest {

    private val field = PushFieldGuidance()
    private val walker = WalkerState.ALIGNED_WITH_CAMERA

    private fun barrierAt(forwardMeters: Double, rightMeters: Double) = Obstacle(
        detection = Detection(ObstacleClass.BARRIER, 1.0, NormalizedBox(0.4, 0.4, 0.6, 0.8)),
        bearingRadians = kotlin.math.atan2(rightMeters, forwardMeters),
        rangeMeters = kotlin.math.hypot(forwardMeters, rightMeters),
        closingSpeedMetersPerSecond = null,
    )

    /** Pavement up to [grassFromRightMeters] to the left of the line, grass beyond it. */
    private fun grassOnTheLeft(grassFromRightMeters: Double) = GroundSurfaceMap { _, right ->
        if (right < grassFromRightMeters) SurfaceClass.GRASS else SurfaceClass.PAVEMENT
    }

    @Test
    fun unknownGroundCostsNothingAnywhere() {
        assertEquals(0.0, field.surfaceCostBits(0.0, 1.4, GroundSurfaceMap.UNKNOWN_EVERYWHERE), 0.0)
        assertEquals(0.0, field.surfaceCostBits(Math.toRadians(-40.0), 1.4, GroundSurfaceMap.UNKNOWN_EVERYWHERE), 0.0)
    }

    @Test
    fun grassBesideThePathIsLeftAloneWhenNothingIsInTheWay() {
        val guidance = field.evaluate(emptyList(), walker, 0L, grassOnTheLeft(-0.4))

        assertEquals(0.0, guidance.desiredHeadingRadians, 0.0)
        assertEquals(0.0, guidance.overallSurpriseBits, 0.0)
    }

    @Test
    fun aWallOnTheRightSendsTheWalkerOverTheGrassOnTheLeft() {
        // A wall along the right edge of a narrow pavement, 0.4 m off the line from 1 m to 4 m
        // ahead, and grass from 0.4 m to the left. Staying straight walks into the wall's berth,
        // so the cheapest line crosses the grass.
        val wall = (2..8).map { barrierAt(forwardMeters = it * 0.5, rightMeters = 0.4) }
        val guidance = field.evaluate(wall, walker, 0L, grassOnTheLeft(-0.4))

        assertTrue("should turn left over the grass, was ${Math.toDegrees(guidance.desiredHeadingRadians)}", guidance.desiredHeadingRadians < Math.toRadians(-10.0))
        assertTrue("the grass line should cost something", field.surfaceCostBits(guidance.desiredHeadingRadians, 1.4, grassOnTheLeft(-0.4)) > 0.0)
    }

    @Test
    fun aRoadCostsMoreThanGrassSoTheWalkerPicksTheGrass() {
        // Nothing but the surface term decides: road to the right of the line, grass to the left,
        // and a wall dead ahead so straight is out.
        val ground = GroundSurfaceMap { _, right ->
            when {
                right > 0.2 -> SurfaceClass.ROAD
                right < -0.2 -> SurfaceClass.GRASS
                else -> SurfaceClass.PAVEMENT
            }
        }
        val guidance = field.evaluate(listOf(barrierAt(2.0, 0.0)), walker, 0L, ground)

        assertTrue("should prefer the grass side, was ${Math.toDegrees(guidance.desiredHeadingRadians)}", guidance.desiredHeadingRadians < 0.0)
    }

    @Test
    fun theSurfaceChargeIsPerMeterOfTheLookahead() {
        val allGrass = GroundSurfaceMap { _, _ -> SurfaceClass.GRASS }
        // 1.4 m/s for 2 s is 2.8 m, sampled every 0.25 m from 0.25 to 2.75, eleven steps of 0.025 bits.
        assertEquals(11 * 0.25 * 0.1, field.surfaceCostBits(0.0, 1.4, allGrass), 1e-12)
    }
}
