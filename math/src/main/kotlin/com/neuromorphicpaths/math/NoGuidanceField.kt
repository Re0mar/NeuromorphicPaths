package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.GroundSurfaceMap
import com.neuromorphicpaths.core.Guidance
import com.neuromorphicpaths.core.GuidanceField
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.ObstacleSurprise
import com.neuromorphicpaths.core.Push
import com.neuromorphicpaths.core.WalkerState

/**
 * Says straight ahead with zero surprise, whatever is in view.
 *
 * Here so the pipeline runs end to end before the push field exists. It reports every obstacle
 * with a zero push, so the display can already list what the locator placed.
 */
class NoGuidanceField : GuidanceField {
    override fun evaluate(obstacles: List<Obstacle>, walker: WalkerState, timestampNanos: Long, surfaces: GroundSurfaceMap): Guidance =
        Guidance(
            desiredHeadingRadians = 0.0,
            overallSurpriseBits = 0.0,
            perObstacle = obstacles.map { ObstacleSurprise(it, surpriseBits = 0.0, push = Push(0.0, 0.0)) },
        )
}
