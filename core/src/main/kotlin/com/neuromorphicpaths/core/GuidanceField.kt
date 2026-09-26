package com.neuromorphicpaths.core

/**
 * Second math contract. Obstacles, the ground and the walker's state become a heading and a surprise.
 *
 * Free of Android, but allowed to keep state between calls, since a running estimate of the
 * walker's own wobble needs history. The timestamp is there for that history. The surface map
 * defaults to knowing nothing, so a pipeline without a segmenter calls this with three arguments.
 */
interface GuidanceField {
    fun evaluate(
        obstacles: List<Obstacle>,
        walker: WalkerState,
        timestampNanos: Long,
        surfaces: GroundSurfaceMap = GroundSurfaceMap.UNKNOWN_EVERYWHERE,
    ): Guidance
}
