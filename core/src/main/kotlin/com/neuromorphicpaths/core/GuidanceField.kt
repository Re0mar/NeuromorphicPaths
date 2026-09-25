package com.neuromorphicpaths.core

/**
 * Second math contract. Obstacles and the walker's state become a heading and a surprise.
 *
 * Free of Android, but allowed to keep state between calls, since a running estimate of the
 * walker's own wobble needs history. The timestamp is there for that history.
 */
interface GuidanceField {
    fun evaluate(obstacles: List<Obstacle>, walker: WalkerState, timestampNanos: Long): Guidance
}
