package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.ObstacleClass

/**
 * What one class of obstacle means to the walker, in three numbers with a physical reading each.
 *
 * Clearance is how far off the walker's line the object has to sit before it is probably not
 * in the path, in meters. Horizon is how far ahead in time contact with it starts to count,
 * in seconds. Contact acceptability is the probability that touching it is fine: brushing a
 * bin is nothing, brushing a car is not.
 *
 * None of these is a weight. Each one enters the collision probability as the thing it names,
 * so a class that needs a wider berth gets a wider Gaussian, not a bigger multiplier.
 */
data class ObstacleProfile(
    val clearanceMeters: Double,
    val horizonSeconds: Double,
    val contactAcceptability: Double,
) {
    init {
        require(clearanceMeters > 0.0) { "clearanceMeters must be positive" }
        require(horizonSeconds > 0.0) { "horizonSeconds must be positive" }
        require(contactAcceptability in 0.0..1.0) { "contactAcceptability must be a probability" }
    }

    companion object {
        /**
         * Defaults per class, argued in meters and seconds rather than found by trial.
         *
         * No clearance is under 0.5 m, since a walker is about that wide and sways, so one
         * clearance off the line is a 61 percent chance of being in the path for anything.
         * Hard and fixed things get no acceptability at all. Things that move themselves out
         * of the way, or that a shin can nudge, get some. A car gets the widest berth and the
         * longest horizon because it is big and may be moving. The three structure classes a
         * segmenter produces are as hard and fixed as a pole, and a flight of stairs is a
         * trip rather than a route, so it gets the same numbers.
         */
        val DEFAULTS: Map<ObstacleClass, ObstacleProfile> = mapOf(
            ObstacleClass.TREE to ObstacleProfile(clearanceMeters = 0.5, horizonSeconds = 2.0, contactAcceptability = 0.0),
            ObstacleClass.BARRIER to ObstacleProfile(clearanceMeters = 0.5, horizonSeconds = 2.0, contactAcceptability = 0.0),
            ObstacleClass.PERSON to ObstacleProfile(clearanceMeters = 0.6, horizonSeconds = 2.0, contactAcceptability = 0.3),
            ObstacleClass.ANIMAL to ObstacleProfile(clearanceMeters = 0.6, horizonSeconds = 2.0, contactAcceptability = 0.2),
            ObstacleClass.CAR to ObstacleProfile(clearanceMeters = 1.0, horizonSeconds = 3.0, contactAcceptability = 0.0),
            ObstacleClass.BIKE to ObstacleProfile(clearanceMeters = 0.5, horizonSeconds = 2.0, contactAcceptability = 0.1),
            ObstacleClass.POLE to ObstacleProfile(clearanceMeters = 0.5, horizonSeconds = 2.0, contactAcceptability = 0.0),
            ObstacleClass.HOLE to ObstacleProfile(clearanceMeters = 0.5, horizonSeconds = 2.0, contactAcceptability = 0.0),
            ObstacleClass.TABLE to ObstacleProfile(clearanceMeters = 0.5, horizonSeconds = 1.5, contactAcceptability = 0.1),
            ObstacleClass.CHAIR to ObstacleProfile(clearanceMeters = 0.5, horizonSeconds = 1.5, contactAcceptability = 0.2),
            ObstacleClass.TRASH_CAN to ObstacleProfile(clearanceMeters = 0.5, horizonSeconds = 1.5, contactAcceptability = 0.5),
            ObstacleClass.UNKNOWN to ObstacleProfile(clearanceMeters = 0.5, horizonSeconds = 2.0, contactAcceptability = 0.0),
            ObstacleClass.BUILDING to ObstacleProfile(clearanceMeters = 0.5, horizonSeconds = 2.0, contactAcceptability = 0.0),
            ObstacleClass.WALL to ObstacleProfile(clearanceMeters = 0.5, horizonSeconds = 2.0, contactAcceptability = 0.0),
            ObstacleClass.STAIRS to ObstacleProfile(clearanceMeters = 0.5, horizonSeconds = 2.0, contactAcceptability = 0.0),
        )
    }
}
