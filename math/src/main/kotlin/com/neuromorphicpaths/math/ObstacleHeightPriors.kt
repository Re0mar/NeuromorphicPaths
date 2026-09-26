package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.ObstacleClass

/**
 * Typical real-world heights, used to estimate range when the bottom of an object is out of
 * frame. Meters. Rough on purpose: a tree can be 3 m or 30 m, so this is the fallback, never
 * the first choice. Null means the class has no usable height, so no range comes from it.
 */
class ObstacleHeightPriors {
    fun heightMeters(obstacleClass: ObstacleClass): Double? = when (obstacleClass) {
        ObstacleClass.TREE -> 5.0
        ObstacleClass.BARRIER -> 1.0
        ObstacleClass.PERSON -> 1.7
        ObstacleClass.ANIMAL -> 0.5
        ObstacleClass.CAR -> 1.5
        ObstacleClass.BIKE -> 1.1
        ObstacleClass.POLE -> 3.0
        ObstacleClass.HOLE -> null
        ObstacleClass.TABLE -> 0.75
        ObstacleClass.CHAIR -> 0.9
        ObstacleClass.TRASH_CAN -> 1.0
        ObstacleClass.UNKNOWN -> null
        // Structure samples arrive from the segmenter with their foot already on the ground, and
        // no detector prompt maps to these, so a box of one never reaches this fallback.
        ObstacleClass.BUILDING -> null
        ObstacleClass.WALL -> null
        ObstacleClass.STAIRS -> null
    }
}
