package com.neuromorphicpaths.core

/**
 * The things the walker is steered around.
 *
 * A detector maps whatever labels its model produces onto this set once, at the boundary, so
 * the rest of the pipeline never sees a model's own label strings. UNKNOWN is for a detection
 * the detector wants to keep but cannot name.
 */
enum class ObstacleClass {
    TREE,
    BARRIER,
    PERSON,
    ANIMAL,
    CAR,
    BIKE,
    POLE,
    HOLE,
    TABLE,
    CHAIR,
    TRASH_CAN,
    UNKNOWN,
}
