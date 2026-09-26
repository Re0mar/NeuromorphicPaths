package com.neuromorphicpaths.core

/**
 * The things the walker is steered around.
 *
 * A detector maps whatever labels its model produces onto this set once, at the boundary, so
 * the rest of the pipeline never sees a model's own label strings. UNKNOWN is for a detection
 * the detector wants to keep but cannot name.
 *
 * The last three come from a segmenter rather than a detector: a building face, a wall or a
 * flight of stairs arrives as a region, and its foot along the ground is sampled into
 * obstacles of these classes. The detector's own "brick wall" and "stairs" prompts still map to
 * BARRIER, so the log can tell the two sources apart.
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
    BUILDING,
    WALL,
    STAIRS,
}
