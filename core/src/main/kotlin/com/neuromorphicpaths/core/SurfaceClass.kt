package com.neuromorphicpaths.core

/**
 * What the ground ahead is made of, as far as anything has told the field.
 *
 * Pavement is where a walker prefers to be. The others can be walked on at some cost, which the
 * field charges per meter. Unknown is what everything is until a segmenter says otherwise, and
 * it costs nothing, so a pipeline without one behaves exactly as before.
 */
enum class SurfaceClass {
    PAVEMENT,
    GRASS,
    DIRT,
    ROAD,
    UNKNOWN,
}
