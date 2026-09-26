package com.neuromorphicpaths.core

/**
 * The ground around the walker, queried by position: how far ahead and how far to the right, in
 * meters along the ground, walker's frame.
 *
 * A segmenter will produce one of these per frame from a class map projected through the ground
 * plane. Until it exists, [UNKNOWN_EVERYWHERE] stands in and the field charges nothing for
 * surfaces. Tests build one from a rule.
 */
fun interface GroundSurfaceMap {
    fun surfaceAt(forwardMeters: Double, rightMeters: Double): SurfaceClass

    companion object {
        /** No knowledge of the ground at all, which the field treats as free to walk on. */
        val UNKNOWN_EVERYWHERE: GroundSurfaceMap = GroundSurfaceMap { _, _ -> SurfaceClass.UNKNOWN }
    }
}
