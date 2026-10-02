package com.neuromorphicpaths.pixel.ar

/**
 * What the floor choice needs to know about one tracked upward-facing plane.
 *
 * ARCore's Plane cannot be built off the device, so the choice runs over this summary and the
 * renderer maps the winner back to the plane it came from.
 */
data class FloorCandidate(
    val extentXMeters: Float,
    val extentZMeters: Float,
    val centerHeightMeters: Float,
)

/** Picks which tracked plane to send the laptop as the floor. */
object FloorChoice {
    /**
     * The index of the candidate to send as the floor, or null for an empty list.
     *
     * The largest plane by extent, not the lowest. On the first walk ARCore also tracked a plane
     * about a meter below the real floor, the lowest-plane rule sent that one on every frame, and
     * the laptop read the real floor as a wall of obstacles. The floor is the largest horizontal
     * surface in view and a false plane under it stays small. A tie in area goes to the lower
     * candidate, so a tie is not a coin flip.
     */
    fun chooseFloorIndex(candidates: List<FloorCandidate>): Int? {
        var bestIndex: Int? = null
        var bestArea = Float.NEGATIVE_INFINITY
        var bestHeight = Float.POSITIVE_INFINITY
        for (index in candidates.indices) {
            val candidate = candidates[index]
            val area = candidate.extentXMeters * candidate.extentZMeters
            val larger = area > bestArea
            val sameAreaButLower = area == bestArea && candidate.centerHeightMeters < bestHeight
            if (larger || sameAreaButLower) {
                bestIndex = index
                bestArea = area
                bestHeight = candidate.centerHeightMeters
            }
        }
        return bestIndex
    }
}
