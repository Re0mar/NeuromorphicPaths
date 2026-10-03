package com.neuromorphicpaths.pixel.wire

/**
 * One planned path as the laptop sends it, `server/docs/arcore_wire_format.md` § What the laptop sends back.
 *
 * Refuses to exist in a shape the display could not draw, so the refusal happens here with the
 * field named, and [PathDecoder] turns it into a dropped message rather than a crash.
 *
 * @property timestampSeconds the depth frame the path was planned for, this phone's clock handed back
 * @property timesSeconds how far into the future each offset is
 * @property lateralOffsetsMeters where to be at that time, sideways from straight ahead, positive right
 * @property firstHeadingRadians where the path is heading over the planner's lookahead, positive right
 * @property alarm something in the walker's way is close at walking pace
 * @property cumulativeCostBits total cost of the chosen path, for display and logging
 */
data class PathMessage(
    val timestampSeconds: Double,
    val timesSeconds: DoubleArray,
    val lateralOffsetsMeters: DoubleArray,
    val firstHeadingRadians: Double,
    val alarm: Boolean,
    val cumulativeCostBits: Double,
) {
    init {
        require(timesSeconds.isNotEmpty()) { "times_seconds must have at least one entry" }
        require(timesSeconds.size == lateralOffsetsMeters.size) {
            "times_seconds has ${timesSeconds.size} entries and lateral_offsets_meters has ${lateralOffsetsMeters.size}"
        }
        require(timestampSeconds.isFinite()) { "timestamp_seconds must be finite, got $timestampSeconds" }
        require(firstHeadingRadians.isFinite()) { "first_heading_radians must be finite, got $firstHeadingRadians" }
        require(cumulativeCostBits.isFinite()) { "cumulative_cost_bits must be finite, got $cumulativeCostBits" }
        for (index in timesSeconds.indices) {
            require(timesSeconds[index].isFinite()) { "times_seconds[$index] must be finite, got ${timesSeconds[index]}" }
            require(lateralOffsetsMeters[index].isFinite()) { "lateral_offsets_meters[$index] must be finite, got ${lateralOffsetsMeters[index]}" }
        }
    }

    // A data class compares arrays by reference. Two paths with the same numbers are the same path.
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is PathMessage) return false
        return timestampSeconds == other.timestampSeconds &&
            timesSeconds.contentEquals(other.timesSeconds) &&
            lateralOffsetsMeters.contentEquals(other.lateralOffsetsMeters) &&
            firstHeadingRadians == other.firstHeadingRadians &&
            alarm == other.alarm &&
            cumulativeCostBits == other.cumulativeCostBits
    }

    override fun hashCode(): Int {
        var result = timestampSeconds.hashCode()
        result = 31 * result + timesSeconds.contentHashCode()
        result = 31 * result + lateralOffsetsMeters.contentHashCode()
        result = 31 * result + firstHeadingRadians.hashCode()
        result = 31 * result + alarm.hashCode()
        result = 31 * result + cumulativeCostBits.hashCode()
        return result
    }
}
