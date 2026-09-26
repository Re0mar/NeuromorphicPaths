package com.neuromorphicpaths.input

import kotlin.math.sqrt

/** One accelerometer sample: wall-clock time in epoch nanoseconds and the magnitude with gravity taken off, in m/s squared. */
data class AccelerationSample(
    val epochNanos: Long,
    val magnitudeMinusGravity: Double,
)

/**
 * The TotalAcceleration.csv that Sensor Logger writes beside a recording, as samples in time order.
 *
 * The columns are `time` in epoch nanoseconds, `seconds_elapsed`, and `x`, `y`, `z` in m/s
 * squared with gravity included, which is what the phone's own accelerometer reports. The step
 * counter wants the magnitude minus gravity, which needs no knowledge of how the phone was
 * held, so that is what each sample carries. Plain Kotlin, tested on the laptop.
 */
class AccelerationLog(samples: List<AccelerationSample>) {

    val samples: List<AccelerationSample> = samples.sortedBy { it.epochNanos }

    init {
        require(samples.isNotEmpty()) { "Acceleration log has no samples" }
    }

    companion object {
        const val GRAVITY_METERS_PER_SECOND_SQUARED = 9.81
        private const val TIME_COLUMN = "time"
        private val AXIS_COLUMNS = listOf("x", "y", "z")

        /** Parses the CSV text. The header names the columns, so their order does not matter. */
        fun parse(text: String): AccelerationLog {
            val lines = text.lineSequence().map { it.trim() }.filter { it.isNotEmpty() }.iterator()
            require(lines.hasNext()) { "Acceleration log is empty" }
            val header = lines.next().split(',').map { it.trim() }
            fun column(name: String): Int = header.indexOf(name).also { index ->
                require(index >= 0) { "Acceleration log has no '$name' column, header was $header" }
            }
            val timeIndex = column(TIME_COLUMN)
            val axisIndices = AXIS_COLUMNS.map(::column)
            val samples = ArrayList<AccelerationSample>()
            for (line in lines) {
                val fields = line.split(',')
                var squared = 0.0
                for (index in axisIndices) {
                    val value = fields[index].toDouble()
                    squared += value * value
                }
                samples += AccelerationSample(
                    epochNanos = fields[timeIndex].toLong(),
                    magnitudeMinusGravity = sqrt(squared) - GRAVITY_METERS_PER_SECOND_SQUARED,
                )
            }
            return AccelerationLog(samples)
        }
    }
}
