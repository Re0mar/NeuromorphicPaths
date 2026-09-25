package com.neuromorphicpaths.input

import com.neuromorphicpaths.core.CameraPose
import java.time.LocalDateTime
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import kotlin.math.PI

/** One orientation sample: wall-clock time in epoch nanoseconds and the three angles in radians. */
data class OrientationSample(
    val epochNanos: Long,
    val pitchRadians: Double,
    val rollRadians: Double,
    val yawRadians: Double,
)

/**
 * The Orientation.csv that Sensor Logger writes beside a recording, as a lookup by wall-clock time.
 *
 * The columns are `time` in epoch nanoseconds, `seconds_elapsed`, a quaternion, then `roll`,
 * `pitch` and `yaw` in radians as Android's getOrientation reports them with no remap. A phone
 * held upright with the back camera facing forward reads a pitch near -90 degrees there, so the
 * camera's pitch down is the logged pitch plus 90 degrees. Yaw feeds only the wobble estimate,
 * which uses its changes and not its reference. Plain Kotlin, so the parsing is tested on the laptop.
 */
class OrientationLog(samples: List<OrientationSample>) {

    val samples: List<OrientationSample> = samples.sortedBy { it.epochNanos }

    init {
        require(samples.isNotEmpty()) { "Orientation log has no samples" }
    }

    val firstEpochNanos: Long get() = samples.first().epochNanos
    val lastEpochNanos: Long get() = samples.last().epochNanos

    /** The sample nearest in time. Before the first or after the last sample, that end sample. */
    fun sampleAt(epochNanos: Long): OrientationSample {
        var low = 0
        var high = samples.size - 1
        while (low < high) {
            val middle = (low + high) / 2
            if (samples[middle].epochNanos < epochNanos) low = middle + 1 else high = middle
        }
        val after = samples[low]
        if (low == 0) return after
        val before = samples[low - 1]
        return if (epochNanos - before.epochNanos <= after.epochNanos - epochNanos) before else after
    }

    /** The camera pose at that moment. Yaw relative to the walker is not in the log, so it is fixed. */
    fun poseAt(epochNanos: Long, heightMeters: Double, yawRadians: Double = 0.0): CameraPose {
        val sample = sampleAt(epochNanos)
        return CameraPose(
            pitchRadians = sample.pitchRadians + UPRIGHT_PHONE_PITCH_OFFSET,
            yawRadians = yawRadians,
            rollRadians = sample.rollRadians,
            heightMeters = heightMeters,
        )
    }

    /** Where the camera pointed over the ground at that moment, for the wobble estimate. */
    fun azimuthAt(epochNanos: Long): Double = sampleAt(epochNanos).yawRadians

    companion object {
        // An upright phone logs a pitch of about -pi/2. Adding pi/2 turns it into camera pitch down.
        private const val UPRIGHT_PHONE_PITCH_OFFSET = PI / 2
        private const val TIME_COLUMN = "time"
        private const val PITCH_COLUMN = "pitch"
        private const val ROLL_COLUMN = "roll"
        private const val YAW_COLUMN = "yaw"

        /** Parses the CSV text. The header names the columns, so their order does not matter. */
        fun parse(text: String): OrientationLog {
            val lines = text.lineSequence().map { it.trim() }.filter { it.isNotEmpty() }.iterator()
            require(lines.hasNext()) { "Orientation log is empty" }
            val header = lines.next().split(',').map { it.trim() }
            fun column(name: String): Int = header.indexOf(name).also { index ->
                require(index >= 0) { "Orientation log has no '$name' column, header was $header" }
            }
            val timeIndex = column(TIME_COLUMN)
            val pitchIndex = column(PITCH_COLUMN)
            val rollIndex = column(ROLL_COLUMN)
            val yawIndex = column(YAW_COLUMN)
            val samples = ArrayList<OrientationSample>()
            for (line in lines) {
                val fields = line.split(',')
                samples += OrientationSample(
                    epochNanos = fields[timeIndex].toLong(),
                    pitchRadians = fields[pitchIndex].toDouble(),
                    rollRadians = fields[rollIndex].toDouble(),
                    yawRadians = fields[yawIndex].toDouble(),
                )
            }
            return OrientationLog(samples)
        }
    }
}

/**
 * When the first frame of a recording was captured, from what the container says about itself.
 *
 * Android's camera stamps the file's date when the take ends, in UTC, as `yyyyMMdd'T'HHmmss.SSS'Z'`.
 * The start is that date minus the duration. The date has whole-second resolution, so the
 * alignment with a sensor log is good to about a second, which one nod at the start of a
 * recording would improve on if anyone needs better.
 */
object RecordingClock {
    private val DATE_FORMAT: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss[.SSS]'Z'")

    fun videoStartEpochMillis(dateMetadata: String, durationMillis: Long): Long {
        val endMillis = LocalDateTime.parse(dateMetadata.trim(), DATE_FORMAT).toInstant(ZoneOffset.UTC).toEpochMilli()
        return endMillis - durationMillis
    }
}
