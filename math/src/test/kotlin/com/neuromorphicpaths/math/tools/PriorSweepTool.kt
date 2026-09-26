package com.neuromorphicpaths.math.tools

import com.neuromorphicpaths.core.WalkerState
import com.neuromorphicpaths.math.HeadingWobbleEstimator
import com.neuromorphicpaths.math.PushFieldGuidance
import com.neuromorphicpaths.math.PushFieldParameters
import org.junit.Test
import java.io.File
import java.util.Locale
import kotlin.math.abs

/**
 * Re-runs the shipped field over a logged replay with different turn priors and scores each
 * against what the walker really did, read from the orientation log's yaw.
 *
 * Not a test. It runs through the `priorSweep` Gradle task, which passes the file paths in as
 * system properties, and the ordinary test task leaves it out. A walker's real turn is a yaw
 * change of at least [TURN_DEGREES] over the next [LEAD_SECONDS], away from path corners, which
 * are the long sustained swings. The field is asked whether it called that turn, in that
 * direction, in the second before it, and how often it called a turn nobody made.
 */
class PriorSweepTool {

    private class YawLog(private val epochNanos: LongArray, private val unwrappedYawRadians: DoubleArray) {
        fun yawAt(nanos: Long): Double {
            var low = 0
            var high = epochNanos.size - 1
            while (low < high) {
                val middle = (low + high) / 2
                if (epochNanos[middle] < nanos) low = middle + 1 else high = middle
            }
            return unwrappedYawRadians[low]
        }

        companion object {
            fun read(file: File): Pair<YawLog, Long> {
                val lines = file.readLines().filter { it.isNotBlank() }
                val header = lines.first().split(',').map { it.trim() }
                val timeIndex = header.indexOf("time")
                val elapsedIndex = header.indexOf("seconds_elapsed")
                val yawIndex = header.indexOf("yaw")
                require(timeIndex >= 0 && elapsedIndex >= 0 && yawIndex >= 0) { "orientation log header was $header" }
                val times = LongArray(lines.size - 1)
                val yaws = DoubleArray(lines.size - 1)
                var previous = 0.0
                var offset = 0.0
                for ((index, line) in lines.drop(1).withIndex()) {
                    val fields = line.split(',')
                    times[index] = fields[timeIndex].toLong()
                    val raw = fields[yawIndex].toDouble()
                    if (index > 0) {
                        val step = HeadingWobbleEstimator.wrapToPi(raw - previous)
                        offset += step - (raw - previous)
                    }
                    previous = raw
                    yaws[index] = raw + offset
                }
                val firstFields = lines[1].split(',')
                val loggerEpochNanos = firstFields[timeIndex].toLong() - (firstFields[elapsedIndex].toDouble() * NANOS_PER_SECOND).toLong()
                return YawLog(times, yaws) to loggerEpochNanos
            }
        }
    }

    private data class Setting(val name: String, val toleranceFor: (ReplayFrame) -> Double)

    private data class Score(
        val name: String,
        val alerts: Int,
        val hits: Int,
        val falseAlarms: Int,
        val realTurns: Int,
        val turnsCalled: Int,
        val meanAbsHeadingDegrees: Double,
        val framesAbove03Bits: Int,
        val medianEntropyBits: Double,
        val minEntropyBits: Double,
        val conePeakBits: Double,
        val coneHeadingDegrees: Double,
        val rackPeakBits: Double,
        val rackHeadingDegrees: Double,
    )

    @Test
    fun run() {
        val replayPath = System.getProperty("replay.log") ?: return
        val orientationPath = System.getProperty("orientation.csv") ?: error("orientation.csv not set")
        val outDir = File(System.getProperty("out.dir") ?: error("out.dir not set")).apply { mkdirs() }
        val videoOffsetSeconds = System.getProperty("video.offset.seconds")?.toDouble() ?: DEFAULT_VIDEO_OFFSET_SECONDS

        val frames = ReplayLog.read(File(replayPath))
        require(frames.isNotEmpty()) { "no replay frames in $replayPath" }
        val (yaw, loggerEpochNanos) = YawLog.read(File(orientationPath))
        val videoStartNanos = loggerEpochNanos + (videoOffsetSeconds * NANOS_PER_SECOND).toLong()
        fun yawAtVideo(seconds: Double): Double = yaw.yawAt(videoStartNanos + (seconds * NANOS_PER_SECOND).toLong())

        // What the walker did after each frame, and whether that frame sits in a corner.
        val turnAhead = DoubleArray(frames.size) { index -> yawAtVideo(frames[index].videoSeconds + LEAD_SECONDS) - yawAtVideo(frames[index].videoSeconds) }
        val inCorner = BooleanArray(frames.size) { index ->
            abs(yawAtVideo(frames[index].videoSeconds + CORNER_SECONDS) - yawAtVideo(frames[index].videoSeconds - 1.0)) >= Math.toRadians(CORNER_DEGREES)
        }
        File(outDir, "walker_turns.csv").printWriter().use { out ->
            out.println("video_seconds,yaw_change_next_2s_deg,corner")
            for ((index, frame) in frames.withIndex()) {
                out.println(String.format(Locale.US, "%.2f,%.1f,%d", frame.videoSeconds, Math.toDegrees(turnAhead[index]), if (inCorner[index]) 1 else 0))
            }
        }
        println(String.format(Locale.US, "yaw change over the cone, 183 to 186 s: %+.0f deg (the walker went left)", Math.toDegrees(yawAtVideo(186.0) - yawAtVideo(183.0))))

        val settings = listOf(8.0, 12.0, 15.0, 20.0, 30.0, 45.0).map { degrees ->
            Setting("fixed_${degrees.toInt()}deg") { Math.toRadians(degrees) }
        } + listOf(
            Setting("wobble_x2.07_floor5") { frame -> ((frame.wobbleRadians ?: Math.toRadians(30.0)) * SWAY_BAND_Z).coerceIn(Math.toRadians(5.0), Math.toRadians(60.0)) },
            Setting("wobble_x4_floor10") { frame -> ((frame.wobbleRadians ?: Math.toRadians(30.0)) * 4.0).coerceIn(Math.toRadians(10.0), Math.toRadians(60.0)) },
            Setting("wobble_x2.07_floor15") { frame -> ((frame.wobbleRadians ?: Math.toRadians(30.0)) * SWAY_BAND_Z).coerceIn(Math.toRadians(15.0), Math.toRadians(60.0)) },
        )

        val scores = settings.map { setting -> score(setting, frames, turnAhead, inCorner, outDir) }
        val summary = buildString {
            appendLine("setting,alerts,hits,false_alarms,real_turns,turns_called,mean_abs_heading_deg,frames_above_0.3_bits,median_entropy_bits,min_entropy_bits,cone_peak_bits,cone_heading_deg,rack_peak_bits,rack_heading_deg")
            for (score in scores) {
                appendLine(
                    String.format(
                        Locale.US,
                        "%s,%d,%d,%d,%d,%d,%.2f,%d,%.2f,%.2f,%.2f,%+.0f,%.2f,%+.0f",
                        score.name, score.alerts, score.hits, score.falseAlarms, score.realTurns, score.turnsCalled, score.meanAbsHeadingDegrees,
                        score.framesAbove03Bits, score.medianEntropyBits, score.minEntropyBits, score.conePeakBits, score.coneHeadingDegrees,
                        score.rackPeakBits, score.rackHeadingDegrees,
                    ),
                )
            }
        }
        File(outDir, "summary.csv").writeText(summary)
        print(summary)
    }

    private fun score(setting: Setting, frames: List<ReplayFrame>, turnAhead: DoubleArray, inCorner: BooleanArray, outDir: File): Score {
        val headings = DoubleArray(frames.size)
        val surprises = DoubleArray(frames.size)
        val entropies = DoubleArray(frames.size)
        for ((index, frame) in frames.withIndex()) {
            val field = PushFieldGuidance(PushFieldParameters(turnToleranceRadians = setting.toleranceFor(frame)))
            val guidance = field.evaluate(frame.obstacles, WalkerState.ALIGNED_WITH_CAMERA, 0L)
            headings[index] = guidance.desiredHeadingRadians
            surprises[index] = guidance.overallSurpriseBits
            entropies[index] = guidance.headingEntropyBits ?: Double.NaN
        }
        File(outDir, "${setting.name}.csv").printWriter().use { out ->
            out.println("video_seconds,heading_deg,surprise_bits,entropy_bits,yaw_change_next_2s_deg")
            for (index in frames.indices) {
                out.println(String.format(Locale.US, "%.2f,%+.1f,%.3f,%.3f,%.1f", frames[index].videoSeconds, Math.toDegrees(headings[index]), surprises[index], entropies[index], Math.toDegrees(turnAhead[index])))
            }
        }

        val alertThreshold = Math.toRadians(ALERT_DEGREES)
        val turnThreshold = Math.toRadians(TURN_DEGREES)
        val quietThreshold = Math.toRadians(QUIET_DEGREES)
        var alerts = 0
        var hits = 0
        var falseAlarms = 0
        for (index in frames.indices) {
            if (inCorner[index] || abs(headings[index]) < alertThreshold) continue
            alerts += 1
            val agreed = abs(turnAhead[index]) >= Math.toRadians(HIT_DEGREES) && Math.signum(turnAhead[index]) == Math.signum(headings[index])
            if (agreed) hits += 1 else if (abs(turnAhead[index]) < quietThreshold) falseAlarms += 1
        }
        var realTurns = 0
        var turnsCalled = 0
        for (index in frames.indices) {
            if (inCorner[index] || abs(turnAhead[index]) < turnThreshold) continue
            realTurns += 1
            // Called if any frame in the second before, this one included, asked for a turn the same way.
            val called = (index downTo maxOf(0, index - FRAMES_PER_SECOND)).any { earlier ->
                abs(headings[earlier]) >= alertThreshold && Math.signum(headings[earlier]) == Math.signum(turnAhead[index])
            }
            if (called) turnsCalled += 1
        }
        val sortedEntropy = entropies.filter { !it.isNaN() }.sorted()
        fun peak(from: Double, to: Double): Pair<Double, Double> {
            val index = frames.indices.filter { frames[it].videoSeconds in from..to }.maxByOrNull { surprises[it] } ?: return 0.0 to 0.0
            return surprises[index] to Math.toDegrees(headings[index])
        }
        val cone = peak(182.0, 187.0)
        val rack = peak(206.0, 219.5)
        return Score(
            name = setting.name,
            alerts = alerts,
            hits = hits,
            falseAlarms = falseAlarms,
            realTurns = realTurns,
            turnsCalled = turnsCalled,
            meanAbsHeadingDegrees = Math.toDegrees(headings.map { abs(it) }.average()),
            framesAbove03Bits = surprises.count { it > 0.3 },
            medianEntropyBits = sortedEntropy[sortedEntropy.size / 2],
            minEntropyBits = sortedEntropy.first(),
            conePeakBits = cone.first,
            coneHeadingDegrees = cone.second,
            rackPeakBits = rack.first,
            rackHeadingDegrees = rack.second,
        )
    }

    private companion object {
        const val NANOS_PER_SECOND = 1e9
        const val DEFAULT_VIDEO_OFFSET_SECONDS = 4.61
        const val LEAD_SECONDS = 2.0
        const val CORNER_SECONDS = 5.0
        const val CORNER_DEGREES = 60.0
        const val TURN_DEGREES = 15.0
        const val HIT_DEGREES = 10.0
        const val QUIET_DEGREES = 5.0
        const val ALERT_DEGREES = 10.0
        const val FRAMES_PER_SECOND = 4

        // 96-percent band of a normal distribution. The wobble-to-tolerance settings above multiply sway by this.
        const val SWAY_BAND_Z = 2.07
    }
}
