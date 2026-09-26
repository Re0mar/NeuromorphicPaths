package com.neuromorphicpaths.math.tools

import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.NormalizedBox
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.ObstacleClass
import java.io.File

/** One replayed frame as the log recorded it: its video time, the obstacles the locator placed, and the walker's wobble. */
class ReplayFrame(
    val videoSeconds: Double,
    val obstacles: List<Obstacle>,
    val wobbleRadians: Double?,
    val loggedHeadingRadians: Double,
    val loggedSurpriseBits: Double,
    val loggedSpeedMetersPerSecond: Double? = null,
)

/**
 * Reads the Guidance lines of a replay's logcat capture back into obstacles, so the field can be
 * re-run over a real walk on the laptop with different parameters.
 *
 * A capture can hold several runs, live ones included, and replay frame times restart at zero,
 * so the file is split there and the longest run kept. The obstacle lines carry everything the
 * field reads, class, confidence, range, bearing and closing speed. The box is not logged and the
 * field does not use it, so a placeholder stands in.
 */
object ReplayLog {
    private val FRAME = Regex(
        """t=(\d+)ns detect=\d+ms detections=\d+ obstacles=\d+ heading=([-+]?[\d.]+)deg surprise=([\d.]+)bits speed=([\w.]+)m/s wobble=([\w.]+)deg""",
    )
    private val OBSTACLE = Regex(
        """t=(\d+)ns track=(-?\d+) class=(\w+) confidence=([\d.]+) range=([\d.]+)m bearing=([-+]?[\d.]+)deg closing=([\w.-]+)m/s""",
    )
    private val PLACEHOLDER_BOX = NormalizedBox(0.4, 0.4, 0.6, 0.8)

    fun read(file: File): List<ReplayFrame> {
        val runs = mutableListOf(mutableListOf<ReplayFrame>())
        val pending = mutableListOf<Obstacle>()
        var current: ReplayFrame? = null

        fun finish() {
            val frame = current ?: return
            runs.last() += ReplayFrame(frame.videoSeconds, pending.toList(), frame.wobbleRadians, frame.loggedHeadingRadians, frame.loggedSurpriseBits, frame.loggedSpeedMetersPerSecond)
            pending.clear()
            current = null
        }

        file.forEachLine { line ->
            val obstacle = OBSTACLE.find(line)
            if (obstacle != null) {
                val (_, track, className, confidence, range, bearing, closing) = obstacle.destructured
                pending += Obstacle(
                    detection = Detection(ObstacleClass.valueOf(className), confidence.toDouble(), PLACEHOLDER_BOX, track.toInt().takeIf { it >= 0 }),
                    bearingRadians = Math.toRadians(bearing.toDouble()),
                    rangeMeters = range.toDouble(),
                    closingSpeedMetersPerSecond = closing.toDoubleOrNull()?.takeIf { it.isFinite() },
                )
                return@forEachLine
            }
            val frame = FRAME.find(line) ?: return@forEachLine
            finish()
            val (nanos, heading, surprise, speed, wobble) = frame.destructured
            val seconds = nanos.toLong() / 1e9
            if (seconds == 0.0 && runs.last().isNotEmpty()) runs.add(mutableListOf())
            current = ReplayFrame(
                videoSeconds = seconds,
                obstacles = emptyList(),
                wobbleRadians = wobble.toDoubleOrNull()?.takeIf { it.isFinite() }?.let(Math::toRadians),
                loggedHeadingRadians = Math.toRadians(heading.toDouble()),
                loggedSurpriseBits = surprise.toDouble(),
                loggedSpeedMetersPerSecond = speed.toDoubleOrNull()?.takeIf { it.isFinite() },
            )
        }
        finish()
        return runs.maxByOrNull { it.size } ?: emptyList()
    }
}
