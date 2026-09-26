package com.neuromorphicpaths.math.tools

import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.GroundSurfaceMap
import com.neuromorphicpaths.core.NormalizedBox
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.ObstacleClass
import com.neuromorphicpaths.math.PushFieldGuidance
import com.neuromorphicpaths.math.PushFieldParameters
import org.junit.Test
import java.io.File
import java.util.Locale
import kotlin.math.atan2
import kotlin.math.hypot

/**
 * Re-runs the shipped field over a logged replay and reports the lowest surprise ahead frame by
 * frame, the number the floating overlay decides STOP on. This is where a STOP threshold or a
 * look-ahead stretch gets tried before it goes to the phone.
 *
 * Not a test. It runs through the `noPath` Gradle task, which passes the log path in as a
 * system property. Surfaces are not logged, so the ground term is missing here and the phone's
 * numbers read a little higher wherever the ground ahead is charged.
 */
class NoPathTool {

    @Test
    fun report() {
        val logPath = requireNotNull(System.getProperty("replay.log")) { "pass -Preplay.log=<absolute path to a logcat capture>" }
        val frames = ReplayLog.read(File(logPath))
        require(frames.isNotEmpty()) { "no Guidance frame lines in $logPath" }
        val parameters = PushFieldParameters()
        val shipped = PushFieldGuidance(parameters)
        val unstretched = PushFieldGuidance(parameters.copy(noWayThroughHorizonStretch = 1.0))

        // The speed the field itself would use for each frame: measured and floored, or the default.
        fun speedOf(frame: ReplayFrame) = frame.loggedSpeedMetersPerSecond?.coerceAtLeast(parameters.minimumWalkerSpeedMetersPerSecond)
            ?: parameters.defaultWalkerSpeedMetersPerSecond
        val aheadBits = DoubleArray(frames.size) { shipped.lowestSurpriseAheadBits(frames[it].obstacles, speedOf(frames[it]), GroundSurfaceMap.UNKNOWN_EVERYWHERE) }
        val unstretchedBits = DoubleArray(frames.size) { unstretched.lowestSurpriseAheadBits(frames[it].obstacles, speedOf(frames[it]), GroundSurfaceMap.UNKNOWN_EVERYWHERE) }
        summarize(String.format(Locale.US, "lowest surprise ahead, horizons x%.1f as shipped", parameters.noWayThroughHorizonStretch), aheadBits, frames)
        summarize("lowest surprise ahead, horizons x1 for comparison", unstretchedBits, frames)

        // A real dead end for scale: a wall straight across the view, one sample per half meter
        // from 4 m left to 4 m right, the way the scene locator thins a wall's feet. Two posts
        // 0.6 m either side of the line are the squeeze a walker fits through.
        val speed = parameters.defaultWalkerSpeedMetersPerSecond
        for (distanceMeters in listOf(1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)) {
            val wall = (-8..8).map { step -> wallSample(step * 0.5, distanceMeters) }
            val posts = listOf(wallSample(-0.6, distanceMeters), wallSample(0.6, distanceMeters))
            println(
                String.format(
                    Locale.US,
                    "at %.1f m, walking at %.1f m/s: wall across %.2f bits, two posts %.2f bits",
                    distanceMeters,
                    speed,
                    shipped.lowestSurpriseAheadBits(wall, speed, GroundSurfaceMap.UNKNOWN_EVERYWHERE),
                    shipped.lowestSurpriseAheadBits(posts, speed, GroundSurfaceMap.UNKNOWN_EVERYWHERE),
                ),
            )
        }
        for ((index, frame) in frames.withIndex()) {
            if (aheadBits[index] >= 3.0) {
                val nearest = frame.obstacles.minByOrNull { it.rangeMeters }
                val describe = nearest?.let { String.format(Locale.US, "%s %.1f m at %+.0f deg", it.obstacleClass, it.rangeMeters, Math.toDegrees(it.bearingRadians)) } ?: "none"
                println(String.format(Locale.US, "  t=%.2f ahead %.2f bits, %d obstacles, speed %s, nearest %s", frame.videoSeconds, aheadBits[index], frame.obstacles.size, frame.loggedSpeedMetersPerSecond, describe))
            }
        }
    }

    private fun wallSample(rightMeters: Double, forwardMeters: Double) = Obstacle(
        detection = Detection(ObstacleClass.WALL, 1.0, NormalizedBox(0.4, 0.4, 0.6, 0.8), null),
        bearingRadians = atan2(rightMeters, forwardMeters),
        rangeMeters = hypot(rightMeters, forwardMeters),
        closingSpeedMetersPerSecond = null,
    )

    private fun summarize(label: String, bits: DoubleArray, frames: List<ReplayFrame>) {
        val sorted = bits.sorted()
        fun quantile(fraction: Double) = sorted[((sorted.size - 1) * fraction).toInt()]
        println(String.format(Locale.US, "%s over %d frames: median %.2f, p90 %.2f, p99 %.2f, max %.2f bits", label, frames.size, quantile(0.5), quantile(0.9), quantile(0.99), sorted.last()))
        for (threshold in listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)) {
            val above = bits.indices.filter { bits[it] >= threshold }
            // Frames sit a quarter second apart, so an episode is a run with no gap over a second.
            val episodes = mutableListOf<Pair<Double, Double>>()
            for (index in above) {
                val seconds = frames[index].videoSeconds
                if (episodes.isNotEmpty() && seconds - episodes.last().second <= 1.0) {
                    episodes[episodes.size - 1] = episodes.last().first to seconds
                } else {
                    episodes += seconds to seconds
                }
            }
            val listed = episodes.joinToString(", ") { (start, end) -> String.format(Locale.US, "%.1f-%.1f", start, end) }
            println(String.format(Locale.US, "  at or above %.0f bits: %d frames, %d episodes: %s", threshold, above.size, episodes.size, listed))
        }
    }
}
