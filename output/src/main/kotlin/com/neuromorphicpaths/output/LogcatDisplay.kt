package com.neuromorphicpaths.output

import android.util.Log
import com.neuromorphicpaths.core.GuidanceDisplay
import com.neuromorphicpaths.core.GuidanceUpdate
import com.neuromorphicpaths.core.ObstacleClass
import com.neuromorphicpaths.core.SceneClass
import java.util.Locale

/** One line per frame in logcat: timestamp, detector time, counts, heading, surprise and the walker's numbers. The cheapest record of a run. */
class LogcatDisplay : GuidanceDisplay {
    override val name: String = "logcat"

    override suspend fun show(update: GuidanceUpdate) {
        val guidance = update.guidance
        Log.d(
            TAG,
            String.format(
                Locale.US,
                "t=%dns detect=%dms detections=%d obstacles=%d heading=%+.1fdeg surprise=%.2fbits speed=%.2fm/s wobble=%.1fdeg tolerance=%.1fdeg entropy=%.2fbits segment=%dms structures=%d",
                update.frame.timestampNanos,
                update.detectorNanos / NANOS_PER_MILLI,
                update.detections.size,
                update.obstacles.size,
                Math.toDegrees(guidance.desiredHeadingRadians),
                guidance.overallSurpriseBits,
                // NaN where nothing has measured it yet, which a log reader can tell from a zero.
                update.walker.speedMetersPerSecond ?: Double.NaN,
                Math.toDegrees(guidance.walkerWobbleRadians ?: Double.NaN),
                Math.toDegrees(guidance.turnToleranceRadians ?: Double.NaN),
                guidance.headingEntropyBits ?: Double.NaN,
                // -1 on a frame the segmenter did not run on, so a timing script can skip those.
                update.segmenterNanos?.let { it / NANOS_PER_MILLI } ?: NO_SEGMENTER_RUN,
                update.obstacles.count { it.obstacleClass in STRUCTURE_CLASSES },
            ),
        )
        // One line per obstacle after the frame line, so a replay can be read per track. The
        // frame line stays first and keeps its shape, since scripts already parse it.
        for (entry in guidance.perObstacle) {
            val obstacle = entry.obstacle
            Log.d(
                TAG,
                String.format(
                    Locale.US,
                    "  t=%dns track=%d class=%s confidence=%.2f range=%.2fm bearing=%+.1fdeg closing=%.2fm/s surprise=%.2fbits",
                    update.frame.timestampNanos,
                    obstacle.detection.trackId ?: -1,
                    obstacle.obstacleClass.name,
                    obstacle.detection.confidence,
                    obstacle.rangeMeters,
                    Math.toDegrees(obstacle.bearingRadians),
                    obstacle.closingSpeedMetersPerSecond ?: Double.NaN,
                    entry.surpriseBits,
                ),
            )
        }
    }

    override fun close() = Unit

    private companion object {
        const val TAG = "Guidance"
        const val NANOS_PER_MILLI = 1_000_000L
        const val NO_SEGMENTER_RUN = -1L

        // The classes only a segmenter produces, so the count says how many samples the scene added.
        val STRUCTURE_CLASSES: Set<ObstacleClass> = SceneClass.entries.mapNotNull { it.structure }.toSet()
    }
}
