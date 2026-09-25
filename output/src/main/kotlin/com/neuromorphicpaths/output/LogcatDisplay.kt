package com.neuromorphicpaths.output

import android.util.Log
import com.neuromorphicpaths.core.GuidanceDisplay
import com.neuromorphicpaths.core.GuidanceUpdate
import java.util.Locale

/** One line per frame in logcat: timestamp, counts, heading and surprise. The cheapest record of a run. */
class LogcatDisplay : GuidanceDisplay {
    override val name: String = "logcat"

    override suspend fun show(update: GuidanceUpdate) {
        val guidance = update.guidance
        Log.d(
            TAG,
            String.format(
                Locale.US,
                "t=%dns detections=%d obstacles=%d heading=%+.1fdeg surprise=%.2fbits",
                update.frame.timestampNanos,
                update.detections.size,
                update.obstacles.size,
                Math.toDegrees(guidance.desiredHeadingRadians),
                guidance.overallSurpriseBits,
            ),
        )
    }

    override fun close() = Unit

    private companion object {
        const val TAG = "Guidance"
    }
}
