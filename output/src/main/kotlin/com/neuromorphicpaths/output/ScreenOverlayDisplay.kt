package com.neuromorphicpaths.output

import android.graphics.Bitmap
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.GuidanceDisplay
import com.neuromorphicpaths.core.GuidanceUpdate
import java.nio.ByteBuffer
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext

/** What [GuidanceOverlay] draws: the frame as an image plus everything computed from it. */
class OverlayState(
    val image: ImageBitmap,
    val update: GuidanceUpdate,
)

/**
 * Holds the newest update as state for a composable to draw.
 *
 * The pixel conversion runs off the caller's thread. Only the latest update is kept, since a
 * screen has no use for the ones it missed. An update with no detections shows the new frame
 * under the previous boxes and arrow for a short hold, so one missed detection doesn't blink.
 */
class ScreenOverlayDisplay(
    private val conversionDispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val hold: OverlayHold = OverlayHold(),
) : GuidanceDisplay {

    override val name: String = "screen"

    private val latestState = MutableStateFlow<OverlayState?>(null)
    val latest: StateFlow<OverlayState?> = latestState.asStateFlow()

    private var lastWithDetections: GuidanceUpdate? = null

    override suspend fun show(update: GuidanceUpdate) {
        val image = withContext(conversionDispatcher) { update.frame.toImageBitmap() }
        latestState.value = OverlayState(image, withHold(update))
    }

    override fun close() {
        latestState.value = null
        lastWithDetections = null
        hold.reset()
    }

    private fun withHold(update: GuidanceUpdate): GuidanceUpdate {
        val hasDetections = update.detections.isNotEmpty()
        val show = hold.shouldShow(hasDetections, nowMillis = update.frame.timestampNanos / NANOS_PER_MILLI)
        if (hasDetections) lastWithDetections = update
        if (show) return update
        val held = lastWithDetections ?: return update
        return update.copy(detections = held.detections, obstacles = held.obstacles, guidance = held.guidance)
    }

    private companion object {
        const val NANOS_PER_MILLI = 1_000_000L
    }

    private fun Frame.toImageBitmap(): ImageBitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(rgba))
        return bitmap.asImageBitmap()
    }
}
