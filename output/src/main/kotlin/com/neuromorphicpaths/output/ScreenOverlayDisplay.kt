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
 * screen has no use for the ones it missed. What is drawn is exactly what the pipeline
 * produced for that frame. A missed detection does not blink the boxes off, because the
 * tracker upstream carries a lost box through a couple of frames, so the numbers on screen
 * and in the log are the same ones.
 */
class ScreenOverlayDisplay(
    private val conversionDispatcher: CoroutineDispatcher = Dispatchers.Default,
) : GuidanceDisplay {

    override val name: String = "screen"

    private val latestState = MutableStateFlow<OverlayState?>(null)
    val latest: StateFlow<OverlayState?> = latestState.asStateFlow()

    override suspend fun show(update: GuidanceUpdate) {
        val image = withContext(conversionDispatcher) { update.frame.toImageBitmap() }
        latestState.value = OverlayState(image, update)
    }

    override fun close() {
        latestState.value = null
    }

    private fun Frame.toImageBitmap(): ImageBitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(rgba))
        return bitmap.asImageBitmap()
    }
}
