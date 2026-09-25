package com.neuromorphicpaths.input

import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import com.neuromorphicpaths.core.CameraIntrinsics
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.FrameSource
import com.neuromorphicpaths.core.PoseProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn

/**
 * Replays a recording by pulling one frame every [frameIntervalMillis] of video time.
 *
 * This is the mock input: a walk recorded on a phone or on the glasses, played back through the
 * same pipeline as the live camera. Frames come out already rotated the way the file says they
 * should be viewed. With [playInRealTime] the source paces itself to the video clock, so what is
 * on screen moves at walking speed. Without it, the source runs as fast as decoding allows.
 */
class VideoFileFrameSource(
    private val context: Context,
    private val uri: Uri,
    private val poseProvider: PoseProvider,
    private val intrinsics: CameraIntrinsics,
    private val frameIntervalMillis: Long = DEFAULT_FRAME_INTERVAL_MILLIS,
    private val playInRealTime: Boolean = true,
) : FrameSource {

    override val name: String = "video:${uri.lastPathSegment ?: uri}"

    override fun frames(): Flow<Frame> = flow {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)
            val durationMillis = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()
                ?: 0L
            var positionMillis = 0L
            while (positionMillis <= durationMillis) {
                val bitmap = retriever.getFrameAtTime(positionMillis * MICROS_PER_MILLI, MediaMetadataRetriever.OPTION_CLOSEST)
                    ?: break
                val frame = bitmap.toRgbaImage().toFrame(positionMillis * NANOS_PER_MILLI, poseProvider.currentPose(), intrinsics)
                bitmap.recycle()
                emit(frame)
                if (playInRealTime) delay(frameIntervalMillis)
                positionMillis += frameIntervalMillis
            }
        } finally {
            retriever.release()
        }
    }.flowOn(Dispatchers.IO)

    override fun close() = Unit

    companion object {
        /** Four frames a second, the rate a phone detector is expected to keep up with. */
        const val DEFAULT_FRAME_INTERVAL_MILLIS = 250L
        private const val MICROS_PER_MILLI = 1_000L
        private const val NANOS_PER_MILLI = 1_000_000L
    }
}
