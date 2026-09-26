package com.neuromorphicpaths.input

import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.DocumentsContract
import com.neuromorphicpaths.core.CameraPose
import java.io.File

/**
 * A recorded walk: the video, and when Sensor Logger ran beside it, the orientation and
 * acceleration logs lined up with the video's own clock.
 *
 * With the logs, replay gets the pitch the camera really had frame by frame, the azimuth for
 * the wobble estimate, and the walker's speed from their steps. Without them, replay falls back
 * to whatever pose the caller supplies, which on a desk is a desk's pose, and to the field's
 * default speed.
 */
class Recording(
    val videoUri: Uri,
    val orientationLog: OrientationLog?,
    accelerationLog: AccelerationLog?,
    private val videoStartEpochMillis: Long?,
) {
    val hasLoggedPose: Boolean get() = orientationLog != null && videoStartEpochMillis != null

    // The step estimator is run over the whole acceleration log once, so a replay can ask for
    // the speed at any frame time without replaying the log alongside the video.
    private val speedTrace: SpeedTrace? = accelerationLog?.let { SpeedTrace.fromLog(it) }

    val hasLoggedSpeed: Boolean get() = speedTrace != null && videoStartEpochMillis != null

    /** The camera pose at a video position, or null when there is no log to read it from. */
    fun poseForPosition(heightMeters: Double): ((positionMillis: Long) -> CameraPose)? {
        val log = orientationLog ?: return null
        val start = videoStartEpochMillis ?: return null
        return { positionMillis -> log.poseAt((start + positionMillis) * NANOS_PER_MILLI, heightMeters) }
    }

    /** Where the camera pointed over the ground at a video position, or null without a log. */
    fun azimuthForPosition(positionMillis: Long): Double? {
        val log = orientationLog ?: return null
        val start = videoStartEpochMillis ?: return null
        return log.azimuthAt((start + positionMillis) * NANOS_PER_MILLI)
    }

    /** The walker's speed from their steps at a video position, or null without an acceleration log. */
    fun speedForPosition(positionMillis: Long): Double? {
        val trace = speedTrace ?: return null
        val start = videoStartEpochMillis ?: return null
        return trace.speedAt((start + positionMillis) * NANOS_PER_MILLI)
    }

    /** Speed against time, precomputed from an acceleration log, answered by nearest sample. */
    private class SpeedTrace(private val epochNanos: LongArray, private val metersPerSecond: DoubleArray) {
        fun speedAt(nanos: Long): Double? {
            if (epochNanos.isEmpty()) return null
            var low = 0
            var high = epochNanos.size - 1
            while (low < high) {
                val middle = (low + high) / 2
                if (epochNanos[middle] < nanos) low = middle + 1 else high = middle
            }
            return metersPerSecond[low]
        }

        companion object {
            fun fromLog(log: AccelerationLog): SpeedTrace {
                val estimator = StepCadenceSpeedEstimator()
                val times = ArrayList<Long>(log.samples.size)
                val speeds = ArrayList<Double>(log.samples.size)
                for (sample in log.samples) {
                    val speed = estimator.add(sample.epochNanos, sample.magnitudeMinusGravity) ?: continue
                    times += sample.epochNanos
                    speeds += speed
                }
                return SpeedTrace(times.toLongArray(), speeds.toDoubleArray())
            }
        }
    }

    companion object {
        const val ORIENTATION_LOG_NAME = "Orientation.csv"
        const val ACCELERATION_LOG_NAME = "TotalAcceleration.csv"
        private const val NANOS_PER_MILLI = 1_000_000L
        private const val VIDEO_MIME_PREFIX = "video/"
        private const val VIDEO_EXTENSION = ".mp4"

        /** A folder in the app's own storage, which needs no permission and which adb can fill. */
        fun fromDirectory(context: Context, directory: File): Recording? {
            val video = directory.listFiles()?.firstOrNull { it.name.endsWith(VIDEO_EXTENSION, ignoreCase = true) } ?: return null
            val orientation = File(directory, ORIENTATION_LOG_NAME).takeIf { it.isFile }?.readText()
            val acceleration = File(directory, ACCELERATION_LOG_NAME).takeIf { it.isFile }?.readText()
            return load(context, Uri.fromFile(video), orientation, acceleration)
        }

        /** A folder the person picked through the system's folder picker. */
        fun fromTree(context: Context, treeUri: Uri): Recording? {
            val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, DocumentsContract.getTreeDocumentId(treeUri))
            val projection = arrayOf(
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_MIME_TYPE,
            )
            var videoUri: Uri? = null
            var orientationUri: Uri? = null
            var accelerationUri: Uri? = null
            context.contentResolver.query(childrenUri, projection, null, null, null)?.use { cursor ->
                while (cursor.moveToNext()) {
                    val documentUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, cursor.getString(0))
                    val name = cursor.getString(1)
                    val mime = cursor.getString(2) ?: ""
                    if (videoUri == null && mime.startsWith(VIDEO_MIME_PREFIX)) videoUri = documentUri
                    if (name == ORIENTATION_LOG_NAME) orientationUri = documentUri
                    if (name == ACCELERATION_LOG_NAME) accelerationUri = documentUri
                }
            }
            val video = videoUri ?: return null
            fun readText(uri: Uri?): String? = uri?.let { context.contentResolver.openInputStream(it)?.bufferedReader()?.use { reader -> reader.readText() } }
            return load(context, video, readText(orientationUri), readText(accelerationUri))
        }

        private fun load(context: Context, videoUri: Uri, orientationText: String?, accelerationText: String?): Recording {
            val orientation = orientationText?.let { OrientationLog.parse(it) }
            val acceleration = accelerationText?.let { AccelerationLog.parse(it) }
            val start = if (orientation != null || acceleration != null) readVideoStartEpochMillis(context, videoUri) else null
            return Recording(videoUri, orientation, acceleration, start)
        }

        /** Null when the container carries no date, in which case the logs cannot be lined up. */
        private fun readVideoStartEpochMillis(context: Context, videoUri: Uri): Long? {
            val retriever = MediaMetadataRetriever()
            return try {
                retriever.setDataSource(context, videoUri)
                val date = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DATE) ?: return null
                val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: return null
                RecordingClock.videoStartEpochMillis(date, duration)
            } finally {
                retriever.release()
            }
        }
    }
}
