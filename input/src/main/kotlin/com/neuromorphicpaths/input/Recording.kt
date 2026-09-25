package com.neuromorphicpaths.input

import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.DocumentsContract
import com.neuromorphicpaths.core.CameraPose
import java.io.File

/**
 * A recorded walk: the video, and when Sensor Logger ran beside it, the orientation log lined up
 * with the video's own clock.
 *
 * With the log, replay gets the pitch the camera really had frame by frame and the azimuth for
 * the wobble estimate. Without it, replay falls back to whatever pose the caller supplies, which
 * on a desk is a desk's pose.
 */
class Recording(
    val videoUri: Uri,
    val orientationLog: OrientationLog?,
    private val videoStartEpochMillis: Long?,
) {
    val hasLoggedPose: Boolean get() = orientationLog != null && videoStartEpochMillis != null

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

    companion object {
        const val ORIENTATION_LOG_NAME = "Orientation.csv"
        private const val NANOS_PER_MILLI = 1_000_000L
        private const val VIDEO_MIME_PREFIX = "video/"
        private const val VIDEO_EXTENSION = ".mp4"

        /** A folder in the app's own storage, which needs no permission and which adb can fill. */
        fun fromDirectory(context: Context, directory: File): Recording? {
            val video = directory.listFiles()?.firstOrNull { it.name.endsWith(VIDEO_EXTENSION, ignoreCase = true) } ?: return null
            val log = File(directory, ORIENTATION_LOG_NAME).takeIf { it.isFile }?.readText()
            return load(context, Uri.fromFile(video), log)
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
            var logUri: Uri? = null
            context.contentResolver.query(childrenUri, projection, null, null, null)?.use { cursor ->
                while (cursor.moveToNext()) {
                    val documentUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, cursor.getString(0))
                    val name = cursor.getString(1)
                    val mime = cursor.getString(2) ?: ""
                    if (videoUri == null && mime.startsWith(VIDEO_MIME_PREFIX)) videoUri = documentUri
                    if (name == ORIENTATION_LOG_NAME) logUri = documentUri
                }
            }
            val video = videoUri ?: return null
            val log = logUri?.let { uri -> context.contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() } }
            return load(context, video, log)
        }

        private fun load(context: Context, videoUri: Uri, logText: String?): Recording {
            val log = logText?.let { OrientationLog.parse(it) }
            val start = log?.let { readVideoStartEpochMillis(context, videoUri) }
            return Recording(videoUri, log, start)
        }

        /** Null when the container carries no date, in which case the log cannot be lined up. */
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
