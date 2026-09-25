package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.NormalizedBox
import kotlin.math.max
import kotlin.math.min

/**
 * How boxes are matched between frames and how long a lost one is kept.
 *
 * Minimum overlap is the intersection over union below which two boxes are not the same thing.
 * Max misses is how many consecutive frames a track survives without a box before it is
 * dropped, and confidence decay is what each missed frame multiplies its confidence by, so a
 * coasting track fades in the field rather than vanishing or staying at full strength.
 */
data class TrackingParameters(
    val minimumOverlap: Double = 0.3,
    val maxMisses: Int = 2,
    val confidenceDecayPerMiss: Double = 0.6,
) {
    init {
        require(minimumOverlap in 0.0..1.0) { "minimumOverlap must be a fraction" }
        require(maxMisses >= 0) { "maxMisses cannot be negative" }
        require(confidenceDecayPerMiss in 0.0..1.0) { "confidenceDecayPerMiss must be a fraction" }
    }
}

/**
 * Gives detections a track id that follows the same object from frame to frame.
 *
 * Matching is greedy by box overlap: the pair with the most overlap is matched first, and a
 * pair below the minimum is not matched at all. A detection nothing claims starts a new track.
 * A track nothing claims is kept for a few frames at a decaying confidence, since the phone's
 * detector runs at two or three frames a second and a single missed box would otherwise flip
 * the desired heading for a frame. The class follows the newest box, so a track that the
 * detector renames keeps its id.
 *
 * Coasting detections carry the last box seen, so anything downstream that estimates motion
 * from box positions should ask [isCoasting] before trusting one.
 */
class DetectionTracker(private val parameters: TrackingParameters = TrackingParameters()) {

    private class Track(val id: Int, var last: Detection, var misses: Int = 0)

    private var tracks = listOf<Track>()
    private var nextId = 1

    /** The track ids that were carried through the last [update] without a fresh box. */
    private val coastingIds = mutableSetOf<Int>()

    fun isCoasting(trackId: Int): Boolean = trackId in coastingIds

    /** Returns [detections] with track ids, plus a fading copy of each recently lost track. */
    fun update(detections: List<Detection>): List<Detection> {
        val candidatePairs = mutableListOf<Triple<Double, Int, Int>>()
        for ((trackIndex, track) in tracks.withIndex()) {
            for ((detectionIndex, detection) in detections.withIndex()) {
                val overlap = intersectionOverUnion(track.last.box, detection.box)
                if (overlap >= parameters.minimumOverlap) candidatePairs += Triple(overlap, trackIndex, detectionIndex)
            }
        }
        candidatePairs.sortByDescending { it.first }

        val claimedTracks = mutableSetOf<Int>()
        val matched = arrayOfNulls<Detection>(detections.size)
        for ((_, trackIndex, detectionIndex) in candidatePairs) {
            if (trackIndex in claimedTracks || matched[detectionIndex] != null) continue
            claimedTracks += trackIndex
            val track = tracks[trackIndex]
            track.last = detections[detectionIndex].copy(trackId = track.id)
            track.misses = 0
            matched[detectionIndex] = track.last
        }

        val started = mutableListOf<Track>()
        val output = mutableListOf<Detection>()
        for ((detectionIndex, detection) in detections.withIndex()) {
            output += matched[detectionIndex] ?: run {
                val track = Track(nextId, detection.copy(trackId = nextId))
                nextId += 1
                started += track
                track.last
            }
        }

        coastingIds.clear()
        val survivors = mutableListOf<Track>()
        for ((trackIndex, track) in tracks.withIndex()) {
            if (trackIndex in claimedTracks) {
                survivors += track
                continue
            }
            track.misses += 1
            if (track.misses > parameters.maxMisses) continue
            track.last = track.last.copy(confidence = track.last.confidence * parameters.confidenceDecayPerMiss)
            coastingIds += track.id
            output += track.last
            survivors += track
        }
        tracks = survivors + started
        return output
    }

    companion object {
        fun intersectionOverUnion(first: NormalizedBox, second: NormalizedBox): Double {
            val overlapWidth = max(0.0, min(first.right, second.right) - max(first.left, second.left))
            val overlapHeight = max(0.0, min(first.bottom, second.bottom) - max(first.top, second.top))
            val overlap = overlapWidth * overlapHeight
            val union = first.width * first.height + second.width * second.height - overlap
            return if (union <= 0.0) 0.0 else overlap / union
        }
    }
}
