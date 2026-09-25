package com.neuromorphicpaths.model

import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.NormalizedBox
import kotlin.math.max
import kotlin.math.min

/**
 * How a frame was fitted into the square model input: scaled to fit, then centered with
 * padding. Needed again on the way out to put boxes back onto the frame.
 */
data class Letterbox(
    val scale: Double,
    val padX: Double,
    val padY: Double,
    val inputSize: Int,
) {
    companion object {
        fun fit(frameWidth: Int, frameHeight: Int, inputSize: Int): Letterbox {
            val scale = min(inputSize / frameWidth.toDouble(), inputSize / frameHeight.toDouble())
            return Letterbox(
                scale = scale,
                padX = (inputSize - frameWidth * scale) / 2.0,
                padY = (inputSize - frameHeight * scale) / 2.0,
                inputSize = inputSize,
            )
        }
    }
}

/** A box straight out of the model: center and size in input pixels, best class and its score. */
data class RawBox(
    val centerX: Double,
    val centerY: Double,
    val width: Double,
    val height: Double,
    val classIndex: Int,
    val score: Double,
)

/**
 * Turns the raw output tensor of a YOLO-style detector into detections on the frame.
 *
 * The export writes one image as a [4 + classes][anchors] matrix, row-major: four box rows
 * (center x, center y, width, height in input pixels) followed by one score row per class.
 * Every anchor takes its best class, low scores go, overlapping boxes go through a single
 * class-agnostic non-maximum suppression, and what survives is mapped back through the
 * letterbox onto the frame as fractions.
 */
class YoloDecoder(
    private val vocabulary: YoloWorldVocabulary,
    private val confidenceThreshold: Double = DEFAULT_CONFIDENCE_THRESHOLD,
    private val iouThreshold: Double = DEFAULT_IOU_THRESHOLD,
) {

    fun decode(output: FloatArray, anchorCount: Int, letterbox: Letterbox, frameWidth: Int, frameHeight: Int): List<Detection> {
        val classCount = vocabulary.size
        require(output.size >= (BOX_ROWS + classCount) * anchorCount) {
            "Output has ${output.size} values, need ${(BOX_ROWS + classCount) * anchorCount} for $classCount classes and $anchorCount anchors"
        }
        val candidates = ArrayList<RawBox>()
        for (anchor in 0 until anchorCount) {
            var bestClass = -1
            var bestScore = 0f
            for (classIndex in 0 until classCount) {
                val score = output[(BOX_ROWS + classIndex) * anchorCount + anchor]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = classIndex
                }
            }
            if (bestClass >= 0 && bestScore >= confidenceThreshold) {
                candidates += RawBox(
                    centerX = output[anchor].toDouble(),
                    centerY = output[anchorCount + anchor].toDouble(),
                    width = output[2 * anchorCount + anchor].toDouble(),
                    height = output[3 * anchorCount + anchor].toDouble(),
                    classIndex = bestClass,
                    score = bestScore.toDouble(),
                )
            }
        }
        return suppressOverlaps(candidates).map { it.toDetection(letterbox, frameWidth, frameHeight) }
    }

    /** Greedy non-maximum suppression across all classes, so "desk" and "table" on one object give one box. */
    private fun suppressOverlaps(candidates: List<RawBox>): List<RawBox> {
        val kept = ArrayList<RawBox>()
        for (candidate in candidates.sortedByDescending { it.score }) {
            if (kept.none { intersectionOverUnion(it, candidate) > iouThreshold }) kept += candidate
        }
        return kept
    }

    private fun RawBox.toDetection(letterbox: Letterbox, frameWidth: Int, frameHeight: Int): Detection {
        fun toFrameX(inputX: Double) = ((inputX - letterbox.padX) / letterbox.scale / frameWidth).coerceIn(0.0, 1.0)
        fun toFrameY(inputY: Double) = ((inputY - letterbox.padY) / letterbox.scale / frameHeight).coerceIn(0.0, 1.0)
        return Detection(
            obstacleClass = vocabulary.obstacleClassAt(classIndex),
            confidence = score,
            box = NormalizedBox(
                left = toFrameX(centerX - width / 2.0),
                top = toFrameY(centerY - height / 2.0),
                right = toFrameX(centerX + width / 2.0),
                bottom = toFrameY(centerY + height / 2.0),
            ),
        )
    }

    private fun intersectionOverUnion(first: RawBox, second: RawBox): Double {
        val overlapWidth = max(0.0, min(first.centerX + first.width / 2, second.centerX + second.width / 2) - max(first.centerX - first.width / 2, second.centerX - second.width / 2))
        val overlapHeight = max(0.0, min(first.centerY + first.height / 2, second.centerY + second.height / 2) - max(first.centerY - first.height / 2, second.centerY - second.height / 2))
        val overlap = overlapWidth * overlapHeight
        val union = first.width * first.height + second.width * second.height - overlap
        return if (union <= 0.0) 0.0 else overlap / union
    }

    companion object {
        const val DEFAULT_CONFIDENCE_THRESHOLD = 0.25
        const val DEFAULT_IOU_THRESHOLD = 0.5
        private const val BOX_ROWS = 4
    }
}
