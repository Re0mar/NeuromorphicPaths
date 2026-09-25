package com.neuromorphicpaths.model

import com.neuromorphicpaths.core.ObstacleClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class YoloDecoderTest {

    private val vocabulary = YoloWorldVocabulary(
        listOf(
            VocabularyEntry("table", ObstacleClass.TABLE),
            VocabularyEntry("desk", ObstacleClass.TABLE),
            VocabularyEntry("chair", ObstacleClass.CHAIR),
        ),
    )
    private val decoder = YoloDecoder(vocabulary)

    /** Builds a [4 + classes][anchors] output with the given boxes, everything else zero. */
    private fun output(anchorCount: Int, vararg boxes: RawBox): FloatArray {
        val rows = 4 + vocabulary.size
        val values = FloatArray(rows * anchorCount)
        for ((anchor, box) in boxes.withIndex()) {
            values[anchor] = box.centerX.toFloat()
            values[anchorCount + anchor] = box.centerY.toFloat()
            values[2 * anchorCount + anchor] = box.width.toFloat()
            values[3 * anchorCount + anchor] = box.height.toFloat()
            values[(4 + box.classIndex) * anchorCount + anchor] = box.score.toFloat()
        }
        return values
    }

    @Test
    fun letterboxCentersTheShorterSide() {
        val letterbox = Letterbox.fit(frameWidth = 640, frameHeight = 480, inputSize = 320)

        assertEquals(0.5, letterbox.scale, 1e-12)
        assertEquals(0.0, letterbox.padX, 1e-12)
        assertEquals(40.0, letterbox.padY, 1e-12)
    }

    @Test
    fun boxesMapBackThroughTheLetterboxOntoTheFrame() {
        val letterbox = Letterbox.fit(640, 480, 320)
        val raw = RawBox(centerX = 160.0, centerY = 160.0, width = 64.0, height = 64.0, classIndex = 2, score = 0.9)

        val detections = decoder.decode(output(anchorCount = 3, raw), anchorCount = 3, letterbox = letterbox, frameWidth = 640, frameHeight = 480)

        val detection = detections.single()
        assertEquals(ObstacleClass.CHAIR, detection.obstacleClass)
        assertEquals(0.9, detection.confidence, 1e-6)
        // Input x 128..192 at scale 0.5 is frame x 256..384, over 640.
        assertEquals(0.4, detection.box.left, 1e-9)
        assertEquals(0.6, detection.box.right, 1e-9)
        // Input y 128..192 less 40 padding at scale 0.5 is frame y 176..304, over 480.
        assertEquals(176.0 / 480.0, detection.box.top, 1e-9)
        assertEquals(304.0 / 480.0, detection.box.bottom, 1e-9)
    }

    @Test
    fun overlappingBoxesOfDifferentPromptsCollapseToTheBest() {
        val letterbox = Letterbox.fit(320, 320, 320)
        val table = RawBox(160.0, 160.0, 100.0, 100.0, classIndex = 0, score = 0.8)
        val desk = RawBox(164.0, 158.0, 100.0, 100.0, classIndex = 1, score = 0.6)

        val detections = decoder.decode(output(2, table, desk), 2, letterbox, 320, 320)

        assertEquals(1, detections.size)
        assertEquals(0.8, detections.single().confidence, 1e-6)
    }

    @Test
    fun lowScoresAreDropped() {
        val letterbox = Letterbox.fit(320, 320, 320)
        val faint = RawBox(100.0, 100.0, 40.0, 40.0, classIndex = 2, score = 0.1)

        assertTrue(decoder.decode(output(1, faint), 1, letterbox, 320, 320).isEmpty())
    }

    @Test
    fun boxesPastTheFrameEdgeAreClamped() {
        val letterbox = Letterbox.fit(320, 320, 320)
        val edge = RawBox(centerX = 310.0, centerY = 160.0, width = 60.0, height = 60.0, classIndex = 2, score = 0.9)

        val detection = decoder.decode(output(1, edge), 1, letterbox, 320, 320).single()

        assertEquals(1.0, detection.box.right, 0.0)
        assertTrue(detection.box.left < detection.box.right)
    }
}
