package com.neuromorphicpaths.core

/**
 * A segmenter's answer for one frame: a coarse grid of [SceneClass] stretched over the whole frame.
 *
 * Cells are row-major from the top-left. Cell (column, row) covers the frame from column / width
 * to (column + 1) / width across and from row / height to (row + 1) / height down, whatever the
 * frame's own size, so a locator asks in frame fractions and never learns the model's input size.
 * Classes are kept as ordinals in a byte array: a map is made a few times a second and boxing
 * four thousand enum values each time would be waste.
 */
class SceneClassMap(
    val width: Int,
    val height: Int,
    private val ordinals: ByteArray,
) {
    init {
        require(width > 0 && height > 0) { "Scene map needs a positive size, got ${width}x$height" }
        require(ordinals.size == width * height) { "Scene map of ${width}x$height needs ${width * height} cells, got ${ordinals.size}" }
    }

    fun classAt(column: Int, row: Int): SceneClass = CLASSES[ordinals[row * width + column].toInt()]

    /** The class under a point given as fractions of the frame, 0..1 from the top-left. Points on the far edge land in the last cell. */
    fun classAtFraction(x: Double, y: Double): SceneClass {
        val column = (x * width).toInt().coerceIn(0, width - 1)
        val row = (y * height).toInt().coerceIn(0, height - 1)
        return classAt(column, row)
    }

    /** How many cells hold each class, for a log line or a test. */
    fun countByClass(): Map<SceneClass, Int> {
        val counts = IntArray(CLASSES.size)
        for (ordinal in ordinals) counts[ordinal.toInt()] += 1
        return CLASSES.indices.associate { CLASSES[it] to counts[it] }
    }

    companion object {
        private val CLASSES: Array<SceneClass> = SceneClass.entries.toTypedArray()

        /** Builds a map from a rule, for tests and scripted segmenters. */
        fun build(width: Int, height: Int, classOf: (column: Int, row: Int) -> SceneClass): SceneClassMap {
            val ordinals = ByteArray(width * height)
            for (row in 0 until height) {
                for (column in 0 until width) {
                    ordinals[row * width + column] = classOf(column, row).ordinal.toByte()
                }
            }
            return SceneClassMap(width, height, ordinals)
        }
    }
}
