package com.neuromorphicpaths.model

import com.neuromorphicpaths.core.SceneClass

/**
 * The segmenter's class table: which [SceneClass] each of the model's output indices means.
 *
 * Parsed from the same file the export script checks against the checkpoint, so model index i
 * means line i on both sides. Kept as a byte per index so a map of a few thousand cells is
 * translated with one array lookup each.
 */
class Ade20kClasses private constructor(private val sceneOrdinals: ByteArray) {

    val size: Int get() = sceneOrdinals.size

    /** The scene class ordinal for a model index. Indices the table does not cover read as OTHER. */
    fun sceneOrdinalAt(modelIndex: Int): Byte =
        if (modelIndex in sceneOrdinals.indices) sceneOrdinals[modelIndex] else OTHER_ORDINAL

    fun sceneClassAt(modelIndex: Int): SceneClass = SceneClass.entries[sceneOrdinalAt(modelIndex).toInt()]

    companion object {
        const val ASSET_PATH = "segformer/ade20k_classes.tsv"

        private val OTHER_ORDINAL = SceneClass.OTHER.ordinal.toByte()

        /**
         * Parses the tab-separated file: index, label, scene class. Blank lines and # comments are
         * skipped. Indices must run from 0 without gaps, since the model's output is positional.
         */
        fun parse(text: String): Ade20kClasses {
            val ordinals = mutableListOf<Byte>()
            for ((lineIndex, rawLine) in text.lineSequence().withIndex()) {
                val line = rawLine.trim()
                if (line.isEmpty() || line.startsWith("#")) continue
                val fields = line.split('\t')
                require(fields.size == 3) { "Class table line ${lineIndex + 1}: expected 'index<TAB>label<TAB>SCENE_CLASS', got '$rawLine'" }
                val index = fields[0].trim().toIntOrNull()
                    ?: throw IllegalArgumentException("Class table line ${lineIndex + 1}: index '${fields[0]}' is not a number")
                require(index == ordinals.size) { "Class table line ${lineIndex + 1}: expected index ${ordinals.size}, got $index" }
                val className = fields[2].trim()
                val sceneClass = SceneClass.entries.firstOrNull { it.name == className }
                    ?: throw IllegalArgumentException("Class table line ${lineIndex + 1}: unknown scene class '$className'")
                ordinals += sceneClass.ordinal.toByte()
            }
            require(ordinals.isNotEmpty()) { "Class table has no entries" }
            return Ade20kClasses(ordinals.toByteArray())
        }
    }
}
