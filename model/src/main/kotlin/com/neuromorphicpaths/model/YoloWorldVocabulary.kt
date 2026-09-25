package com.neuromorphicpaths.model

import com.neuromorphicpaths.core.ObstacleClass

/** One model output class: the prompt it was exported with and the obstacle class it maps to. */
data class VocabularyEntry(
    val prompt: String,
    val obstacleClass: ObstacleClass,
)

/**
 * The class list the detector was exported with, in output order.
 *
 * Parsed from the same file the export script reads, so model index i means entry i on both
 * sides. Several prompts can map to one obstacle class, which is how "desk" and "table" both
 * end up as TABLE.
 */
class YoloWorldVocabulary(val entries: List<VocabularyEntry>) {

    val size: Int get() = entries.size

    fun obstacleClassAt(index: Int): ObstacleClass = entries.getOrNull(index)?.obstacleClass ?: ObstacleClass.UNKNOWN

    companion object {
        const val ASSET_PATH = "yolo_world/vocabulary.tsv"

        /** Parses the tab-separated file: prompt, tab, class name. Blank lines and # comments are skipped. */
        fun parse(text: String): YoloWorldVocabulary {
            val entries = text.lineSequence().withIndex().mapNotNull { (lineIndex, rawLine) ->
                val line = rawLine.trim()
                if (line.isEmpty() || line.startsWith("#")) return@mapNotNull null
                val fields = line.split('\t')
                require(fields.size == 2) { "Vocabulary line ${lineIndex + 1}: expected 'prompt<TAB>CLASS', got '$rawLine'" }
                val className = fields[1].trim()
                val obstacleClass = ObstacleClass.entries.firstOrNull { it.name == className }
                    ?: throw IllegalArgumentException("Vocabulary line ${lineIndex + 1}: unknown obstacle class '$className'")
                VocabularyEntry(fields[0].trim(), obstacleClass)
            }.toList()
            require(entries.isNotEmpty()) { "Vocabulary has no entries" }
            return YoloWorldVocabulary(entries)
        }
    }
}
