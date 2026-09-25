package com.neuromorphicpaths.model

import com.neuromorphicpaths.core.ObstacleClass
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class YoloWorldVocabularyTest {

    @Test
    fun parsesPromptsInOrderAndSkipsComments() {
        val vocabulary = YoloWorldVocabulary.parse(
            """
            # a comment
            tree	TREE

            trash can	TRASH_CAN
            """.trimIndent(),
        )

        assertEquals(listOf("tree", "trash can"), vocabulary.entries.map { it.prompt })
        assertEquals(ObstacleClass.TRASH_CAN, vocabulary.obstacleClassAt(1))
        assertEquals(ObstacleClass.UNKNOWN, vocabulary.obstacleClassAt(7))
    }

    @Test
    fun unknownClassNameIsRejectedWithTheLineNumber() {
        val failure = assertThrows(IllegalArgumentException::class.java) {
            YoloWorldVocabulary.parse("tree\tTREE\nrock\tROCK")
        }

        assertTrue(failure.message.orEmpty().contains("line 2"))
    }

    @Test
    fun shippedVocabularyCoversEveryObstacleClassExceptUnknown() {
        // Gradle runs unit tests with the module directory as the working directory.
        val shipped = File("src/main/assets/" + YoloWorldVocabulary.ASSET_PATH)
        val vocabulary = YoloWorldVocabulary.parse(shipped.readText())

        val covered = vocabulary.entries.map { it.obstacleClass }.toSet()
        val expected = ObstacleClass.entries.toSet() - ObstacleClass.UNKNOWN
        assertEquals(expected, covered)
    }
}
