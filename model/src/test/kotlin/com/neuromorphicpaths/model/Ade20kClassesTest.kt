package com.neuromorphicpaths.model

import com.neuromorphicpaths.core.SceneClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import java.io.File

class Ade20kClassesTest {

    /** The table the app ships, read from the source tree so the test fails when the asset is missing. */
    private val shippedTable: String = File("src/main/assets/segformer/ade20k_classes.tsv").readText()

    @Test
    fun theShippedTableCoversAllHundredAndFiftyClassesInOrder() {
        val classes = Ade20kClasses.parse(shippedTable)

        assertEquals(150, classes.size)
        assertEquals(SceneClass.WALL, classes.sceneClassAt(0))
        assertEquals(SceneClass.BUILDING, classes.sceneClassAt(1))
        assertEquals(SceneClass.SKY, classes.sceneClassAt(2))
        assertEquals(SceneClass.PAVEMENT, classes.sceneClassAt(3))
        assertEquals(SceneClass.ROAD, classes.sceneClassAt(6))
        assertEquals(SceneClass.GRASS, classes.sceneClassAt(9))
        assertEquals(SceneClass.PAVEMENT, classes.sceneClassAt(11))
        assertEquals(SceneClass.DIRT, classes.sceneClassAt(13))
        assertEquals(SceneClass.STAIRS, classes.sceneClassAt(53))
        assertEquals(SceneClass.OTHER, classes.sceneClassAt(12))
    }

    @Test
    fun anIndexPastTheTableReadsAsOther() {
        val classes = Ade20kClasses.parse(shippedTable)

        assertEquals(SceneClass.OTHER, classes.sceneClassAt(150))
        assertEquals(SceneClass.OTHER, classes.sceneClassAt(-1))
    }

    @Test
    fun commentsAndBlankLinesAreSkipped() {
        val classes = Ade20kClasses.parse("# header\n\n0\twall\tWALL\n1\tsky\tSKY\n")

        assertEquals(2, classes.size)
        assertEquals(SceneClass.SKY, classes.sceneClassAt(1))
    }

    @Test
    fun aGapInTheIndicesIsRefused() {
        assertThrows(IllegalArgumentException::class.java) { Ade20kClasses.parse("0\twall\tWALL\n2\tsky\tSKY\n") }
    }

    @Test
    fun anUnknownSceneClassIsRefused() {
        assertThrows(IllegalArgumentException::class.java) { Ade20kClasses.parse("0\twall\tBRICK\n") }
    }

    @Test
    fun aLineWithTheWrongShapeIsRefused() {
        assertThrows(IllegalArgumentException::class.java) { Ade20kClasses.parse("0\tWALL\n") }
    }
}
