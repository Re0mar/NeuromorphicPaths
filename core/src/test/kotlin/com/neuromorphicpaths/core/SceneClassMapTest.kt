package com.neuromorphicpaths.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class SceneClassMapTest {

    /** Sky in the top half, pavement in the bottom half, a wall down the right column. */
    private val map = SceneClassMap.build(4, 4) { column, row ->
        when {
            column == 3 -> SceneClass.WALL
            row < 2 -> SceneClass.SKY
            else -> SceneClass.PAVEMENT
        }
    }

    @Test
    fun cellsReadBackByColumnAndRow() {
        assertEquals(SceneClass.SKY, map.classAt(0, 0))
        assertEquals(SceneClass.PAVEMENT, map.classAt(0, 3))
        assertEquals(SceneClass.WALL, map.classAt(3, 0))
    }

    @Test
    fun aFractionLandsInTheCellThatCoversIt() {
        assertEquals(SceneClass.SKY, map.classAtFraction(0.1, 0.1))
        assertEquals(SceneClass.PAVEMENT, map.classAtFraction(0.1, 0.6))
        assertEquals(SceneClass.WALL, map.classAtFraction(0.9, 0.6))
        // The far edge belongs to the last cell rather than falling off the grid.
        assertEquals(SceneClass.WALL, map.classAtFraction(1.0, 1.0))
    }

    @Test
    fun countsCoverEveryCellOnce() {
        val counts = map.countByClass()
        assertEquals(16, counts.values.sum())
        assertEquals(4, counts[SceneClass.WALL])
        assertEquals(6, counts[SceneClass.SKY])
        assertEquals(6, counts[SceneClass.PAVEMENT])
        assertEquals(0, counts[SceneClass.GRASS])
    }

    @Test
    fun aGridThatDoesNotMatchItsSizeIsRefused() {
        assertThrows(IllegalArgumentException::class.java) { SceneClassMap(4, 4, ByteArray(15)) }
    }
}
