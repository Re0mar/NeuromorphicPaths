package com.neuromorphicpaths.pixel.ar

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * The plane the app sends as the floor. The first walk sent the lowest one, which was a false
 * plane a meter under the real floor, and every arrow after it was nonsense.
 */
class FloorChoiceTest {
    @Test
    fun theLargestPlaneWinsNotTheLowest() {
        // The false plane: small and a meter lower. The floor: large and higher.
        val falsePlaneBelowTheFloor = FloorCandidate(extentXMeters = 0.6f, extentZMeters = 0.5f, centerHeightMeters = -2.3f)
        val floor = FloorCandidate(extentXMeters = 3.0f, extentZMeters = 4.0f, centerHeightMeters = -1.2f)

        assertEquals(1, FloorChoice.chooseFloorIndex(listOf(falsePlaneBelowTheFloor, floor)))
    }

    @Test
    fun anEmptyListHasNoFloor() {
        assertNull(FloorChoice.chooseFloorIndex(emptyList()))
    }

    @Test
    fun aTieGoesToTheLowerPlane() {
        val table = FloorCandidate(extentXMeters = 2.0f, extentZMeters = 1.0f, centerHeightMeters = -0.4f)
        val floor = FloorCandidate(extentXMeters = 1.0f, extentZMeters = 2.0f, centerHeightMeters = -1.2f)

        assertEquals(1, FloorChoice.chooseFloorIndex(listOf(table, floor)))
        // Order of arrival does not decide a tie.
        assertEquals(0, FloorChoice.chooseFloorIndex(listOf(floor, table)))
    }

    @Test
    fun aSinglePlaneIsChosenWhateverItsHeight() {
        val onlyOne = FloorCandidate(extentXMeters = 0.3f, extentZMeters = 0.3f, centerHeightMeters = 0.8f)

        assertEquals(0, FloorChoice.chooseFloorIndex(listOf(onlyOne)))
    }

    @Test
    fun aCandidateWithNoAreaIsNeverChosen() {
        // NaN extents compare false with everything, so a plane ARCore has not measured yet can
        // neither win nor tie. Alone it leaves the laptop to fit its own floor.
        val unmeasured = FloorCandidate(extentXMeters = Float.NaN, extentZMeters = Float.NaN, centerHeightMeters = -1.2f)
        val floor = FloorCandidate(extentXMeters = 2.0f, extentZMeters = 2.0f, centerHeightMeters = -1.2f)

        assertNull(FloorChoice.chooseFloorIndex(listOf(unmeasured)))
        assertEquals(1, FloorChoice.chooseFloorIndex(listOf(unmeasured, floor)))
    }

    @Test
    fun aLargerHigherPlaneBeatsASmallerLowerOneByArea() {
        // Area, not a single extent. A long thin strip loses to a squarer plane with more area.
        val strip = FloorCandidate(extentXMeters = 5.0f, extentZMeters = 0.4f, centerHeightMeters = -1.5f)
        val square = FloorCandidate(extentXMeters = 1.6f, extentZMeters = 1.6f, centerHeightMeters = -1.0f)

        assertEquals(1, FloorChoice.chooseFloorIndex(listOf(strip, square)))
    }
}
