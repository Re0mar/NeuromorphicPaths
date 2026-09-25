package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.NormalizedBox
import com.neuromorphicpaths.core.ObstacleClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DetectionTrackerTest {

    private fun box(left: Double, top: Double, right: Double, bottom: Double, obstacleClass: ObstacleClass = ObstacleClass.PERSON, confidence: Double = 0.9) =
        Detection(obstacleClass, confidence, NormalizedBox(left, top, right, bottom))

    @Test
    fun theSameBoxKeepsItsIdAndAShiftedOneFollowsIt() {
        val tracker = DetectionTracker()
        val first = tracker.update(listOf(box(0.4, 0.4, 0.6, 0.8)))
        val second = tracker.update(listOf(box(0.42, 0.41, 0.62, 0.81)))

        assertEquals(1, first.single().trackId)
        assertEquals(1, second.single().trackId)
        assertFalse(tracker.isCoasting(1))
    }

    @Test
    fun aBoxSomewhereElseStartsANewTrack() {
        val tracker = DetectionTracker()
        tracker.update(listOf(box(0.1, 0.1, 0.2, 0.3)))
        val second = tracker.update(listOf(box(0.7, 0.5, 0.9, 0.9)))

        // The new box gets the next id, and the first box coasts beside it for now.
        assertEquals(listOf(2, 1), second.map { it.trackId })
        assertFalse(tracker.isCoasting(2))
        assertTrue(tracker.isCoasting(1))
    }

    @Test
    fun twoBoxesKeepTheirOwnIdsWhenBothMove() {
        val tracker = DetectionTracker()
        tracker.update(listOf(box(0.1, 0.4, 0.3, 0.8), box(0.6, 0.4, 0.8, 0.8)))
        val second = tracker.update(listOf(box(0.62, 0.4, 0.82, 0.8), box(0.12, 0.4, 0.32, 0.8)))

        assertEquals(listOf(2, 1), second.map { it.trackId })
    }

    @Test
    fun aMissedFrameKeepsTheTrackAtLowerConfidenceThenDropsIt() {
        val tracker = DetectionTracker(TrackingParameters(maxMisses = 2, confidenceDecayPerMiss = 0.5))
        tracker.update(listOf(box(0.4, 0.4, 0.6, 0.8, confidence = 0.8)))

        val firstMiss = tracker.update(emptyList())
        assertEquals(1, firstMiss.single().trackId)
        assertEquals(0.4, firstMiss.single().confidence, 1e-12)
        assertTrue(tracker.isCoasting(1))

        val secondMiss = tracker.update(emptyList())
        assertEquals(0.2, secondMiss.single().confidence, 1e-12)

        val thirdMiss = tracker.update(emptyList())
        assertTrue(thirdMiss.isEmpty())
        assertFalse(tracker.isCoasting(1))
    }

    @Test
    fun aTrackThatComesBackAfterAMissIsFreshAgain() {
        val tracker = DetectionTracker()
        tracker.update(listOf(box(0.4, 0.4, 0.6, 0.8, confidence = 0.8)))
        tracker.update(emptyList())
        val back = tracker.update(listOf(box(0.41, 0.4, 0.61, 0.8, confidence = 0.7)))

        assertEquals(1, back.single().trackId)
        assertEquals(0.7, back.single().confidence, 1e-12)
        assertFalse(tracker.isCoasting(1))
    }

    @Test
    fun theClassFollowsTheNewestBox() {
        val tracker = DetectionTracker()
        tracker.update(listOf(box(0.4, 0.4, 0.6, 0.8, ObstacleClass.POLE)))
        val renamed = tracker.update(listOf(box(0.4, 0.4, 0.6, 0.8, ObstacleClass.TREE)))

        assertEquals(1, renamed.single().trackId)
        assertEquals(ObstacleClass.TREE, renamed.single().obstacleClass)
    }

    @Test
    fun overlapIsIntersectionOverUnion() {
        val a = NormalizedBox(0.0, 0.0, 0.5, 0.5)
        val b = NormalizedBox(0.25, 0.0, 0.75, 0.5)
        assertEquals(1.0 / 3.0, DetectionTracker.intersectionOverUnion(a, b), 1e-12)
        assertEquals(0.0, DetectionTracker.intersectionOverUnion(a, NormalizedBox(0.6, 0.6, 0.9, 0.9)), 0.0)
        assertNotEquals(0.0, DetectionTracker.intersectionOverUnion(a, a))
    }
}
