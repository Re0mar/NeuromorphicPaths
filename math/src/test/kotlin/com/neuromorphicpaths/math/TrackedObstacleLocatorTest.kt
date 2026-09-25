package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.CameraIntrinsics
import com.neuromorphicpaths.core.CameraPose
import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.NormalizedBox
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.ObstacleClass
import com.neuromorphicpaths.core.ObstacleLocator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class TrackedObstacleLocatorTest {

    private val pose = CameraPose(pitchRadians = 0.3, yawRadians = 0.0, rollRadians = 0.0, heightMeters = 1.4)
    private val intrinsics = CameraIntrinsics(horizontalFovRadians = Math.toRadians(60.0))

    private fun frameAt(seconds: Double) = Frame(
        timestampNanos = (seconds * 1e9).toLong(),
        width = 2,
        height = 2,
        rgba = ByteArray(2 * 2 * Frame.BYTES_PER_PIXEL),
        pose = pose,
        intrinsics = intrinsics,
    )

    /** Places every detection at the range its box's bottom edge encodes, so tests can dictate ranges. */
    private object RangeFromBottomEdge : ObstacleLocator {
        override fun locate(detections: List<Detection>, frame: Frame): List<Obstacle> =
            detections.map { Obstacle(it, bearingRadians = 0.0, rangeMeters = it.box.bottom * 10.0, closingSpeedMetersPerSecond = null) }
    }

    // The box grows from the top with the range, so boxes for nearby ranges still overlap and
    // the tracker keeps them on one track.
    private fun detectionAtRange(rangeMeters: Double) =
        Detection(ObstacleClass.PERSON, 0.9, NormalizedBox(0.4, 0.0, 0.6, rangeMeters / 10.0))

    @Test
    fun rangeShrinkingOverThreeFramesGivesTheClosingSpeed() {
        val tracker = DetectionTracker()
        val locator = TrackedObstacleLocator(inner = RangeFromBottomEdge, tracker = tracker)

        // 4.0, 3.3, 2.6 m at half-second steps: 1.4 m/s of closing.
        var obstacles = locator.locate(tracker.update(listOf(detectionAtRange(4.0))), frameAt(0.0))
        assertNull(obstacles.single().closingSpeedMetersPerSecond)
        obstacles = locator.locate(tracker.update(listOf(detectionAtRange(3.3))), frameAt(0.5))
        assertNull(obstacles.single().closingSpeedMetersPerSecond)
        obstacles = locator.locate(tracker.update(listOf(detectionAtRange(2.6))), frameAt(1.0))

        assertEquals(1.4, obstacles.single().closingSpeedMetersPerSecond!!, 1e-9)
    }

    @Test
    fun aCoastingBoxIsNotASampleAndOldSamplesFallOutOfTheWindow() {
        val tracker = DetectionTracker()
        val locator = TrackedObstacleLocator(inner = RangeFromBottomEdge, tracker = tracker, windowSeconds = 2.0)

        locator.locate(tracker.update(listOf(detectionAtRange(4.0))), frameAt(0.0))
        locator.locate(tracker.update(listOf(detectionAtRange(3.5))), frameAt(0.5))
        // A missed frame: the tracker coasts the last box, whose range would flatten the slope.
        val coasted = locator.locate(tracker.update(emptyList()), frameAt(1.0))
        assertNull(coasted.single().closingSpeedMetersPerSecond)
        val back = locator.locate(tracker.update(listOf(detectionAtRange(2.5))), frameAt(1.5))
        assertEquals(1.0, back.single().closingSpeedMetersPerSecond!!, 1e-9)

        // 3 s later the whole history has aged out, so the estimate starts over.
        val later = locator.locate(tracker.update(listOf(detectionAtRange(2.0))), frameAt(4.5))
        assertNull(later.single().closingSpeedMetersPerSecond)
    }

    @Test
    fun anObjectWalkingAwayHasANegativeClosingSpeed() {
        val tracker = DetectionTracker()
        val locator = TrackedObstacleLocator(inner = RangeFromBottomEdge, tracker = tracker)

        locator.locate(tracker.update(listOf(detectionAtRange(2.0))), frameAt(0.0))
        locator.locate(tracker.update(listOf(detectionAtRange(2.5))), frameAt(0.5))
        val obstacles = locator.locate(tracker.update(listOf(detectionAtRange(3.0))), frameAt(1.0))

        assertEquals(-1.0, obstacles.single().closingSpeedMetersPerSecond!!, 1e-9)
    }

    @Test
    fun anUntrackedDetectionPassesThroughUnchanged() {
        val locator = TrackedObstacleLocator(inner = RangeFromBottomEdge, tracker = DetectionTracker())
        val obstacles = locator.locate(listOf(detectionAtRange(3.0)), frameAt(0.0))

        assertNull(obstacles.single().detection.trackId)
        assertNull(obstacles.single().closingSpeedMetersPerSecond)
    }
}
