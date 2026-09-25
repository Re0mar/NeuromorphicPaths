package com.neuromorphicpaths.input

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import kotlin.math.PI

private const val SECOND_NANOS = 1_000_000_000L

class OrientationLogTest {

    // Three samples a second apart, in Sensor Logger's column order, upright phone tipped 18 degrees down.
    private val csv = """
        time,seconds_elapsed,qz,qy,qx,qw,roll,pitch,yaw
        1000000000,0.0,0,0,0,1,-0.15,-1.2566,1.77
        2000000000,1.0,0,0,0,1,-0.10,-1.2566,1.80
        3000000000,2.0,0,0,0,1,-0.05,-1.0000,1.90
    """.trimIndent()

    @Test
    fun parsesByColumnNameAndKeepsTimeOrder() {
        val log = OrientationLog.parse(csv)
        assertEquals(3, log.samples.size)
        assertEquals(SECOND_NANOS, log.firstEpochNanos)
        assertEquals(3 * SECOND_NANOS, log.lastEpochNanos)
        assertEquals(1.80, log.samples[1].yawRadians, 1e-12)
    }

    @Test
    fun nearestSampleWinsAndTheEndsClamp() {
        val log = OrientationLog.parse(csv)
        assertEquals(1.77, log.sampleAt(0L).yawRadians, 1e-12)
        assertEquals(1.77, log.sampleAt(1_400_000_000L).yawRadians, 1e-12)
        assertEquals(1.80, log.sampleAt(1_600_000_000L).yawRadians, 1e-12)
        assertEquals(1.90, log.sampleAt(99 * SECOND_NANOS).yawRadians, 1e-12)
    }

    @Test
    fun loggedPitchBecomesCameraPitchDown() {
        val log = OrientationLog.parse(csv)
        val pose = log.poseAt(SECOND_NANOS, heightMeters = 1.4)

        // -1.2566 rad is -72 degrees. An upright phone tipped 18 degrees toward the ground.
        assertEquals(18.0, Math.toDegrees(pose.pitchRadians), 0.01)
        assertEquals(-0.15, pose.rollRadians, 1e-12)
        assertEquals(1.4, pose.heightMeters, 0.0)
        assertEquals(0.0, pose.yawRadians, 0.0)
        assertEquals(1.77, log.azimuthAt(SECOND_NANOS), 1e-12)
    }

    @Test
    fun aLogWithoutTheAnglesIsRefused() {
        assertThrows(IllegalArgumentException::class.java) { OrientationLog.parse("time,seconds_elapsed\n1,0\n") }
        assertThrows(IllegalArgumentException::class.java) { OrientationLog.parse("time,roll,pitch,yaw\n") }
    }

    @Test
    fun theOutdoorRecordingStartsFourPointSixSecondsAfterItsLogger() {
        // The container says the take ended at 09:35:22 UTC and ran 229.4 s. Sensor Logger started
        // at epoch 1790328687990 ms. The gap is what the earlier by-hand alignment found.
        val start = RecordingClock.videoStartEpochMillis("20260925T093522.000Z", durationMillis = 229_400)
        assertEquals(1_790_328_692_600L, start)
        assertEquals(4_610L, start - 1_790_328_687_990L)
    }

    @Test
    fun theDateParsesWithOrWithoutMilliseconds() {
        assertEquals(
            RecordingClock.videoStartEpochMillis("20260925T093522.000Z", 0L),
            RecordingClock.videoStartEpochMillis("20260925T093522Z", 0L),
        )
    }

    @Test
    fun upsideDownIsStillAnAngle() {
        val log = OrientationLog(listOf(OrientationSample(0L, pitchRadians = -PI / 2, rollRadians = 0.0, yawRadians = 0.0)))
        assertEquals(0.0, log.poseAt(0L, 1.0).pitchRadians, 1e-12)
    }
}
