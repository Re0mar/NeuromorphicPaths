package com.neuromorphicpaths.pixel.wire

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.OutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/** Recording a walk on its own thread, read back with [WalkFileReader]. */
class WalkRecorderTest {
    private fun message(stamp: Double) = DepthMessage(
        timestampSeconds = stamp,
        depthMillimeters = shortArrayOf(1, 2, 3, 4),
        rows = 2,
        columns = 2,
        intrinsics = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        orientationWxyz = doubleArrayOf(1.0, 0.0, 0.0, 0.0),
        positionXyz = null,
        groundPlane = null,
    )

    /** A frame bigger than the recorder's 8 KB buffer, so writing it goes straight to the file. */
    private fun largeMessage(stamp: Double) = message(stamp).copy(depthMillimeters = ShortArray(80 * 80), rows = 80, columns = 80)

    private fun readBack(bytes: ByteArray): List<RecordedMessage> {
        val reader = WalkFileReader(ByteArrayInputStream(bytes))
        return generateSequence { reader.next() }.toList()
    }

    /** An output whose first write waits until the test lets it go, so the queue behind it fills. */
    private class HeldOutput(private val target: OutputStream) : OutputStream() {
        val firstWriteStarted = CountDownLatch(1)
        val release = CountDownLatch(1)

        override fun write(byte: Int) = write(byteArrayOf(byte.toByte()), 0, 1)

        override fun write(buffer: ByteArray, offset: Int, length: Int) {
            firstWriteStarted.countDown()
            release.await(5, TimeUnit.SECONDS)
            target.write(buffer, offset, length)
        }
    }

    @Test
    fun everyOfferedFrameIsWrittenInOrder() {
        val file = ByteArrayOutputStream()
        val recorder = WalkRecorder(file)
        recorder.start()
        val stamps = (0 until 50).map { 100.0 + it / 30.0 }
        stamps.forEach { recorder.offer(message(it)) }
        recorder.stop()

        val messages = readBack(file.toByteArray())
        assertEquals(stamps, messages.map { it.timestampSeconds })
        assertContentEquals(FrameEncoder.encode(message(stamps[7])), messages[7].wireBytes)
        assertEquals(RecordingProgress(50, 0, file.size().toLong(), null), recorder.progress)
    }

    @Test
    fun fullQueueDropsAndCountsWithoutWaiting() {
        val file = ByteArrayOutputStream()
        val held = HeldOutput(file)
        val recorder = WalkRecorder(held, capacityFrames = 2)
        recorder.start()
        // The first frame goes to the writer, which then waits inside the file write.
        recorder.offer(largeMessage(0.0))
        assertTrue(held.firstWriteStarted.await(5, TimeUnit.SECONDS), "the writer never reached the file")
        val started = System.nanoTime()
        repeat(20) { recorder.offer(largeMessage(1.0 + it)) }
        val offeringMillis = (System.nanoTime() - started) / 1_000_000
        assertTrue(offeringMillis < 1_000, "offering 20 frames took $offeringMillis ms, the AR thread must never wait")
        // Two fit in the queue behind the stuck writer. The other eighteen are dropped and counted.
        assertEquals(18, recorder.progress.framesDropped)
        held.release.countDown()
        recorder.stop()
        assertEquals(3, recorder.progress.framesWritten)
        assertEquals(3, readBack(file.toByteArray()).size)
    }

    @Test
    fun offerAfterStopIsIgnored() {
        val file = ByteArrayOutputStream()
        val recorder = WalkRecorder(file)
        recorder.start()
        recorder.offer(message(1.0))
        recorder.stop()
        val stopped = recorder.progress
        repeat(400) { recorder.offer(message(2.0 + it)) }
        assertEquals(1, readBack(file.toByteArray()).size)
        // Not written, and not counted as dropped either. A stopped recording takes nothing.
        assertEquals(stopped, recorder.progress)
    }

    @Test
    fun aFailedWriteIsReportedAndStopsTheRecording() {
        val broken = object : OutputStream() {
            override fun write(byte: Int) = throw IOException("no space left on device")
            override fun write(buffer: ByteArray, offset: Int, length: Int) = throw IOException("no space left on device")
        }
        val recorder = WalkRecorder(broken)
        recorder.start()
        // Bigger than the buffer in front of the file, so the write really reaches it.
        recorder.offer(largeMessage(0.0))
        recorder.stop()
        val failure = recorder.progress.failure
        assertNotNull(failure)
        assertTrue(failure.contains("no space left"), failure)
    }
}
