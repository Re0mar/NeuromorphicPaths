package com.neuromorphicpaths.pixel.timing

import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.json.JSONObject
import org.junit.Assume.assumeFalse

/**
 * The phone's timing log against an in-memory output and an injected clock.
 *
 * The fixture written at the bottom is the other half of a contract: the laptop's timing report
 * reads it with its own parser. A log that only round trips through this test proves nothing about
 * the report, so the fixture is the test that matters for the join.
 */
class FrameTimingLogTest {
    /** Lines in memory. With a gate, every write waits on it, which is how a stalled writer is staged. */
    private class MemoryOutput(private val gate: CountDownLatch? = null) : TimingOutput {
        val lines = CopyOnWriteArrayList<String>()
        var closed = false

        override fun writeLine(line: String) {
            gate?.await()
            lines.add(line)
        }

        override fun flush() = Unit

        override fun close() {
            closed = true
        }
    }

    private val session = TimingRecord.Session(startedWall = "2026-10-04T10:12:03Z", device = "Pixel 8", buildType = "release")

    private fun counterClock(start: Long = 1_000L, step: Long = 10L): () -> Long {
        val next = AtomicLong(start)
        return { next.getAndAdd(step) }
    }

    private fun typesOf(lines: List<String>): List<String> = lines.map { JSONObject(it).getString(TimingRecord.TYPE) }

    @Test
    fun theFirstLineIsTheSessionRecord() {
        val output = MemoryOutput()
        val log = FrameTimingLog(output, session, counterClock())
        log.frameHandled(5L)
        log.close()

        val first = JSONObject(output.lines.first())
        assertEquals(TimingRecord.TYPE_SESSION, first.getString(TimingRecord.TYPE))
        assertEquals(TimingRecord.SCHEMA_VERSION, first.getInt(TimingRecord.SCHEMA))
        assertEquals("Pixel 8", first.getString(TimingRecord.DEVICE))
        assertEquals("release", first.getString(TimingRecord.BUILD_TYPE))
        assertTrue(output.closed, "close must close the output")
    }

    @Test
    fun aHandledAndSentFrameWritesTwoLinesWithTheInjectedClock() {
        val output = MemoryOutput()
        val log = FrameTimingLog(output, session, counterClock(start = 1_000L, step = 10L))
        log.frameHandled(123_456_789_012_345L)
        log.frameSent(123_456_789_012_345L)
        log.close()

        assertEquals(
            listOf(
                """{"type":"frame","frame_ns":123456789012345,"handled_ns":1000}""",
                """{"type":"sent","frame_ns":123456789012345,"sent_ns":1010}""",
            ),
            output.lines.drop(1),
        )
    }

    @Test
    fun drawnIsWrittenOncePerFrameHoweverOftenItIsDrawn() {
        val output = MemoryOutput()
        val log = FrameTimingLog(output, session, counterClock())
        repeat(3) { log.pathDrawn(42L) }
        log.pathDrawn(43L)
        log.close()

        assertEquals(listOf(TimingRecord.TYPE_DRAWN, TimingRecord.TYPE_DRAWN), typesOf(output.lines.drop(1)))
    }

    @Test
    fun secondsToFrameNanosRecoversTheLongNearADayOfUptime() {
        // ARCore timestamps are boot-relative nanoseconds. A day of uptime is about 8.6e13, and
        // the range here runs to eleven days, with a random nanosecond tail on every value.
        val random = Random(20261003)
        repeat(10_000) {
            val original = random.nextLong(10_000_000_000_000L, 1_000_000_000_000_000L)
            val seconds = original / 1.0e9
            assertEquals(original, TimingRecord.frameNanosFromSeconds(seconds), "lost precision at $original")
        }
    }

    @Test
    fun theFixtureRegeneratesByteIdentical() {
        val first = fixtureBytes()
        val second = fixtureBytes()

        assertTrue(first.isNotEmpty())
        assertTrue(first.contentEquals(second), "two runs of the same calls wrote different bytes")
    }

    @Test
    fun theCommittedFixtureIsWhatThisCodeWrites() {
        // Skipped on the run that regenerates the committed file, which writes it rather than reads it.
        val committed = File(System.getProperty("committedTimingFixturePath") ?: "../../server/tests/fixtures/pixel_app_timing.jsonl")
        val target = File(System.getProperty("timingFixturePath") ?: "build/pixel_app_timing.jsonl")
        assumeFalse(committed.canonicalFile == target.canonicalFile)

        assertTrue(committed.isFile, "no committed fixture at ${committed.absolutePath}. Regenerate it with -PtimingFixturePath")
        assertTrue(
            committed.readBytes().contentEquals(fixtureBytes()),
            "the committed fixture is stale. Regenerate it with -PtimingFixturePath and commit it",
        )
    }

    @Test
    fun writesTheFixtureTheLaptopReads() {
        // The laptop's timing report test reads this file. Regenerate it by running this test with
        // -PtimingFixturePath pointing at server/tests/fixtures, and commit it with the change.
        val target = File(System.getProperty("timingFixturePath") ?: "build/pixel_app_timing.jsonl")
        target.parentFile?.mkdirs()
        target.writeBytes(fixtureBytes())
        assertTrue(target.length() > 0)
    }

    @Test
    fun theLogStopsAtMaxBytesWithOneTruncatedLine() {
        val output = MemoryOutput()
        val log = FrameTimingLog(output, session, counterClock(), maxBytes = 1_024L)
        repeat(200) { log.frameHandled(it.toLong()) }
        log.close()

        val types = typesOf(output.lines)
        assertEquals(TimingRecord.TYPE_TRUNCATED, types.last())
        assertEquals(1, types.count { it == TimingRecord.TYPE_TRUNCATED })
        val bytes = output.lines.sumOf { it.toByteArray(Charsets.UTF_8).size + 1 }
        assertTrue(bytes <= 1_024, "the log wrote $bytes bytes past its cap of 1024")
    }

    @Test
    fun aFullQueueDropsAndWritesALostLineRatherThanBlocking() {
        // The writer blocks on its first line, the session, so nothing leaves the queue. Two fit,
        // and the other three must be counted and dropped without the caller waiting.
        val gate = CountDownLatch(1)
        val output = MemoryOutput(gate)
        val log = FrameTimingLog(output, session, counterClock(), queueCapacity = 2)
        val started = System.nanoTime()
        try {
            repeat(5) { log.frameHandled(it.toLong()) }
            val elapsedMillis = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started)
            assertTrue(elapsedMillis < 500, "recording blocked for $elapsedMillis ms on a stalled writer")
        } finally {
            gate.countDown()
        }
        log.close()

        val records = output.lines.drop(1).map { JSONObject(it) }
        val lost = records.single { it.getString(TimingRecord.TYPE) == TimingRecord.TYPE_LOST }
        assertEquals(3, lost.getInt(TimingRecord.COUNT))
        assertEquals(2, records.count { it.getString(TimingRecord.TYPE) == TimingRecord.TYPE_FRAME })
    }

    @Test
    fun closeReturnsWithinItsDeadlineWhenTheWriterIsStuck() {
        val gate = CountDownLatch(1)
        val output = MemoryOutput(gate)
        val log = FrameTimingLog(output, session, counterClock(), closeTimeoutMillis = 200L)
        try {
            log.frameHandled(1L)
            val started = System.nanoTime()
            log.close()
            val elapsedMillis = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started)

            assertTrue(elapsedMillis < 1_500, "close waited $elapsedMillis ms on a stuck writer")
        } finally {
            gate.countDown()
        }
    }

    @Test
    fun aRecordArrivingWhileCloseDrainsIsNotWritten() {
        // After close returns the writer is gone, so a late record could never be written anyway.
        // The window that matters is while close is still waiting on the writer: a record that
        // slipped in then would land after what the caller thought was the last line.
        val gate = CountDownLatch(1)
        val output = MemoryOutput(gate)
        val log = FrameTimingLog(output, session, counterClock(), closeTimeoutMillis = 5_000L)
        val closer = kotlin.concurrent.thread { log.close() }
        try {
            // Long enough for close to have marked the log closing and started its wait.
            Thread.sleep(200)
            log.frameHandled(1L)
        } finally {
            gate.countDown()
        }
        closer.join(5_000)

        assertTrue(!closer.isAlive, "close never returned")
        assertEquals(listOf(TimingRecord.TYPE_SESSION), typesOf(output.lines))
    }

    @Test
    fun aStorageFailureStopsTheLogWithoutKillingTheProcess() {
        // On Android an exception that escapes any thread ends the app. A full or vanished disk
        // must cost the measurement, never the walker's display.
        val failing = object : TimingOutput {
            override fun writeLine(line: String) {
                throw java.io.IOException("no space left on device")
            }

            override fun flush() = Unit

            override fun close() = Unit
        }
        val escaped = CopyOnWriteArrayList<Throwable>()
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { _, thrown -> escaped.add(thrown) }
        try {
            val log = FrameTimingLog(failing, session, counterClock())
            log.frameHandled(1L)
            log.close()
        } finally {
            Thread.setDefaultUncaughtExceptionHandler(previous)
        }

        assertTrue(escaped.isEmpty(), "the writer let $escaped escape its thread")
    }

    @Test
    fun aMaxBytesTooSmallForTheSessionLineIsRefused() {
        val refused = runCatching { FrameTimingLog(MemoryOutput(), session, counterClock(), maxBytes = 100L) }

        assertTrue(refused.isFailure, "a cap smaller than one session line cannot hold a log")
    }

    /**
     * One published frame drawn twice, one frame handled and dropped, and one handled with no depth
     * so it was never sent. The frame values are ARCore-sized nanoseconds about a day and a half
     * into uptime, 33 ms apart. The clock is a counter, so the bytes never change between runs, and
     * its 20 ms step keeps every handled stamp after its own frame, as a real phone's would be.
     */
    private fun fixtureBytes(): ByteArray {
        val output = MemoryOutput()
        val log = FrameTimingLog(output, session, counterClock(start = 123_456_800_000_000L, step = 20_000_000L))
        val published = 123_456_789_012_345L
        val dropped = published + 33_333_333L
        val noDepth = dropped + 33_333_333L
        log.frameHandled(published)
        log.frameSent(published)
        log.frameHandled(dropped)
        log.frameDropped(dropped)
        log.frameHandled(noDepth)
        log.pathReceived(published)
        log.pathDrawn(published)
        log.pathDrawn(published)
        log.close()

        val bytes = ByteArrayOutputStream()
        output.lines.forEach { bytes.write("$it\n".toByteArray(Charsets.UTF_8)) }
        return bytes.toByteArray()
    }
}
