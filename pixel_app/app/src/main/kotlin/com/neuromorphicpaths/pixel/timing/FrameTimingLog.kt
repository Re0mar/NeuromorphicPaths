package com.neuromorphicpaths.pixel.timing

import android.os.SystemClock
import android.util.Log
import java.io.IOException
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread

/**
 * The phone's half of the frame-to-arrow measurement: a JSON-lines log of when each frame was
 * handled, sent or dropped, and when its path came back and was first drawn.
 *
 * The callers are the GL thread, the sender, the path reader and the draw pass, and none of them
 * may wait on a file. So each call reads the clock, offers one record to a bounded queue and
 * returns. A writer thread of its own drains the queue to [output]. A full queue drops the record
 * rather than blocking, and the count is written as a `lost` line, so a gap in the log says it is
 * a gap.
 *
 * @param session the first line, built by the caller so the tests can fix it and the fixture stays byte-identical
 * @param nowNanos the clock every stamp is read from. The device's elapsed-realtime clock by default,
 * the same one the path connection and the overlay use. Tests pass a counter
 * @param maxBytes the file's size cap, newlines included. At the cap one `truncated` line is written and nothing after it
 * @param queueCapacity records waiting for the writer before new ones are dropped
 * @param closeTimeoutMillis the longest [close] waits for the writer to finish before giving up on it
 */
class FrameTimingLog(
    private val output: TimingOutput,
    private val session: TimingRecord.Session,
    private val nowNanos: () -> Long = { SystemClock.elapsedRealtimeNanos() },
    private val maxBytes: Long = DEFAULT_MAX_BYTES,
    queueCapacity: Int = DEFAULT_QUEUE_CAPACITY,
    private val closeTimeoutMillis: Long = DEFAULT_CLOSE_TIMEOUT_MILLIS,
) : TimingRecorder {
    private val queue = LinkedBlockingQueue<TimingRecord>(queueCapacity)
    private val lost = AtomicInteger(0)
    private val closing = AtomicBoolean(false)
    private val drawnRecently = LinkedHashSet<Long>()
    private val writer: Thread

    init {
        // Room for the session line and the truncated line at least. Anything smaller cannot hold a log.
        require(maxBytes >= MIN_MAX_BYTES) { "maxBytes must be at least $MIN_MAX_BYTES, got $maxBytes" }
        writer = thread(name = "timing-writer", isDaemon = true) { drain() }
    }

    // Keyed the way every other line is, from the seconds the frame travels in, so the five lines of
    // one frame always share a key. See TimingRecord.frameNanosFromArCore.
    override fun frameHandled(frameNanos: Long) = record(TimingRecord.Frame(TimingRecord.frameNanosFromArCore(frameNanos), nowNanos()))

    override fun frameSent(frameNanos: Long) = record(TimingRecord.Sent(frameNanos, nowNanos()))

    override fun frameDropped(frameNanos: Long) = record(TimingRecord.Dropped(frameNanos))

    override fun pathReceived(frameNanos: Long) = record(TimingRecord.Received(frameNanos, nowNanos()))

    /**
     * Only the first draw of a frame's path is recorded. The overlay already reports once per path
     * for as long as it stays on screen, and this catches a repeat after it leaves and comes back,
     * or a path that arrived twice.
     */
    override fun pathDrawn(frameNanos: Long) {
        val drawnNanos = nowNanos()
        val first = synchronized(drawnRecently) {
            val added = drawnRecently.add(frameNanos)
            if (drawnRecently.size > DRAWN_MEMORY) drawnRecently.remove(drawnRecently.first())
            added
        }
        if (first) record(TimingRecord.Drawn(frameNanos, drawnNanos))
    }

    /** Lets the writer finish what is queued, up to [closeTimeoutMillis], then closes the output. */
    override fun close() {
        if (!closing.compareAndSet(false, true)) return
        writer.join(closeTimeoutMillis)
        if (writer.isAlive) {
            // The output is stuck, most likely on storage. Closing it under a writer that is still
            // inside a write would race, so it is left to the daemon thread and the process.
            Log.w(TAG, "timing writer did not finish within $closeTimeoutMillis ms, leaving it")
            return
        }
        output.close()
    }

    private fun record(timingRecord: TimingRecord) {
        // A record that arrives while close drains is counted, not written. The writer's last
        // `lost` line includes it if the writer is still running. Neither connection's stop()
        // waits for its thread, so a send finishing as the app closes ends up here.
        if (closing.get() || !queue.offer(timingRecord)) lost.incrementAndGet()
    }

    private fun drain() {
        var bytesWritten = 0L
        var truncated = false

        fun emit(line: TimingRecord) {
            if (truncated) return
            val text = line.toJsonLine()
            val size = text.toByteArray(Charsets.UTF_8).size + 1L
            if (bytesWritten + size + TRUNCATED_RESERVE_BYTES > maxBytes) {
                val note = TimingRecord.Truncated(bytesWritten).toJsonLine()
                output.writeLine(note)
                bytesWritten += note.toByteArray(Charsets.UTF_8).size + 1L
                truncated = true
                Log.w(TAG, "timing log reached its $maxBytes byte cap and stopped")
                return
            }
            output.writeLine(text)
            bytesWritten += size
        }

        fun emitLost() {
            val missed = lost.getAndSet(0)
            if (missed > 0) emit(TimingRecord.Lost(missed))
        }

        try {
            emit(session)
            while (!closing.get() || queue.isNotEmpty()) {
                val next = queue.poll(POLL_MILLIS, TimeUnit.MILLISECONDS)
                if (next == null) {
                    emitLost()
                    output.flush()
                    continue
                }
                emitLost()
                emit(next)
                // Flushed whenever the queue runs dry, so a walk that ends with the app killed
                // still leaves everything but the last moment on disk.
                if (queue.isEmpty()) output.flush()
            }
            emitLost()
            output.flush()
        } catch (interrupted: InterruptedException) {
            // Nothing in the app interrupts this thread. If something ever does, it is being told
            // to stop, and the lines already written are flushed up to the last empty queue.
            Thread.currentThread().interrupt()
        } catch (failed: IOException) {
            // Storage refused the write. The measurement is lost, the app must keep working.
            Log.w(TAG, "timing log stopped writing", failed)
        }
    }

    companion object {
        const val DEFAULT_MAX_BYTES = 20_000_000L
        const val DEFAULT_QUEUE_CAPACITY = 1024
        const val DEFAULT_CLOSE_TIMEOUT_MILLIS = 2_000L
        private const val MIN_MAX_BYTES = 1_024L
        // Longer than any truncated line, so the note always fits under the cap.
        private const val TRUNCATED_RESERVE_BYTES = 64L
        private const val POLL_MILLIS = 50L
        // Paths arrive at up to the frame rate, about 30 a second, so this is about 8 s of history.
        // A repeat comes within a frame or two, well inside that.
        private const val DRAWN_MEMORY = 256
        private const val TAG = "FrameTimingLog"
    }
}
