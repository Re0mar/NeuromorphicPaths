package com.neuromorphicpaths.pixel.wire

import org.json.JSONException
import java.io.BufferedOutputStream
import java.io.IOException
import java.io.OutputStream
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread

/**
 * Records a walk to a file on the phone, with nothing connected.
 *
 * The file is the wire stream itself, every frame exactly as [FrameEncoder] would send it, so a
 * recording replays to the laptop through the same path a live walk takes. The AR thread only
 * hands a frame over. Encoding and writing happen on this recorder's own thread, behind a bounded
 * queue. If the queue is full the frame is dropped and counted, so the AR thread never waits and
 * a dropped frame is never silent.
 *
 * @param onProgress Called from the recorder's thread after each frame is written, and once more
 *   when the recording ends or fails.
 */
class WalkRecorder(
    private val output: OutputStream,
    private val onProgress: (RecordingProgress) -> Unit = {},
    capacityFrames: Int = DEFAULT_CAPACITY_FRAMES,
) {
    private sealed interface Item {
        class Frame(val message: DepthMessage) : Item

        data object End : Item
    }

    private val queue = ArrayBlockingQueue<Item>(capacityFrames)
    private val written = AtomicInteger(0)
    private val dropped = AtomicInteger(0)
    private val unencodable = AtomicInteger(0)
    private val bytes = AtomicLong(0)
    // Held while a frame is queued and while the recording is shut, so once stop() or a failure has
    // shut it no frame can land behind the end marker unwritten and uncounted.
    private val door = Any()
    private var accepting = false
    @Volatile private var failure: String? = null
    private var worker: Thread? = null

    /** Where the recording stands now. */
    val progress: RecordingProgress
        get() = RecordingProgress(written.get(), dropped.get(), bytes.get(), failure, unencodable.get())

    fun start() {
        if (worker != null) return
        synchronized(door) { accepting = true }
        worker = thread(name = "walk-recorder", isDaemon = true) { loop() }
    }

    /** Hand over a frame. Never waits on the writer. After [stop], or once writing has failed, it is ignored. */
    fun offer(message: DepthMessage) {
        synchronized(door) {
            if (!accepting) return
            if (!queue.offer(Item.Frame(message))) {
                dropped.incrementAndGet()
            }
        }
    }

    /** Write what is still queued, close the file, and wait for that to finish. */
    fun stop() {
        val running = worker ?: return
        shut()
        // put waits for room. It can't wait forever, because a writer that dies shuts the door and
        // empties the queue on its way out, and onDestroy calls this on the UI thread.
        queue.put(Item.End)
        running.join()
        worker = null
    }

    private fun shut() {
        synchronized(door) { accepting = false }
    }

    private fun loop() {
        try {
            BufferedOutputStream(output).use { out ->
                while (true) {
                    when (val item = queue.take()) {
                        Item.End -> break
                        is Item.Frame -> write(out, item.message)
                    }
                }
            }
        } catch (unwritable: IOException) {
            // A full disk, or storage that went away. The frames written so far are a good
            // recording. Later ones are refused rather than queued for a file that can't take them.
            failure = unwritable.message ?: unwritable.javaClass.simpleName
        } catch (interrupted: InterruptedException) {
            Thread.currentThread().interrupt()
        } catch (unexpected: RuntimeException) {
            // Nobody predicted this one. Not rethrown, because an uncaught exception here takes the
            // whole app down, and the frames already written are still a good recording.
            failure = "UNEXPECTED ${unexpected.javaClass.simpleName}, may need a handler: ${unexpected.message}"
        } finally {
            shut()
            // Frames still queued when the writer stopped early were never written. Counted, not lost.
            dropped.addAndGet(queue.count { it is Item.Frame })
            queue.clear()
        }
        onProgress(progress)
    }

    private fun write(out: BufferedOutputStream, message: DepthMessage) {
        val encoded = try {
            FrameEncoder.encode(message)
        } catch (unencodable: JSONException) {
            // A header value JSON can't hold, such as a NaN in the pose. Costs this frame only.
            this.unencodable.incrementAndGet()
            onProgress(progress)
            return
        }
        out.write(encoded)
        written.incrementAndGet()
        bytes.addAndGet(encoded.size.toLong())
        onProgress(progress)
    }

    private companion object {
        // Ten seconds of frames at 30 a second. A phone that falls this far behind writing has a
        // problem the count on screen should show, not a queue that grows until it runs out of memory.
        const val DEFAULT_CAPACITY_FRAMES = 300
    }
}

/**
 * Frames written, frames dropped because the writer fell behind or stopped early, bytes in the
 * file, why it stopped if it failed, and frames left out because they couldn't be encoded.
 */
data class RecordingProgress(
    val framesWritten: Int,
    val framesDropped: Int,
    val bytesWritten: Long,
    val failure: String?,
    val framesUnencodable: Int,
)
