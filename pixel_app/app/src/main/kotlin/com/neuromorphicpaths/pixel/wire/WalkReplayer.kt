package com.neuromorphicpaths.pixel.wire

import java.io.IOException
import java.io.InputStream
import java.net.InetSocketAddress
import java.net.Socket
import java.net.SocketTimeoutException
import kotlin.concurrent.thread

/**
 * Sends a recorded walk to the laptop over the depth connection, as if it were happening now.
 *
 * Every message goes unchanged and in order, paced by the timestamps it was recorded with, so the
 * laptop's `--record-to` writes the walk the phone recorded. Unlike [LaptopConnection] it never
 * drops a frame: a live sender should skip stale frames, a replay must not. A lost connection ends
 * the replay with the count sent. It doesn't reconnect and carry on, because a walk with a hole
 * in the middle is a different walk. Replay again into a fresh directory instead.
 *
 * After the last frame the phone closes its sending side and waits for the laptop to close back.
 * The laptop closes only once it has read the end of the stream, so that close is the one sign
 * every frame left the phone's send buffer and arrived.
 *
 * @param openWalk Opens the recording. Called once, on the replay's own thread.
 * @param nanoTime A monotonic clock, for pacing. A parameter so a test can see what it does.
 * @param confirmTimeoutMillis How long to wait for the laptop to close back after the last frame.
 *   A laptop still working through a backlog of frames closes late, so this is generous.
 */
class WalkReplayer(
    private val host: String,
    private val port: Int,
    private val openWalk: () -> InputStream,
    private val onStatus: (ReplayStatus) -> Unit,
    private val nanoTime: () -> Long = System::nanoTime,
    private val confirmTimeoutMillis: Long = DEFAULT_CONFIRM_TIMEOUT_MILLIS,
) {
    @Volatile private var running = false
    private var worker: Thread? = null

    fun start() {
        if (running) return
        running = true
        worker = thread(name = "walk-replayer", isDaemon = true) { replay() }
    }

    fun stop() {
        running = false
        worker?.interrupt()
        worker = null
    }

    private fun replay() {
        var sent = 0
        try {
            onStatus(ReplayStatus.Connecting(host, port))
            Socket().use { socket ->
                // No apply block, for the reason LaptopConnection gives: a bare `port` inside one is
                // the socket's own getPort().
                socket.tcpNoDelay = true
                socket.connect(InetSocketAddress(host, port), CONNECT_TIMEOUT_MILLIS)
                socket.soTimeout = PEER_PROBE_MILLIS
                val output = socket.getOutputStream()
                val input = socket.getInputStream()
                WalkFileReader(openWalk()).use { reader ->
                    var firstTimestampSeconds: Double? = null
                    var startNanos = 0L
                    while (running) {
                        val message = reader.next() ?: break
                        val first = firstTimestampSeconds ?: message.timestampSeconds.also {
                            firstTimestampSeconds = it
                            startNanos = nanoTime()
                        }
                        val dueNanos = startNanos + ((message.timestampSeconds - first) * NANOS_PER_SECOND).toLong()
                        val waitNanos = dueNanos - nanoTime()
                        if (waitNanos > 0) Thread.sleep(waitNanos / NANOS_PER_MILLI, (waitNanos % NANOS_PER_MILLI).toInt())
                        // The laptop never writes on this socket, so a byte or the end of the stream
                        // means it went away. Checked before each frame, because a write into a
                        // closed connection can sit in the send buffer and look like success.
                        if (laptopWentAway(input)) throw IOException("the laptop closed the connection")
                        output.write(message.wireBytes)
                        output.flush()
                        sent += 1
                        onStatus(ReplayStatus.Replaying(host, port, sent))
                    }
                    if (!running) {
                        onStatus(ReplayStatus.Failed("stopped before the end", sent))
                        return
                    }
                    socket.shutdownOutput()
                    onStatus(ReplayStatus.Finished(sent, reader.truncatedBytes, laptopClosedBack(socket, input)))
                }
            }
        } catch (interrupted: InterruptedException) {
            // stop() during a pacing sleep. Leaving is the point.
            Thread.currentThread().interrupt()
            onStatus(ReplayStatus.Failed("stopped before the end", sent))
        } catch (unreplayable: IOException) {
            // Refused, reset, closed by the laptop, or a recording that won't read. Each one is a
            // replay the laptop didn't get whole, so it ends here and says so.
            onStatus(ReplayStatus.Failed(unreplayable.message ?: unreplayable.javaClass.simpleName, sent))
        } finally {
            running = false
        }
    }

    /** A byte or the end of the stream both mean the laptop went away. Only silence means it's there. */
    private fun laptopWentAway(input: InputStream): Boolean = try {
        input.read()
        true
    } catch (alive: SocketTimeoutException) {
        // Nothing within the probe window. The connection is up.
        false
    }

    /**
     * Whether the laptop closed its side cleanly once it had read everything. Waits in short slices
     * so stop() still ends the wait promptly.
     */
    private fun laptopClosedBack(socket: Socket, input: InputStream): Boolean {
        val deadline = System.nanoTime() + confirmTimeoutMillis * NANOS_PER_MILLI
        while (running) {
            val remainingMillis = (deadline - System.nanoTime()) / NANOS_PER_MILLI
            if (remainingMillis <= 0) return false
            socket.soTimeout = minOf(remainingMillis, CONFIRM_SLICE_MILLIS).toInt().coerceAtLeast(1)
            try {
                // The end of the stream is the clean close. A byte is something this side never
                // asked for, so it confirms nothing.
                return input.read() == END_OF_STREAM
            } catch (stillOpen: SocketTimeoutException) {
                continue
            } catch (reset: IOException) {
                // A reset rather than a close means the laptop dropped the connection with bytes
                // still unread, so it did not get everything.
                return false
            }
        }
        return false
    }

    private companion object {
        const val CONNECT_TIMEOUT_MILLIS = 3_000
        // The probe waits this long for a byte that should never come. Short against a frame interval.
        const val PEER_PROBE_MILLIS = 2
        const val DEFAULT_CONFIRM_TIMEOUT_MILLIS = 30_000L
        const val CONFIRM_SLICE_MILLIS = 100L
        const val END_OF_STREAM = -1
        const val NANOS_PER_SECOND = 1_000_000_000.0
        const val NANOS_PER_MILLI = 1_000_000L
    }
}

/** What the screen shows about a replay. */
sealed interface ReplayStatus {
    data class Connecting(val host: String, val port: Int) : ReplayStatus

    data class Replaying(val host: String, val port: Int, val framesSent: Int) : ReplayStatus

    /** Every frame was sent. [laptopConfirmed] says whether the laptop then closed back, having read them all. */
    data class Finished(val framesSent: Int, val truncatedBytes: Int, val laptopConfirmed: Boolean) : ReplayStatus

    data class Failed(val reason: String, val framesSent: Int) : ReplayStatus
}
