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
 * @param openWalk Opens the recording. Called once, on the replay's own thread.
 * @param nanoTime A monotonic clock, for pacing. A parameter so a test can see what it does.
 */
class WalkReplayer(
    private val host: String,
    private val port: Int,
    private val openWalk: () -> InputStream,
    private val onStatus: (ReplayStatus) -> Unit,
    private val nanoTime: () -> Long = System::nanoTime,
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
                    onStatus(ReplayStatus.Finished(sent, reader.truncatedBytes))
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

    private companion object {
        const val CONNECT_TIMEOUT_MILLIS = 3_000
        // The probe waits this long for a byte that should never come. Short against a frame interval.
        const val PEER_PROBE_MILLIS = 2
        const val NANOS_PER_SECOND = 1_000_000_000.0
        const val NANOS_PER_MILLI = 1_000_000L
    }
}

/** What the screen shows about a replay. */
sealed interface ReplayStatus {
    data class Connecting(val host: String, val port: Int) : ReplayStatus

    data class Replaying(val host: String, val port: Int, val framesSent: Int) : ReplayStatus

    data class Finished(val framesSent: Int, val truncatedBytes: Int) : ReplayStatus

    data class Failed(val reason: String, val framesSent: Int) : ReplayStatus
}
