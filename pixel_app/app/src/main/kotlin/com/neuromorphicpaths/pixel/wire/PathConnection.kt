package com.neuromorphicpaths.pixel.wire

import android.os.SystemClock
import android.util.Log
import java.io.DataInputStream
import java.io.IOException
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread

/**
 * A path as it arrived: the message, and when, on this phone's own clock.
 *
 * The message's `timestamp_seconds` is this phone's own ARCore timestamp for the depth frame the
 * path was planned from, handed back unchanged by the laptop. The timing log uses it to tie the
 * path to its frame. The arrow's age is measured from [receivedAtMillis] instead, because the age
 * text is about how long ago the path arrived, and that is what makes a stale arrow visibly stale.
 */
class ReceivedPath(val message: PathMessage, val receivedAtMillis: Long)

/**
 * The second connection to the laptop, on the path port, reading paths as they are planned.
 *
 * The same shape as [LaptopConnection]: one daemon thread, connect, work, reconnect after a pause
 * when the socket fails. The work is a read loop rather than a write slot, which is why the two
 * are separate classes rather than one with a pluggable body. No idle probe: this socket is read
 * continuously, so a closed far end is seen at once. A message that will not decode costs that
 * one message, is counted and shown, and the loop keeps reading.
 *
 * @param nowMillis the clock each received path is stamped with. The device's elapsed-realtime
 * clock by default. Tests pass a counter.
 */
class PathConnection(
    private val host: String,
    private val port: Int,
    private val onStatus: (PathConnectionStatus) -> Unit,
    private val onPath: (ReceivedPath) -> Unit,
    private val nowMillis: () -> Long = { SystemClock.elapsedRealtime() },
) {
    private val received = AtomicInteger(0)
    private val refused = AtomicInteger(0)
    @Volatile private var running = false
    @Volatile private var socket: Socket? = null
    private var worker: Thread? = null

    /** Paths decoded and handed to the callback so far. */
    val pathsReceived: Int get() = received.get()

    /** Messages that arrived and would not decode. */
    val pathsRefused: Int get() = refused.get()

    fun start() {
        if (running) return
        running = true
        worker = thread(name = "path-connection", isDaemon = true) { loop() }
    }

    /** Ends the thread. The socket is closed here because a thread blocked in a read ignores interrupts. */
    fun stop() {
        running = false
        closeSocket()
        worker?.interrupt()
        worker = null
    }

    private fun loop() {
        while (running) {
            var lastRefusal: String? = null
            try {
                onStatus(PathConnectionStatus.Connecting(host, port))
                // No apply block on the socket on purpose. Inside one, a bare `port` is the
                // socket's own getPort(), 0 before connecting.
                val opened = Socket()
                socket = opened
                opened.tcpNoDelay = true
                opened.connect(InetSocketAddress(host, port), CONNECT_TIMEOUT_MILLIS)
                val input = DataInputStream(opened.getInputStream().buffered())
                onStatus(PathConnectionStatus.Connected(host, port, received.get(), refused.get(), lastRefusal))
                while (running) {
                    val payload = PathDecoder.readFramed(input)
                    try {
                        val message = PathDecoder.decode(payload)
                        received.incrementAndGet()
                        onPath(ReceivedPath(message, nowMillis()))
                    } catch (bad: PathDecodeException) {
                        // One corrupt path on a hotspot is normal and costs that one path.
                        refused.incrementAndGet()
                        lastRefusal = bad.message
                    }
                    onStatus(PathConnectionStatus.Connected(host, port, received.get(), refused.get(), lastRefusal))
                }
            } catch (interrupted: InterruptedException) {
                // stop() interrupted the sleep. Leaving the loop is the whole point.
                Thread.currentThread().interrupt()
                return
            } catch (network: IOException) {
                // Refused, reset, end of stream, or a prefix that says the stream is out of step.
                // All of them mean reconnect. Unless stop() closed the socket under the read, in
                // which case the exception is ours and the loop is over.
                if (!running) return
                Log.w(TAG, "path connection lost, retrying in $RECONNECT_PAUSE_MILLIS ms", network)
                onStatus(PathConnectionStatus.Disconnected(network.message ?: network.javaClass.simpleName))
                try {
                    Thread.sleep(RECONNECT_PAUSE_MILLIS)
                } catch (interrupted: InterruptedException) {
                    Thread.currentThread().interrupt()
                    return
                }
            } finally {
                closeSocket()
            }
        }
    }

    private fun closeSocket() {
        val open = socket ?: return
        socket = null
        try {
            open.close()
        } catch (closing: IOException) {
            // Nothing left to do with a socket that will not close.
        }
    }

    private companion object {
        const val TAG = "PathConnection"
        const val CONNECT_TIMEOUT_MILLIS = 3_000
        const val RECONNECT_PAUSE_MILLIS = 1_000L
    }
}

/** What the screen shows about the path connection. Named after [ConnectionStatus], the depth one's. */
sealed interface PathConnectionStatus {
    data class Connecting(val host: String, val port: Int) : PathConnectionStatus
    data class Connected(
        val host: String,
        val port: Int,
        val received: Int,
        val refused: Int,
        val lastRefusal: String?,
    ) : PathConnectionStatus
    data class Disconnected(val reason: String) : PathConnectionStatus
}
