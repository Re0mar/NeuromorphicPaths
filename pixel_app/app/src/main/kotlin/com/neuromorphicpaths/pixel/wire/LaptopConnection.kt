package com.neuromorphicpaths.pixel.wire

import android.util.Log
import com.neuromorphicpaths.pixel.timing.TimingRecord
import java.io.IOException
import java.net.InetSocketAddress
import java.net.Socket
import java.net.SocketTimeoutException
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

/**
 * One TCP connection to the laptop, fed from the AR thread, draining on its own.
 *
 * The AR thread must never block on the network, so [offer] only replaces the pending message
 * and returns. The sender thread takes the newest pending message, encodes it and writes it. A
 * message replaced before it was sent is counted as dropped. When the socket fails the thread
 * reconnects after a pause and keeps going, because a walk does not stop for a hotspot hiccup.
 *
 * @param onSent called on the sender thread once a frame's message has been written and flushed,
 * with the frame's ARCore timestamp in nanoseconds
 * @param onDropped called on the caller's thread of [offer] for a frame replaced before it went,
 * with that frame's ARCore timestamp in nanoseconds
 */
class LaptopConnection(
    private val host: String,
    private val port: Int,
    private val onSent: (Long) -> Unit = {},
    private val onDropped: (Long) -> Unit = {},
    private val onStatus: (ConnectionStatus) -> Unit,
) {
    private val pending = AtomicReference<DepthMessage?>(null)
    private val sent = AtomicInteger(0)
    private val dropped = AtomicInteger(0)
    @Volatile private var running = false
    private var worker: Thread? = null

    /** Messages written to the socket so far. */
    val framesSent: Int get() = sent.get()

    /** Messages replaced in the pending slot before the sender took them. */
    val framesDropped: Int get() = dropped.get()

    fun start() {
        if (running) return
        running = true
        worker = thread(name = "laptop-connection", isDaemon = true) { loop() }
    }

    fun stop() {
        running = false
        worker?.interrupt()
        worker = null
    }

    /** Hand over a frame. If the previous one has not gone yet, it is replaced and counted. */
    fun offer(message: DepthMessage) {
        val replaced = pending.getAndSet(message) ?: return
        dropped.incrementAndGet()
        onDropped(TimingRecord.frameNanosFromSeconds(replaced.timestampSeconds))
    }

    private fun loop() {
        while (running) {
            var socket: Socket? = null
            try {
                onStatus(ConnectionStatus.Connecting(host, port))
                // No apply block here on purpose. Inside one, a bare `port` is the socket's own
                // getPort(), 0 before connecting, and every connect went to 127.0.0.1:0.
                val opened = Socket()
                socket = opened
                opened.tcpNoDelay = true
                opened.connect(InetSocketAddress(host, port), CONNECT_TIMEOUT_MILLIS)
                opened.soTimeout = PEER_PROBE_MILLIS
                val output = opened.getOutputStream()
                val input = opened.getInputStream()
                onStatus(ConnectionStatus.Connected(host, port, sent.get(), dropped.get()))
                while (running) {
                    val message = pending.getAndSet(null)
                    if (message == null) {
                        // Nothing to send, so read instead. The laptop never writes on this
                        // socket, which makes a byte or the end of the stream here the far end
                        // going away. Without the probe a dead laptop was only noticed on the
                        // next frame, and a phone with no depth yet has no next frame.
                        try {
                            if (input.read() == -1) throw IOException("laptop closed the connection")
                        } catch (alive: SocketTimeoutException) {
                            // Nothing to read within the probe window. The connection is up.
                        }
                        continue
                    }
                    output.write(FrameEncoder.encode(message))
                    output.flush()
                    onSent(TimingRecord.frameNanosFromSeconds(message.timestampSeconds))
                    sent.incrementAndGet()
                    onStatus(ConnectionStatus.Connected(host, port, sent.get(), dropped.get()))
                }
            } catch (interrupted: InterruptedException) {
                // stop() interrupted the sleep. Leaving the loop is the whole point.
                Thread.currentThread().interrupt()
                return
            } catch (network: IOException) {
                // Refused, reset, or the hotspot dropped. Expected in the field. Wait and retry
                // rather than giving up, and say so on screen.
                Log.w(TAG, "laptop connection lost, retrying in ${RECONNECT_PAUSE_MILLIS} ms", network)
                onStatus(ConnectionStatus.Disconnected(network.message ?: network.javaClass.simpleName))
                try {
                    Thread.sleep(RECONNECT_PAUSE_MILLIS)
                } catch (interrupted: InterruptedException) {
                    Thread.currentThread().interrupt()
                    return
                }
            } finally {
                try {
                    socket?.close()
                } catch (closing: IOException) {
                    // Nothing left to do with a socket that will not close.
                }
            }
        }
    }

    private companion object {
        const val TAG = "LaptopConnection"
        const val CONNECT_TIMEOUT_MILLIS = 3_000
        const val RECONNECT_PAUSE_MILLIS = 1_000L
        // The longest a frame can wait while the idle probe is reading. Well under a frame interval.
        const val PEER_PROBE_MILLIS = 20
    }
}

/** What the screen shows about the link. */
sealed interface ConnectionStatus {
    data class Connecting(val host: String, val port: Int) : ConnectionStatus
    data class Connected(val host: String, val port: Int, val framesSent: Int, val framesDropped: Int) : ConnectionStatus
    data class Disconnected(val reason: String) : ConnectionStatus
}
