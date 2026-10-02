package com.neuromorphicpaths.pixel.wire

import android.util.Log
import java.io.IOException
import java.net.InetSocketAddress
import java.net.Socket
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
 */
class LaptopConnection(
    private val host: String,
    private val port: Int,
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
        if (pending.getAndSet(message) != null) {
            dropped.incrementAndGet()
        }
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
                val output = opened.getOutputStream()
                onStatus(ConnectionStatus.Connected(host, port, sent.get(), dropped.get()))
                while (running) {
                    val message = pending.getAndSet(null)
                    if (message == null) {
                        Thread.sleep(IDLE_SLEEP_MILLIS)
                        continue
                    }
                    output.write(FrameEncoder.encode(message))
                    output.flush()
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
        const val IDLE_SLEEP_MILLIS = 2L
    }
}

/** What the screen shows about the link. */
sealed interface ConnectionStatus {
    data class Connecting(val host: String, val port: Int) : ConnectionStatus
    data class Connected(val host: String, val port: Int, val framesSent: Int, val framesDropped: Int) : ConnectionStatus
    data class Disconnected(val reason: String) : ConnectionStatus
}
