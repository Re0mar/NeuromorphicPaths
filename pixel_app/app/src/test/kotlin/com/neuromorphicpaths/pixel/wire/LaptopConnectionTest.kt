package com.neuromorphicpaths.pixel.wire

import java.io.DataInputStream
import java.net.ServerSocket
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.json.JSONObject

/**
 * The connection against a real socket on the loopback, standing in for the laptop.
 *
 * The positive case first: a message sent is a message the server reads back framed. Only then
 * do the negatives mean anything, because a server that never received anything would make a
 * refused connection and a dropped frame look the same as success.
 */
class LaptopConnectionTest {
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

    // Bound to the IPv4 loopback by name. getLoopbackAddress() can be ::1 on a dual-stack machine,
    // and a server listening there never hears a client dialing 127.0.0.1, which made the positive
    // test fail for a reason that had nothing to do with the code under test.
    private val loopback: java.net.InetAddress = java.net.InetAddress.getByName("127.0.0.1")

    private fun waitUntil(timeoutMillis: Long = 5_000, condition: () -> Boolean): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMillis
        while (System.currentTimeMillis() < deadline) {
            if (condition()) return true
            Thread.sleep(5)
        }
        return condition()
    }

    @Test
    fun aPlainLoopbackConnectWorksInThisJvm() {
        // The control for every other test here. None of LaptopConnection is involved. If this
        // fails, the test JVM cannot connect to a loopback listener and nothing below means anything.
        ServerSocket(0, 1, loopback).use { server ->
            val acceptor = thread(isDaemon = true) { runCatching { server.accept().close() } }
            java.net.Socket().use { client ->
                client.connect(java.net.InetSocketAddress("127.0.0.1", server.localPort), 3_000)
                assertTrue(client.isConnected, "plain connect to ${server.localSocketAddress} failed")
            }
            acceptor.join(2_000)
        }
    }

    @Test
    fun aMessageOfferedArrivesFramedAtTheServer() {
        val server = ServerSocket(0, 1, loopback)
        val received = CopyOnWriteArrayList<ByteArray>()
        val acceptor = thread(isDaemon = true) {
            server.accept().use { socket ->
                val input = DataInputStream(socket.getInputStream())
                val length = input.readInt()
                val payload = ByteArray(length)
                input.readFully(payload)
                received.add(payload)
            }
        }
        val statuses = CopyOnWriteArrayList<ConnectionStatus>()
        val connection = LaptopConnection("127.0.0.1", server.localPort) { statuses.add(it) }
        try {
            connection.start()
            assertTrue(waitUntil { statuses.any { it is ConnectionStatus.Connected } }, "never connected, saw $statuses")
            connection.offer(message(7.5))
            acceptor.join(TimeUnit.SECONDS.toMillis(5))
        } finally {
            connection.stop()
            server.close()
        }

        assertEquals(1, received.size, "exactly one message")
        val payload = received[0]
        val newline = payload.indexOf('\n'.code.toByte())
        val header = JSONObject(String(payload, 0, newline, Charsets.UTF_8))
        assertEquals(7.5, header.getDouble("timestamp_seconds"))
        assertEquals(8, payload.size - newline - 1, "four uint16 values after the header")
        assertEquals(1, connection.framesSent)
    }

    @Test
    fun aLaptopThatGoesAwayWhileTheAppIsIdleIsNoticedWithoutAFrame() {
        // On the first phone run the laptop process was killed while the phone had no depth to
        // send. The screen said Connected for minutes, because nothing ever wrote to the socket.
        val server = ServerSocket(0, 1, loopback)
        val accepted = java.util.concurrent.CountDownLatch(1)
        var connection: java.net.Socket? = null
        thread(isDaemon = true) { connection = server.accept(); accepted.countDown() }
        val statuses = CopyOnWriteArrayList<ConnectionStatus>()
        val app = LaptopConnection("127.0.0.1", server.localPort) { statuses.add(it) }
        try {
            app.start()
            assertTrue(accepted.await(3, TimeUnit.SECONDS), "the app never connected")
            assertTrue(waitUntil { statuses.any { it is ConnectionStatus.Connected } })
            connection?.close()
            server.close()

            assertTrue(waitUntil(2_000) { statuses.any { it is ConnectionStatus.Disconnected } }, "the closed laptop was not noticed, saw $statuses")
        } finally {
            app.stop()
        }
    }

    @Test
    fun aQuietLaptopThatIsStillThereIsNotMistakenForAGoneOne() {
        val server = ServerSocket(0, 1, loopback)
        thread(isDaemon = true) { runCatching { server.accept(); Thread.sleep(3_000) } }
        val statuses = CopyOnWriteArrayList<ConnectionStatus>()
        val app = LaptopConnection("127.0.0.1", server.localPort) { statuses.add(it) }
        try {
            app.start()
            assertTrue(waitUntil { statuses.any { it is ConnectionStatus.Connected } })
            Thread.sleep(600)

            assertTrue(statuses.none { it is ConnectionStatus.Disconnected }, "a silent but open laptop read as gone: $statuses")
        } finally {
            app.stop()
            server.close()
        }
    }

    @Test
    fun theNewestMessageWinsAndReplacedOnesAreCounted() {
        // No server yet, so nothing is taken from the slot and every offer but the first replaces.
        val connection = LaptopConnection("127.0.0.1", 1) { }
        connection.offer(message(1.0))
        connection.offer(message(2.0))
        connection.offer(message(3.0))

        assertEquals(2, connection.framesDropped)
        assertEquals(0, connection.framesSent)
    }

    @Test
    fun aReplacedMessageReportsTheDroppedFrame() {
        // No server, so the slot is never emptied and the second offer replaces the first.
        val dropped = CopyOnWriteArrayList<Long>()
        val connection = LaptopConnection("127.0.0.1", 1, onDropped = { dropped.add(it) }) { }
        connection.offer(message(1.0))
        connection.offer(message(2.0))

        assertEquals(listOf(1_000_000_000L), dropped.toList(), "the replaced frame, not the newer one")
    }

    @Test
    fun aFramePendingAtStopIsReportedDropped() {
        val dropped = CopyOnWriteArrayList<Long>()
        // Never started, so nothing takes the message and it is still pending at stop.
        val connection = LaptopConnection("127.0.0.1", 9, onDropped = { dropped.add(it) }) { }

        connection.offer(message(7.5))
        connection.stop()

        assertEquals(listOf(7_500_000_000L), dropped.toList())
        assertEquals(1, connection.framesDropped)
    }

    @Test
    fun everyOfferedFrameEndsSentOrDroppedWhenWritesFail() {
        // A laptop that resets every connection the moment it opens. Writes then fail, and a
        // frame caught mid-write must still end as dropped, never as nothing or as both.
        val server = ServerSocket(0, 50, loopback)
        val accepting = thread(isDaemon = true) {
            while (!server.isClosed) {
                runCatching {
                    server.accept().use { socket ->
                        socket.setSoLinger(true, 0)
                        socket.getInputStream().read()
                    }
                }
            }
        }
        val sent = CopyOnWriteArrayList<Long>()
        val dropped = CopyOnWriteArrayList<Long>()
        val statuses = CopyOnWriteArrayList<ConnectionStatus>()
        val connection = LaptopConnection("127.0.0.1", server.localPort, onSent = { sent.add(it) }, onDropped = { dropped.add(it) }) {
            statuses.add(it)
        }
        val offered = mutableListOf<Long>()
        try {
            connection.start()
            val deadline = System.currentTimeMillis() + 5_000
            var seconds = 1.0
            while (System.currentTimeMillis() < deadline && statuses.none { it is ConnectionStatus.Disconnected }) {
                connection.offer(message(seconds))
                offered.add(Math.round(seconds * 1e9))
                seconds += 0.001
                Thread.sleep(2)
            }
            assertTrue(statuses.any { it is ConnectionStatus.Disconnected }, "no write ever failed, so this proved nothing")
        } finally {
            connection.stop()
            server.close()
            accepting.join(2_000)
        }
        // stop() reports what was pending. Give a send still in flight a moment to report too.
        Thread.sleep(200)

        val outcomes = sent + dropped
        assertEquals(offered.sorted(), outcomes.sorted(), "each offered frame needs exactly one of sent or dropped")
    }

    @Test
    fun sentIsReportedAfterTheWrite() {
        val server = ServerSocket(0, 1, loopback)
        val readAtServer = java.util.concurrent.CountDownLatch(1)
        thread(isDaemon = true) {
            server.accept().use { socket ->
                val input = DataInputStream(socket.getInputStream())
                input.readFully(ByteArray(input.readInt()))
                readAtServer.countDown()
                Thread.sleep(1_000)
            }
        }
        val sent = CopyOnWriteArrayList<Long>()
        val statuses = CopyOnWriteArrayList<ConnectionStatus>()
        val connection = LaptopConnection("127.0.0.1", server.localPort, onSent = { sent.add(it) }) { statuses.add(it) }
        try {
            connection.start()
            assertTrue(waitUntil { statuses.any { it is ConnectionStatus.Connected } }, "never connected, saw $statuses")
            connection.offer(message(7.5))
            assertTrue(readAtServer.await(5, TimeUnit.SECONDS), "the server never read the message")
            assertTrue(waitUntil(2_000) { sent.isNotEmpty() }, "the write was never reported")
        } finally {
            connection.stop()
            server.close()
        }

        assertEquals(listOf(7_500_000_000L), sent.toList())
    }

    @Test
    fun aRefusedPortIsReportedAndRetriedNotFatal() {
        val probe = ServerSocket(0, 1, loopback)
        val closedPort = probe.localPort
        probe.close()
        val statuses = CopyOnWriteArrayList<ConnectionStatus>()
        val connection = LaptopConnection("127.0.0.1", closedPort) { statuses.add(it) }
        try {
            connection.start()
            assertTrue(waitUntil(4_000) { statuses.count { it is ConnectionStatus.Disconnected } >= 2 }, "expected repeated disconnected reports, got $statuses")
        } finally {
            connection.stop()
        }

        assertTrue(statuses.none { it is ConnectionStatus.Connected })
        assertEquals(0, connection.framesSent)
    }

    @Test
    fun stopEndsTheThreadPromptlyWhileWaitingToReconnect() {
        val probe = ServerSocket(0, 1, loopback)
        val closedPort = probe.localPort
        probe.close()
        val connection = LaptopConnection("127.0.0.1", closedPort) { }
        connection.start()
        Thread.sleep(100)

        val started = System.currentTimeMillis()
        connection.stop()

        assertTrue(System.currentTimeMillis() - started < 1_500, "stop must not wait out the reconnect pause")
    }

    @Test
    fun startTwiceIsOneConnection() {
        val server = ServerSocket(0, 1, loopback)
        val accepted = CopyOnWriteArrayList<Int>()
        thread(isDaemon = true) {
            while (!server.isClosed) {
                try {
                    server.accept().use { accepted.add(1); Thread.sleep(300) }
                } catch (closed: java.io.IOException) {
                    return@thread
                }
            }
        }
        val connection = LaptopConnection("127.0.0.1", server.localPort) { }
        try {
            connection.start()
            connection.start()
            Thread.sleep(400)
        } finally {
            connection.stop()
            server.close()
        }

        assertEquals(1, accepted.size)
    }
}
