package com.neuromorphicpaths.pixel.wire

import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The path connection against a real socket on the loopback, standing in for the laptop's sink.
 *
 * The positive case first. A connection that never received a path would make a refused port
 * and a dropped message look the same as success.
 */
class PathConnectionTest {
    private val loopback: InetAddress = InetAddress.getByName("127.0.0.1")

    private val good = """{"timestamp_seconds": 1.0, "times_seconds": [0.0, 0.1], "lateral_offsets_meters": [0.0, 0.05], "lookahead_heading_radians": 0.25, "alarm": false, "cumulative_cost_bits": 2.5}"""

    private fun framed(payload: ByteArray, declaredLength: Int = payload.size): ByteArray {
        val out = ByteArrayOutputStream()
        DataOutputStream(out).use {
            it.writeInt(declaredLength)
            it.write(payload)
        }
        return out.toByteArray()
    }

    private fun framed(json: String): ByteArray = framed(json.toByteArray(Charsets.UTF_8))

    private fun waitUntil(timeoutMillis: Long = 5_000, condition: () -> Boolean): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMillis
        while (System.currentTimeMillis() < deadline) {
            if (condition()) return true
            Thread.sleep(5)
        }
        return condition()
    }

    private class Harness(val statuses: CopyOnWriteArrayList<PathConnectionStatus> = CopyOnWriteArrayList(), val paths: CopyOnWriteArrayList<ReceivedPath> = CopyOnWriteArrayList()) {
        val clock = AtomicLong(0)
        fun connection(port: Int) = PathConnection("127.0.0.1", port, { statuses.add(it) }, { paths.add(it) }) { clock.incrementAndGet() }
    }

    @Test
    fun aFramedPathFromTheLaptopReachesTheCallback() {
        val server = ServerSocket(0, 1, loopback)
        val served = CountDownLatch(1)
        thread(isDaemon = true) {
            server.accept().use { socket ->
                socket.getOutputStream().write(framed(good))
                socket.getOutputStream().flush()
                served.countDown()
                Thread.sleep(500)
            }
        }
        val harness = Harness()
        val connection = harness.connection(server.localPort)
        try {
            connection.start()
            assertTrue(served.await(5, TimeUnit.SECONDS), "the server never served")
            assertTrue(waitUntil { harness.paths.size == 1 }, "no path reached the callback, saw ${harness.statuses}")
        } finally {
            connection.stop()
            server.close()
        }

        assertEquals(0.25, harness.paths[0].message.lookaheadHeadingRadians)
        assertEquals(1, connection.pathsReceived)
        assertEquals(0, connection.pathsRefused)
        val last = harness.statuses.last { it is PathConnectionStatus.Connected } as PathConnectionStatus.Connected
        assertEquals(1, last.received)
    }

    @Test
    fun aReceivedPathCarriesTheInjectedClocksStamp() {
        val server = ServerSocket(0, 1, loopback)
        thread(isDaemon = true) {
            server.accept().use { socket ->
                socket.getOutputStream().write(framed(good) + framed(good))
                socket.getOutputStream().flush()
                Thread.sleep(500)
            }
        }
        val harness = Harness()
        harness.clock.set(4_241)
        val connection = harness.connection(server.localPort)
        try {
            connection.start()
            assertTrue(waitUntil { harness.paths.size == 2 }, "expected two paths, saw ${harness.paths.size}")
        } finally {
            connection.stop()
            server.close()
        }

        // The counter clock hands out 4242 then 4243: the stamp is taken per path, on receipt.
        assertEquals(listOf(4_242L, 4_243L), harness.paths.map { it.receivedAtMillis })
    }

    @Test
    fun aRefusedMessageIsCountedAndTheNextOneStillArrives() {
        val server = ServerSocket(0, 1, loopback)
        thread(isDaemon = true) {
            server.accept().use { socket ->
                socket.getOutputStream().write(framed("garbage") + framed(good))
                socket.getOutputStream().flush()
                Thread.sleep(500)
            }
        }
        val harness = Harness()
        val connection = harness.connection(server.localPort)
        try {
            connection.start()
            assertTrue(waitUntil { harness.paths.size == 1 }, "the good message after the bad one never arrived")
        } finally {
            connection.stop()
            server.close()
        }

        assertEquals(1, connection.pathsRefused)
        assertEquals(1, connection.pathsReceived)
        val last = harness.statuses.last { it is PathConnectionStatus.Connected } as PathConnectionStatus.Connected
        assertEquals(1, last.refused)
        assertTrue(last.lastRefusal!!.contains("not a JSON object"), last.lastRefusal)
    }

    @Test
    fun aLaptopThatClosesIsReportedAndTheNextConnectionIsServed() {
        // The listener stays, as the laptop's does. The accepted socket closes, which is what a
        // run ending and restarting looks like from the phone.
        val server = ServerSocket(0, 1, loopback)
        val accepts = CopyOnWriteArrayList<Int>()
        thread(isDaemon = true) {
            while (!server.isClosed) {
                try {
                    val socket = server.accept()
                    accepts.add(1)
                    if (accepts.size == 1) {
                        socket.close()
                    } else {
                        socket.getOutputStream().write(framed(good))
                        socket.getOutputStream().flush()
                        Thread.sleep(500)
                        socket.close()
                    }
                } catch (closed: java.io.IOException) {
                    return@thread
                }
            }
        }
        val harness = Harness()
        val connection = harness.connection(server.localPort)
        try {
            connection.start()
            assertTrue(waitUntil { harness.statuses.any { it is PathConnectionStatus.Disconnected } }, "the closed laptop was not noticed, saw ${harness.statuses}")
            assertTrue(waitUntil { harness.paths.size == 1 }, "the reconnect was not served, accepts ${accepts.size}, saw ${harness.statuses}")
        } finally {
            connection.stop()
            server.close()
        }

        assertEquals(2, accepts.size)
    }

    @Test
    fun aBadLengthPrefixEndsTheConnectionAndItReconnects() {
        val server = ServerSocket(0, 1, loopback)
        val accepts = CopyOnWriteArrayList<Int>()
        thread(isDaemon = true) {
            while (!server.isClosed) {
                try {
                    val socket = server.accept()
                    accepts.add(1)
                    val payload = if (accepts.size == 1) framed(ByteArray(0), declaredLength = 0) else framed(good)
                    socket.getOutputStream().write(payload)
                    socket.getOutputStream().flush()
                    Thread.sleep(500)
                    socket.close()
                } catch (closed: java.io.IOException) {
                    return@thread
                }
            }
        }
        val harness = Harness()
        val connection = harness.connection(server.localPort)
        try {
            connection.start()
            assertTrue(waitUntil { harness.statuses.any { it is PathConnectionStatus.Disconnected && it.reason.contains("out of step") } }, "the bad prefix did not end the connection, saw ${harness.statuses}")
            assertTrue(waitUntil { harness.paths.size == 1 }, "the reconnect was not served")
        } finally {
            connection.stop()
            server.close()
        }

        assertEquals(0, connection.pathsRefused, "a bad prefix is not a refused message, it is a lost connection")
    }

    @Test
    fun aRefusedPortIsRetriedNotFatal() {
        val probe = ServerSocket(0, 1, loopback)
        val closedPort = probe.localPort
        probe.close()
        val harness = Harness()
        val connection = harness.connection(closedPort)
        try {
            connection.start()
            assertTrue(waitUntil(4_000) { harness.statuses.count { it is PathConnectionStatus.Disconnected } >= 2 }, "expected repeated disconnected reports, got ${harness.statuses}")
        } finally {
            connection.stop()
        }

        assertTrue(harness.statuses.none { it is PathConnectionStatus.Connected })
    }

    @Test
    fun stopClosesTheSocketWhileBlockedInARead() {
        // A thread blocked in a socket read ignores interrupts. stop() has to close the socket,
        // and the laptop sees that as the phone leaving.
        val server = ServerSocket(0, 1, loopback)
        val peerSawEof = CountDownLatch(1)
        var accepted: Socket? = null
        thread(isDaemon = true) {
            accepted = server.accept()
            if (accepted!!.getInputStream().read() == -1) peerSawEof.countDown()
        }
        val harness = Harness()
        val connection = harness.connection(server.localPort)
        try {
            connection.start()
            assertTrue(waitUntil { harness.statuses.any { it is PathConnectionStatus.Connected } })

            connection.stop()

            assertTrue(peerSawEof.await(2, TimeUnit.SECONDS), "the laptop never saw the phone leave")
        } finally {
            connection.stop()
            accepted?.close()
            server.close()
        }
        assertTrue(harness.statuses.none { it is PathConnectionStatus.Disconnected }, "stop is not a lost connection: ${harness.statuses}")
    }

    @Test
    fun stopEndsTheThreadPromptlyWhileWaitingToReconnect() {
        val probe = ServerSocket(0, 1, loopback)
        val closedPort = probe.localPort
        probe.close()
        val connection = Harness().connection(closedPort)
        connection.start()
        Thread.sleep(100)

        val started = System.currentTimeMillis()
        connection.stop()

        assertTrue(System.currentTimeMillis() - started < 1_500, "stop must not wait out the reconnect pause")
    }
}
