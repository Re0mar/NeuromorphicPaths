package com.neuromorphicpaths.pixel.wire

import java.io.ByteArrayInputStream
import java.io.DataInputStream
import java.io.EOFException
import java.net.InetAddress
import java.net.ServerSocket
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.concurrent.thread
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Replaying a recorded walk to a real socket on the loopback, standing in for the laptop. */
class WalkReplayerTest {
    // The IPv4 loopback by name, for the reason LaptopConnectionTest gives.
    private val loopback: InetAddress = InetAddress.getByName("127.0.0.1")

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

    private fun recording(stamps: List<Double>): ByteArray = stamps.map { FrameEncoder.encode(message(it)) }.reduce(ByteArray::plus)

    private fun waitFor(statuses: List<ReplayStatus>, timeoutMillis: Long = 10_000): ReplayStatus? {
        val deadline = System.currentTimeMillis() + timeoutMillis
        while (System.currentTimeMillis() < deadline) {
            statuses.lastOrNull { it is ReplayStatus.Finished || it is ReplayStatus.Failed }?.let { return it }
            Thread.sleep(5)
        }
        return null
    }

    /** Reads whole messages until the client closes, noting when each arrived. */
    private class Laptop(private val server: ServerSocket, private val closeAfter: Int = Int.MAX_VALUE) {
        val messages = CopyOnWriteArrayList<ByteArray>()
        val arrivalNanos = CopyOnWriteArrayList<Long>()
        val reader = thread(isDaemon = true) {
            server.accept().use { socket ->
                val input = DataInputStream(socket.getInputStream())
                try {
                    while (messages.size < closeAfter) {
                        val length = input.readInt()
                        val payload = ByteArray(length)
                        input.readFully(payload)
                        arrivalNanos.add(System.nanoTime())
                        messages.add(java.nio.ByteBuffer.allocate(4).putInt(length).array() + payload)
                    }
                } catch (finished: EOFException) {
                    // The replay closed the connection. Everything it sent has been read.
                }
            }
        }
    }

    @Test
    fun everyMessageArrivesInOrderUnchanged() {
        val stamps = (0 until 40).map { 5.0 + it * 0.005 }
        val bytes = recording(stamps)
        ServerSocket(0, 1, loopback).use { server ->
            val laptop = Laptop(server)
            val statuses = CopyOnWriteArrayList<ReplayStatus>()
            WalkReplayer("127.0.0.1", server.localPort, { ByteArrayInputStream(bytes) }, { statuses.add(it) }).start()
            assertEquals(ReplayStatus.Finished(40, 0), waitFor(statuses))
            laptop.reader.join(5_000)
            assertEquals(40, laptop.messages.size)
            stamps.forEachIndexed { index, stamp -> assertContentEquals(FrameEncoder.encode(message(stamp)), laptop.messages[index]) }
        }
    }

    @Test
    fun replayIsPacedByRecordedTimestamps() {
        val stamps = listOf(0.0, 0.1, 0.2, 0.3)
        val bytes = recording(stamps)
        ServerSocket(0, 1, loopback).use { server ->
            val laptop = Laptop(server)
            val statuses = CopyOnWriteArrayList<ReplayStatus>()
            WalkReplayer("127.0.0.1", server.localPort, { ByteArrayInputStream(bytes) }, { statuses.add(it) }).start()
            assertEquals(ReplayStatus.Finished(4, 0), waitFor(statuses))
            laptop.reader.join(5_000)
            val gapsMillis = laptop.arrivalNanos.zipWithNext { earlier, later -> (later - earlier) / 1_000_000.0 }
            assertTrue(gapsMillis.all { it >= 85.0 }, "messages recorded 100 ms apart arrived $gapsMillis ms apart")
        }
    }

    @Test
    fun refusedConnectionFailsWithAReason() {
        // A port that was free a moment ago and has nothing listening now.
        val port = ServerSocket(0, 1, loopback).use { it.localPort }
        val statuses = CopyOnWriteArrayList<ReplayStatus>()
        WalkReplayer("127.0.0.1", port, { ByteArrayInputStream(recording(listOf(1.0))) }, { statuses.add(it) }).start()
        val outcome = waitFor(statuses)
        assertTrue(outcome is ReplayStatus.Failed && outcome.framesSent == 0, "expected a failure with nothing sent, got $outcome")
    }

    @Test
    fun lostConnectionEndsTheReplay() {
        val stamps = (0 until 20).map { it * 0.03 }
        val bytes = recording(stamps)
        ServerSocket(0, 1, loopback).use { server ->
            val laptop = Laptop(server, closeAfter = 3)
            val statuses = CopyOnWriteArrayList<ReplayStatus>()
            WalkReplayer("127.0.0.1", server.localPort, { ByteArrayInputStream(bytes) }, { statuses.add(it) }).start()
            val outcome = waitFor(statuses)
            assertTrue(outcome is ReplayStatus.Failed, "a replay the laptop hung up on must fail, got $outcome")
            // Exactly the three the laptop read. A write into a closed connection can sit in the send
            // buffer and look sent, which is what checking for the laptop before each frame prevents.
            assertEquals(3, outcome.framesSent)
            assertEquals(3, laptop.messages.size)
        }
    }

    @Test
    fun aRecordingThatWontReadFails() {
        val notAWalk = java.nio.ByteBuffer.allocate(4).putInt(Int.MAX_VALUE).array()
        ServerSocket(0, 1, loopback).use { server ->
            Laptop(server)
            val statuses = CopyOnWriteArrayList<ReplayStatus>()
            WalkReplayer("127.0.0.1", server.localPort, { ByteArrayInputStream(notAWalk) }, { statuses.add(it) }).start()
            val outcome = waitFor(statuses)
            assertTrue(outcome is ReplayStatus.Failed && outcome.reason.contains("isn't a recorded walk"), "got $outcome")
        }
    }
}
