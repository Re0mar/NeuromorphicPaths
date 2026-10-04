package com.neuromorphicpaths.pixel.wire

import java.io.ByteArrayInputStream
import java.io.DataInputStream
import java.io.EOFException
import java.net.InetAddress
import java.net.ServerSocket
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
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

    /**
     * Reads whole messages until the client closes, noting when each arrived, then closes back the
     * way the laptop does. With holdOpen it reads to the end and then keeps the connection open
     * until released, like a laptop that never got round to closing.
     */
    private class Laptop(
        private val server: ServerSocket,
        private val closeAfter: Int = Int.MAX_VALUE,
        private val holdOpen: Boolean = false,
    ) {
        val messages = CopyOnWriteArrayList<ByteArray>()
        val arrivalNanos = CopyOnWriteArrayList<Long>()
        val release = CountDownLatch(1)
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
                    // The replay closed its side. Everything it sent has been read.
                }
                if (holdOpen) release.await(10, TimeUnit.SECONDS)
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
            assertEquals(ReplayStatus.Finished(40, 0, laptopConfirmed = true), waitFor(statuses))
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
            assertEquals(ReplayStatus.Finished(4, 0, laptopConfirmed = true), waitFor(statuses))
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
        // 100 ms apart, so the laptop's close lands well before the fourth frame is due.
        val stamps = (0 until 20).map { it * 0.1 }
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

    @Test
    fun aLaptopThatNeverClosesBackIsNotConfirmed() {
        val stamps = (0 until 5).map { it * 0.01 }
        ServerSocket(0, 1, loopback).use { server ->
            val laptop = Laptop(server, holdOpen = true)
            val statuses = CopyOnWriteArrayList<ReplayStatus>()
            WalkReplayer(
                "127.0.0.1",
                server.localPort,
                { ByteArrayInputStream(recording(stamps)) },
                { statuses.add(it) },
                confirmTimeoutMillis = 300,
            ).start()
            // Every frame went, and the laptop has them, but without its close the phone can't know that.
            assertEquals(ReplayStatus.Finished(5, 0, laptopConfirmed = false), waitFor(statuses))
            assertEquals(5, laptop.messages.size)
            laptop.release.countDown()
        }
    }

    @Test
    fun stoppingDuringAPauseReportsAFailure() = assertStoppingFails((0 until 40).map { it * 0.1 })

    // One timestamp throughout, so the replay never sleeps and stop() lands between sends, not in a pause.
    @Test
    fun stoppingBetweenSendsReportsAFailure() = assertStoppingFails(List(2_000) { 1.0 })

    private fun assertStoppingFails(stamps: List<Double>) {
        ServerSocket(0, 1, loopback).use { server ->
            Laptop(server)
            val statuses = CopyOnWriteArrayList<ReplayStatus>()
            val replayer = WalkReplayer("127.0.0.1", server.localPort, { ByteArrayInputStream(recording(stamps)) }, { statuses.add(it) })
            replayer.start()
            val deadline = System.currentTimeMillis() + 5_000
            while (statuses.none { it is ReplayStatus.Replaying && it.framesSent >= 2 } && System.currentTimeMillis() < deadline) Thread.sleep(5)
            replayer.stop()
            val outcome = waitFor(statuses)
            assertTrue(outcome is ReplayStatus.Failed && outcome.reason == "stopped before the end", "a stopped replay must not read as finished, got $outcome")
            assertTrue(outcome.framesSent in 2 until stamps.size, "got $outcome")
        }
    }

    @Test
    fun timestampsThatGoBackwardsStillSendEverything() {
        val stamps = listOf(5.0, 4.0, 3.0, 6.0)
        ServerSocket(0, 1, loopback).use { server ->
            val laptop = Laptop(server)
            val statuses = CopyOnWriteArrayList<ReplayStatus>()
            WalkReplayer("127.0.0.1", server.localPort, { ByteArrayInputStream(recording(stamps)) }, { statuses.add(it) }).start()
            assertEquals(ReplayStatus.Finished(4, 0, laptopConfirmed = true), waitFor(statuses))
            laptop.reader.join(5_000)
            stamps.forEachIndexed { index, stamp -> assertContentEquals(FrameEncoder.encode(message(stamp)), laptop.messages[index]) }
        }
    }
}
