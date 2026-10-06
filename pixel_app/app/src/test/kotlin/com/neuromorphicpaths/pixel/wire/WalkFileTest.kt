package com.neuromorphicpaths.pixel.wire

import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Reading a recorded walk back, from bytes [FrameEncoder] wrote. */
class WalkFileTest {
    private fun message(stamp: Double) = DepthMessage(
        timestampSeconds = stamp,
        depthMillimeters = shortArrayOf(1, 2, 3, 4),
        rows = 2,
        columns = 2,
        intrinsics = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        orientationWxyz = doubleArrayOf(1.0, 0.0, 0.0, 0.0),
        positionXyz = doubleArrayOf(0.5, 1.2, -3.0),
        groundPlane = null,
    )

    private fun readAll(bytes: ByteArray): Pair<List<RecordedMessage>, Int> {
        val reader = WalkFileReader(ByteArrayInputStream(bytes))
        val messages = generateSequence { reader.next() }.toList()
        return messages to reader.truncatedBytes
    }

    @Test
    fun readsMessagesAndTheirTimestamps() {
        val encoded = listOf(10.0, 10.033, 10.067).map { FrameEncoder.encode(message(it)) }
        val (messages, truncated) = readAll(encoded.reduce(ByteArray::plus))
        assertEquals(listOf(10.0, 10.033, 10.067), messages.map { it.timestampSeconds })
        encoded.zip(messages).forEach { (written, read) -> assertContentEquals(written, read.wireBytes) }
        assertEquals(0, truncated)
    }

    @Test
    fun truncatedTailIsReportedNotThrown() {
        val first = FrameEncoder.encode(message(1.0))
        val second = FrameEncoder.encode(message(2.0))
        val cut = first + second.copyOf(second.size - 5)
        val (messages, truncated) = readAll(cut)
        assertEquals(listOf(1.0), messages.map { it.timestampSeconds })
        assertEquals(second.size - 5, truncated)
    }

    @Test
    fun aPartialLengthPrefixIsATruncatedTail() {
        val first = FrameEncoder.encode(message(1.0))
        val (messages, truncated) = readAll(first + byteArrayOf(0, 0))
        assertEquals(1, messages.size)
        assertEquals(2, truncated)
    }

    @Test
    fun absurdLengthIsRefused() {
        val claim = ByteBuffer.allocate(4).putInt(Int.MAX_VALUE).array()
        val failure = assertFailsWith<MalformedWalkException> { readAll(claim + ByteArray(16)) }
        assertTrue(failure.message.orEmpty().contains(Int.MAX_VALUE.toString()), failure.message)
    }

    @Test
    fun aPayloadWithNoHeaderIsRefused() {
        val payload = "no newline anywhere".toByteArray()
        val bytes = ByteBuffer.allocate(4).putInt(payload.size).array() + payload
        assertFailsWith<MalformedWalkException> { readAll(bytes) }
    }

    @Test
    fun emptyFileHasNoMessages() {
        val reader = WalkFileReader(ByteArrayInputStream(ByteArray(0)))
        assertNull(reader.next())
        assertEquals(0, reader.truncatedBytes)
    }
}
