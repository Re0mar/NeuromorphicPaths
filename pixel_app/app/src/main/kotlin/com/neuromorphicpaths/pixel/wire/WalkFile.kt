package com.neuromorphicpaths.pixel.wire

import java.io.BufferedInputStream
import java.io.Closeable
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import org.json.JSONException
import org.json.JSONObject

/**
 * One message of a recorded walk: its bytes exactly as they go on the wire, length prefix
 * included, and the capture time from its header.
 */
class RecordedMessage(val wireBytes: ByteArray, val timestampSeconds: Double)

/** A file that isn't a recorded walk, or a message inside one that can't be read. */
class MalformedWalkException(message: String) : IOException(message)

/**
 * Reads a recorded walk back one message at a time.
 *
 * A recording is the wire stream itself: each message a 4-byte big-endian length and then the
 * payload [FrameEncoder] wrote. So the length prefix is all the framing a file needs. A message
 * cut off at the end, which is what an app killed mid-write leaves behind, ends the read and is
 * counted in [truncatedBytes], not thrown. Everything before it is a good recording.
 */
class WalkFileReader(input: InputStream) : Closeable {
    private val stream = BufferedInputStream(input)

    /** Bytes at the end of the file that didn't make a whole message. Zero for a clean file. */
    var truncatedBytes: Int = 0
        private set

    /**
     * The next message, or null at the end of the recording.
     *
     * @throws MalformedWalkException On a length no message could have, or a payload with no
     *   readable header. Either means the file isn't a recording, and nothing after it can be trusted.
     */
    fun next(): RecordedMessage? {
        val prefix = ByteArray(LENGTH_PREFIX_BYTES)
        val prefixRead = readFully(prefix)
        if (prefixRead == 0) return null
        if (prefixRead < LENGTH_PREFIX_BYTES) {
            truncatedBytes += prefixRead
            return null
        }
        val payloadLength = ByteBuffer.wrap(prefix).int
        if (payloadLength <= 0 || payloadLength > MAX_PAYLOAD_BYTES) {
            throw MalformedWalkException("a message claims $payloadLength bytes, so this isn't a recorded walk")
        }
        val payload = ByteArray(payloadLength)
        val payloadRead = readFully(payload)
        if (payloadRead < payloadLength) {
            truncatedBytes += LENGTH_PREFIX_BYTES + payloadRead
            return null
        }
        return RecordedMessage(prefix + payload, timestampOf(payload))
    }

    override fun close() {
        stream.close()
    }

    /** Reads until the buffer is full or the stream ends, and says how much it got. */
    private fun readFully(buffer: ByteArray): Int {
        var filled = 0
        while (filled < buffer.size) {
            val count = stream.read(buffer, filled, buffer.size - filled)
            if (count < 0) break
            filled += count
        }
        return filled
    }

    private fun timestampOf(payload: ByteArray): Double {
        val newline = payload.indexOf('\n'.code.toByte())
        if (newline < 0) throw MalformedWalkException("a message has no header line")
        return try {
            JSONObject(String(payload, 0, newline, Charsets.UTF_8)).getDouble("timestamp_seconds")
        } catch (unreadable: JSONException) {
            throw MalformedWalkException("a message header has no readable timestamp_seconds: ${unreadable.message}")
        }
    }

    private companion object {
        const val LENGTH_PREFIX_BYTES = 4
        // Far above any depth frame the phone sends (about 30 KB), and far below what a stray
        // four bytes of some other file would usually claim.
        const val MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
    }
}
