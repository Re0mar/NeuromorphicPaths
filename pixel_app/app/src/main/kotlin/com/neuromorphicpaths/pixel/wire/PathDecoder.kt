package com.neuromorphicpaths.pixel.wire

import java.io.DataInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

/** What arrived is not a valid path. The receiver drops it, shows the rule it broke, and keeps reading. */
class PathDecodeException(message: String) : Exception(message)

/**
 * Reads and decodes the laptop's path messages, with the laptop's own refusal discipline.
 *
 * Every key is required, every number is checked for its type before it is read and for being
 * finite after, and `alarm` must be a JSON boolean. Nothing here uses the `opt*` accessors or
 * `getDouble`: `org.json` coerces a `true` and a numeric string into a number through those, and
 * a message that breaks the contract would then be read as if it kept it.
 */
object PathDecoder {
    /** A path is a few hundred bytes. A prefix past this is a stream out of step, not a long path. */
    const val MAX_PATH_BYTES = 1 shl 20

    private val REQUIRED_KEYS = listOf(
        "timestamp_seconds",
        "times_seconds",
        "lateral_offsets_meters",
        "lookahead_heading_radians",
        "alarm",
        "cumulative_cost_bits",
    )

    /**
     * Read one length-prefixed message off the stream.
     *
     * A zero or oversized prefix is thrown as an [IOException], not a [PathDecodeException]: the
     * stream is out of step and the only recovery is reconnecting, which is the rule the laptop
     * applies to its own prefix on the depth side. End of stream propagates as [java.io.EOFException].
     */
    fun readFramed(input: DataInputStream): ByteArray {
        val length = input.readInt()
        if (length <= 0) throw IOException("path length prefix is $length, the stream is out of step")
        if (length > MAX_PATH_BYTES) throw IOException("path claims $length bytes, over the $MAX_PATH_BYTES limit, the stream is out of step")
        val payload = ByteArray(length)
        input.readFully(payload)
        return payload
    }

    /** Decode one payload, or throw [PathDecodeException] naming the key or bound it broke. */
    fun decode(payload: ByteArray): PathMessage {
        val text = try {
            Charsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(payload))
                .toString()
        } catch (encoding: CharacterCodingException) {
            throw PathDecodeException("path is not valid UTF-8: ${encoding.message ?: "malformed input"}")
        }
        val json = try {
            JSONObject(text)
        } catch (notJson: JSONException) {
            throw PathDecodeException("path is not a JSON object: ${notJson.message}")
        }
        for (key in REQUIRED_KEYS) {
            if (!json.has(key)) throw PathDecodeException("missing the key '$key'")
        }
        try {
            return PathMessage(
                timestampSeconds = number(json.get("timestamp_seconds"), "timestamp_seconds"),
                timesSeconds = numbers(json.get("times_seconds"), "times_seconds"),
                lateralOffsetsMeters = numbers(json.get("lateral_offsets_meters"), "lateral_offsets_meters"),
                lookaheadHeadingRadians = number(json.get("lookahead_heading_radians"), "lookahead_heading_radians"),
                alarm = boolean(json.get("alarm"), "alarm"),
                cumulativeCostBits = number(json.get("cumulative_cost_bits"), "cumulative_cost_bits"),
            )
        } catch (inconsistent: IllegalArgumentException) {
            // The message's own init, so the receiver has one exception type to catch.
            throw PathDecodeException("fields decoded but do not make a path: ${inconsistent.message}")
        }
    }

    private fun number(raw: Any, field: String): Double {
        // Checked by type before it is read. A Boolean is not a Number in Java, so a true here is
        // refused rather than read as 1.0, and a numeric string is refused rather than parsed.
        if (raw !is Number) throw PathDecodeException("$field must be a number, got ${describe(raw)}")
        val value = raw.toDouble()
        // Android's org.json parses an Infinity token into a Double. The test library's does not,
        // but the device is what this guards.
        if (!value.isFinite()) throw PathDecodeException("$field must be finite, got $value")
        return value
    }

    private fun numbers(raw: Any, field: String): DoubleArray {
        if (raw !is JSONArray) throw PathDecodeException("$field must be a list, got ${describe(raw)}")
        return DoubleArray(raw.length()) { index -> number(raw.get(index), "$field[$index]") }
    }

    private fun boolean(raw: Any, field: String): Boolean {
        if (raw !is Boolean) throw PathDecodeException("$field must be true or false, got ${describe(raw)}")
        return raw
    }

    private fun describe(raw: Any): String = when {
        raw === JSONObject.NULL -> "null"
        raw is String -> "a string"
        raw is Boolean -> "a boolean"
        raw is Number -> "a number"
        raw is JSONArray -> "a list of ${raw.length()}"
        raw is JSONObject -> "an object"
        else -> raw.javaClass.simpleName
    }
}
