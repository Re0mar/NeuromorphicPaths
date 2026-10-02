package com.neuromorphicpaths.pixel.wire

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.io.File
import java.io.IOException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import org.json.JSONArray
import org.json.JSONObject

/**
 * The decoder against the wire format, one rule per test, and the laptop's committed fixture.
 *
 * Every refusal case breaks exactly one rule with everything else valid, and asserts the key or
 * the bound in the message, which no exception the runtime raises on its own would carry.
 */
class PathDecoderTest {
    private fun wellFormed(): JSONObject = JSONObject().apply {
        put("timestamp_seconds", 12.345)
        put("times_seconds", JSONArray(listOf(0.0, 0.1, 0.2)))
        put("lateral_offsets_meters", JSONArray(listOf(0.0, 0.05, 0.12)))
        put("first_heading_radians", 0.0423)
        put("alarm", true)
        put("cumulative_cost_bits", 18.4)
    }

    private fun bytes(json: JSONObject): ByteArray = json.toString().toByteArray(Charsets.UTF_8)

    private fun refusal(payload: ByteArray): String =
        assertFailsWith<PathDecodeException> { PathDecoder.decode(payload) }.message!!

    private fun refusal(json: JSONObject): String = refusal(bytes(json))

    private fun framed(payload: ByteArray, declaredLength: Int = payload.size): DataInputStream {
        val out = ByteArrayOutputStream()
        DataOutputStream(out).use {
            it.writeInt(declaredLength)
            it.write(payload)
        }
        return DataInputStream(ByteArrayInputStream(out.toByteArray()))
    }

    @Test
    fun aWellFormedMessageDecodesToItsValues() {
        val path = PathDecoder.decode(bytes(wellFormed()))

        assertEquals(12.345, path.timestampSeconds)
        assertTrue(path.timesSeconds.contentEquals(doubleArrayOf(0.0, 0.1, 0.2)))
        assertTrue(path.lateralOffsetsMeters.contentEquals(doubleArrayOf(0.0, 0.05, 0.12)))
        assertEquals(0.0423, path.firstHeadingRadians)
        assertEquals(true, path.alarm)
        assertEquals(18.4, path.cumulativeCostBits)
    }

    @Test
    fun theLaptopsFixtureDecodesToTheStatedValues() {
        // Written by the laptop's test_laptop_path_fixture.py from stated values. The literals
        // here are those values typed again, not read from anything the encoder produced.
        val fixture = File(System.getProperty("laptopPathFixturePath") ?: error("laptopPathFixturePath is not set"))
        assertTrue(fixture.isFile, "${fixture.path} is missing. Run the laptop's tests first, they write it")

        val input = DataInputStream(fixture.inputStream().buffered())
        val path = input.use { stream ->
            val payload = PathDecoder.readFramed(stream)
            assertEquals(-1, stream.read(), "one framed message and nothing after it")
            PathDecoder.decode(payload)
        }

        assertEquals(12.345, path.timestampSeconds)
        assertTrue(path.timesSeconds.contentEquals(doubleArrayOf(0.0, 0.1, 0.2)))
        assertTrue(path.lateralOffsetsMeters.contentEquals(doubleArrayOf(0.0, 0.05, 0.12)))
        assertEquals(0.0423, path.firstHeadingRadians)
        assertEquals(true, path.alarm)
        assertEquals(18.4, path.cumulativeCostBits)
    }

    @Test
    fun aMissingKeyIsRefusedByName() {
        for (key in listOf("timestamp_seconds", "times_seconds", "lateral_offsets_meters", "first_heading_radians", "alarm", "cumulative_cost_bits")) {
            val json = wellFormed()
            json.remove(key)

            assertTrue(refusal(json).contains("'$key'"), "missing $key: ${refusal(json)}")
        }
    }

    @Test
    fun alarmAsANumberIsRefused() {
        val message = refusal(wellFormed().put("alarm", 1))
        assertTrue(message.contains("alarm") && message.contains("true or false"), message)
    }

    @Test
    fun aNumberAsAStringIsRefused() {
        // getDouble would have parsed "0.04". The raw type is checked first.
        val message = refusal(wellFormed().put("first_heading_radians", "0.04"))
        assertTrue(message.contains("first_heading_radians") && message.contains("a string"), message)
    }

    @Test
    fun aNumberAsABooleanIsRefused() {
        val message = refusal(wellFormed().put("cumulative_cost_bits", true))
        assertTrue(message.contains("cumulative_cost_bits") && message.contains("a boolean"), message)
    }

    @Test
    fun aNullNumberIsRefused() {
        val message = refusal(wellFormed().put("timestamp_seconds", JSONObject.NULL))
        assertTrue(message.contains("timestamp_seconds") && message.contains("null"), message)
    }

    @Test
    fun anArrayThatIsNotAnArrayIsRefused() {
        val message = refusal(wellFormed().put("times_seconds", 0.1))
        assertTrue(message.contains("times_seconds") && message.contains("a list"), message)
    }

    @Test
    fun aBadArrayEntryIsRefusedWithItsIndex() {
        val message = refusal(wellFormed().put("times_seconds", JSONArray(listOf(0.0, "x", 0.2))))
        assertTrue(message.contains("times_seconds[1]"), message)
    }

    @Test
    fun mismatchedArrayLengthsAreRefused() {
        val message = refusal(wellFormed().put("lateral_offsets_meters", JSONArray(listOf(0.0, 0.05))))
        assertTrue(message.contains("3 entries") && message.contains("has 2"), message)
    }

    @Test
    fun anEmptyPathIsRefused() {
        val json = wellFormed().put("times_seconds", JSONArray()).put("lateral_offsets_meters", JSONArray())
        assertTrue(refusal(json).contains("at least one entry"), refusal(json))
    }

    @Test
    fun aPayloadThatIsNotAnObjectIsRefused() {
        assertTrue(refusal("[1, 2]".toByteArray(Charsets.UTF_8)).contains("not a JSON object"))
        assertTrue(refusal("not json at all".toByteArray(Charsets.UTF_8)).contains("not a JSON object"))
    }

    @Test
    fun aPayloadThatIsNotUtf8IsRefused() {
        val message = refusal(byteArrayOf(0xFF.toByte(), 0xFE.toByte(), '{'.code.toByte(), '}'.code.toByte()))
        assertTrue(message.contains("not valid UTF-8"), message)
    }

    @Test
    fun readFramedReturnsThePayloadBehindItsPrefix() {
        val payload = bytes(wellFormed())

        assertTrue(PathDecoder.readFramed(framed(payload)).contentEquals(payload))
    }

    @Test
    fun aZeroLengthPrefixIsAStreamOutOfStep() {
        val message = assertFailsWith<IOException> { PathDecoder.readFramed(framed(ByteArray(0), declaredLength = 0)) }.message!!
        assertTrue(message.contains("is 0") && message.contains("out of step"), message)
    }

    @Test
    fun aLengthPrefixOverTheCapIsAStreamOutOfStep() {
        val message = assertFailsWith<IOException> { PathDecoder.readFramed(framed(ByteArray(0), declaredLength = PathDecoder.MAX_PATH_BYTES + 1)) }.message!!
        assertTrue(message.contains("${PathDecoder.MAX_PATH_BYTES + 1} bytes") && message.contains("out of step"), message)
    }

    @Test
    fun aTruncatedPayloadEndsTheStreamRatherThanHanging() {
        val payload = bytes(wellFormed())
        // The sender closed after the bad prefix, which is what a ByteArrayInputStream models.
        // A read that waited for the missing bytes would block forever on a socket.
        assertFailsWith<EOFException> { PathDecoder.readFramed(framed(payload.copyOf(10), declaredLength = payload.size)) }
    }
}
