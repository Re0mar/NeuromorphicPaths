package com.neuromorphicpaths.pixel.wire

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.json.JSONObject

/**
 * The encoder against the wire format, and a fixture for the laptop to decode.
 *
 * These tests check the framing the document describes. The other half of the contract, that the
 * laptop decodes what this produced, lives in the laptop's own suite, which reads the fixture
 * this test writes. A message that only round trips through our own code proves nothing about
 * the laptop, so the fixture is the test that matters.
 */
class FrameEncoderTest {
    private fun sampleMessage(): DepthMessage = DepthMessage(
        timestampSeconds = 12.345,
        depthMillimeters = shortArrayOf(1500, 2000, 2500, 3000),
        rows = 2,
        columns = 2,
        intrinsics = doubleArrayOf(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0),
        orientationWxyz = doubleArrayOf(1.0, 0.0, 0.0, 0.0),
        positionXyz = doubleArrayOf(0.0, 0.0, 0.0),
        groundPlane = GroundPlane(normal = doubleArrayOf(0.0, -1.0, 0.0), offsetMeters = 1.6),
    )

    @Test
    fun lengthPrefixIsBigEndianAndCountsEverythingAfterIt() {
        val message = FrameEncoder.encode(sampleMessage())
        val declared = ByteBuffer.wrap(message, 0, 4).order(ByteOrder.BIG_ENDIAN).int
        assertEquals(message.size - 4, declared)
    }

    @Test
    fun headerIsJsonThenOneNewlineThenExactlyTheDepthBytes() {
        val message = FrameEncoder.encode(sampleMessage())
        val payload = message.copyOfRange(4, message.size)
        val newline = payload.indexOf('\n'.code.toByte())
        assertTrue(newline > 0, "a newline must separate the header from the depth")

        val header = JSONObject(String(payload, 0, newline, Charsets.UTF_8))
        val depthBytes = payload.copyOfRange(newline + 1, payload.size)
        assertEquals(header.getJSONObject("depth").getInt("byte_length"), depthBytes.size)
        assertEquals(8, depthBytes.size)
    }

    @Test
    fun depthIsLittleEndianUint16Millimeters() {
        val message = FrameEncoder.encode(sampleMessage())
        val payload = message.copyOfRange(4, message.size)
        val depthBytes = payload.copyOfRange(payload.indexOf('\n'.code.toByte()) + 1, payload.size)
        // 1500 is 0x05DC, little-endian DC 05.
        assertEquals(0xDC.toByte(), depthBytes[0])
        assertEquals(0x05.toByte(), depthBytes[1])
    }

    @Test
    fun everyRequiredKeyIsPresentIncludingTheNullOnes() {
        val header = headerOf(FrameEncoder.encode(sampleMessage()))
        for (key in listOf("version", "timestamp_seconds", "depth", "intrinsics", "pose", "ground_plane", "gaze_pixel")) {
            assertTrue(header.has(key), "header must carry $key")
        }
        assertTrue(header.isNull("gaze_pixel"))
        assertEquals(1, header.getInt("version"))
        assertEquals("uint16", header.getJSONObject("depth").getString("dtype"))
    }

    @Test
    fun lostTrackingSendsANullPositionAndSaysSo() {
        val header = headerOf(FrameEncoder.encode(sampleMessage().copy(positionXyz = null)))
        val pose = header.getJSONObject("pose")
        assertTrue(pose.isNull("position_xyz"))
        assertEquals(false, pose.getBoolean("has_position"))
    }

    @Test
    fun aDepthArrayOfTheWrongSizeIsRefused() {
        val failed = runCatching { sampleMessage().copy(depthMillimeters = shortArrayOf(1, 2, 3)) }
        assertTrue(failed.isFailure, "three values cannot be a 2 by 2 image")
    }

    @Test
    fun writesTheFixtureTheLaptopDecodes() {
        // The laptop's suite reads this file and decodes it with the real decoder. Regenerate it by
        // running this test, and commit it with the change that moved it.
        val target = File(System.getProperty("wireFixturePath") ?: "build/pixel_app_frame.bin")
        target.parentFile?.mkdirs()
        target.writeBytes(FrameEncoder.encode(sampleMessage()))
        assertTrue(target.length() > 4)
    }

    private fun headerOf(message: ByteArray): JSONObject {
        val payload = message.copyOfRange(4, message.size)
        return JSONObject(String(payload, 0, payload.indexOf('\n'.code.toByte()), Charsets.UTF_8))
    }
}
