package com.neuromorphicpaths.pixel.wire

import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject

/**
 * Encodes a [DepthMessage] in the wire format the laptop decodes, `server/docs/arcore_wire_format.md`.
 *
 * One message is a 4-byte big-endian length, then a UTF-8 JSON header, one newline, then the depth
 * bytes. Depth goes as `uint16` millimeters, little-endian, which is what ARCore hands over and what
 * the laptop divides by 1000. Every nullable field is still a key, written as `null`, because the
 * laptop treats a missing key as an error and an absent value as a value.
 */
object FrameEncoder {
    const val WIRE_VERSION = 1
    private const val DEPTH_DTYPE = "uint16"
    private const val BYTES_PER_DEPTH_VALUE = 2

    /** The complete message, length prefix included. */
    fun encode(message: DepthMessage): ByteArray {
        val depthBytes = depthBytes(message.depthMillimeters)
        val header = header(message, depthBytes.size).toString().toByteArray(Charsets.UTF_8)

        val payload = ByteArrayOutputStream(4 + header.size + 1 + depthBytes.size)
        DataOutputStream(payload).use { out ->
            // DataOutputStream.writeInt is big-endian, which is network order and what the
            // laptop's struct.Struct(">I") reads.
            out.writeInt(header.size + 1 + depthBytes.size)
            out.write(header)
            out.write('\n'.code)
            out.write(depthBytes)
        }
        return payload.toByteArray()
    }

    private fun header(message: DepthMessage, byteLength: Int): JSONObject = JSONObject().apply {
        put("version", WIRE_VERSION)
        put("timestamp_seconds", message.timestampSeconds)
        put(
            "depth",
            JSONObject().apply {
                put("dtype", DEPTH_DTYPE)
                put("shape", JSONArray(listOf(message.rows, message.columns)))
                put("byte_length", byteLength)
            },
        )
        put("intrinsics", matrixRows(message.intrinsics))
        put(
            "pose",
            JSONObject().apply {
                put("orientation_wxyz", JSONArray(message.orientationWxyz.toList()))
                put("position_xyz", message.positionXyz?.let { JSONArray(it.toList()) } ?: JSONObject.NULL)
                put("has_position", message.hasPosition)
            },
        )
        put(
            "ground_plane",
            message.groundPlane?.let { plane ->
                JSONObject().apply {
                    put("normal", JSONArray(plane.normal.toList()))
                    put("offset_meters", plane.offsetMeters)
                }
            } ?: JSONObject.NULL,
        )
        // The Pixel has no eye tracker. The key is still required, so it is null.
        put("gaze_pixel", JSONObject.NULL)
    }

    private fun matrixRows(flat: DoubleArray): JSONArray = JSONArray().apply {
        for (row in 0 until 3) {
            put(JSONArray(listOf(flat[row * 3], flat[row * 3 + 1], flat[row * 3 + 2])))
        }
    }

    private fun depthBytes(depth: ShortArray): ByteArray {
        // Explicitly little-endian. The wire format says so, and relying on the device being
        // little-endian is not the same as the format saying so.
        val buffer = ByteBuffer.allocate(depth.size * BYTES_PER_DEPTH_VALUE).order(ByteOrder.LITTLE_ENDIAN)
        buffer.asShortBuffer().put(depth)
        return buffer.array()
    }
}
