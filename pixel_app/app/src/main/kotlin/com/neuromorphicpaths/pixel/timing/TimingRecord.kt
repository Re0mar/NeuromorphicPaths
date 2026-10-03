package com.neuromorphicpaths.pixel.timing

import org.json.JSONObject

/**
 * One line of the phone's timing log, as JSON.
 *
 * The laptop's timing report reads these lines and joins them with its own log on `frame_ns`, the
 * ARCore frame timestamp in nanoseconds. Every other `*_ns` field is `SystemClock.elapsedRealtimeNanos()`,
 * so any two of them subtract without a conversion. The field names below are the contract with the
 * report. The laptop declares the same names, and the fixture this app's test writes into
 * `server/tests/fixtures/pixel_app_timing.jsonl` is what the laptop's suite checks them against.
 *
 * Lines are built by hand rather than through [JSONObject]. Android's `JSONObject` keeps insertion
 * order and the JVM library the tests run against does not, so a hand-built line is the only way the
 * fixture comes out the same on both.
 */
sealed interface TimingRecord {
    fun toJsonLine(): String

    /** The first line of every log. Says which phone, which build and when, so a pulled file explains itself. */
    data class Session(val startedWall: String, val device: String, val buildType: String) : TimingRecord {
        override fun toJsonLine(): String =
            "{${field(TYPE, quoted(TYPE_SESSION))},${field(SCHEMA, SCHEMA_VERSION.toString())}," +
                "${field(STARTED_WALL, quoted(startedWall))},${field(DEVICE, quoted(device))}," +
                "${field(BUILD_TYPE, quoted(buildType))}}"
    }

    /** The app handled a new ARCore frame. Written for every frame, with or without depth. */
    data class Frame(val frameNanos: Long, val handledNanos: Long) : TimingRecord {
        override fun toJsonLine(): String = stamped(TYPE_FRAME, frameNanos, HANDLED_NS, handledNanos)
    }

    /** The frame's depth message finished writing to the laptop's socket. */
    data class Sent(val frameNanos: Long, val sentNanos: Long) : TimingRecord {
        override fun toJsonLine(): String = stamped(TYPE_SENT, frameNanos, SENT_NS, sentNanos)
    }

    /** A newer frame replaced this one in the send slot before it went. It never reached the laptop. */
    data class Dropped(val frameNanos: Long) : TimingRecord {
        override fun toJsonLine(): String = "{${field(TYPE, quoted(TYPE_DROPPED))},${field(FRAME_NS, frameNanos.toString())}}"
    }

    /** The path planned from this frame arrived back on the phone. */
    data class Received(val frameNanos: Long, val receivedNanos: Long) : TimingRecord {
        override fun toJsonLine(): String = stamped(TYPE_RECEIVED, frameNanos, RECEIVED_NS, receivedNanos)
    }

    /** The arrow was drawn from this frame's path for the first time. */
    data class Drawn(val frameNanos: Long, val drawnNanos: Long) : TimingRecord {
        override fun toJsonLine(): String = stamped(TYPE_DRAWN, frameNanos, DRAWN_NS, drawnNanos)
    }

    /** This many records were dropped because the writer had fallen behind. Written before the next line that made it. */
    data class Lost(val count: Int) : TimingRecord {
        override fun toJsonLine(): String = "{${field(TYPE, quoted(TYPE_LOST))},${field(COUNT, count.toString())}}"
    }

    /** The log hit its size cap here and nothing after this line was written. */
    data class Truncated(val atBytes: Long) : TimingRecord {
        override fun toJsonLine(): String = "{${field(TYPE, quoted(TYPE_TRUNCATED))},${field(AT_BYTES, atBytes.toString())}}"
    }

    companion object {
        const val SCHEMA_VERSION = 1

        const val TYPE = "type"
        const val SCHEMA = "schema"
        const val STARTED_WALL = "started_wall"
        const val DEVICE = "device"
        const val BUILD_TYPE = "build_type"
        const val FRAME_NS = "frame_ns"
        const val HANDLED_NS = "handled_ns"
        const val SENT_NS = "sent_ns"
        const val RECEIVED_NS = "received_ns"
        const val DRAWN_NS = "drawn_ns"
        const val COUNT = "count"
        const val AT_BYTES = "at_bytes"

        const val TYPE_SESSION = "session"
        const val TYPE_FRAME = "frame"
        const val TYPE_SENT = "sent"
        const val TYPE_DROPPED = "dropped"
        const val TYPE_RECEIVED = "received"
        const val TYPE_DRAWN = "drawn"
        const val TYPE_LOST = "lost"
        const val TYPE_TRUNCATED = "truncated"

        private const val NANOSECONDS_PER_SECOND = 1.0e9

        /**
         * The frame's ARCore timestamp in nanoseconds, back from the seconds it travels in.
         *
         * `ArCoreToWire` divides the `Long` by 1e9 to fill `timestamp_seconds`. A double holds that
         * exactly enough for rounding to recover the original nanoseconds at any realistic uptime.
         * The laptop's `frame_ns_from_seconds` is this function's twin, and the two must agree.
         */
        fun frameNanosFromSeconds(timestampSeconds: Double): Long = Math.round(timestampSeconds * NANOSECONDS_PER_SECOND)

        private fun quoted(value: String): String = JSONObject.quote(value)

        private fun field(name: String, jsonValue: String): String = "${quoted(name)}:$jsonValue"

        private fun stamped(type: String, frameNanos: Long, stampName: String, stampNanos: Long): String =
            "{${field(TYPE, quoted(type))},${field(FRAME_NS, frameNanos.toString())},${field(stampName, stampNanos.toString())}}"
    }
}
