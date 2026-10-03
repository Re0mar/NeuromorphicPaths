package com.neuromorphicpaths.pixel.timing

/**
 * Where the app reports the moments a frame-to-arrow measurement needs.
 *
 * Every method is called on a thread that must not wait: the GL thread, the sender, the path
 * reader, the draw pass. An implementation reads its clock and hands the record off, nothing more.
 * Each takes the frame's ARCore timestamp in nanoseconds, which is how the laptop's log is joined.
 */
interface TimingRecorder {
    fun frameHandled(frameNanos: Long)

    fun frameSent(frameNanos: Long)

    fun frameDropped(frameNanos: Long)

    fun pathReceived(frameNanos: Long)

    fun pathDrawn(frameNanos: Long)

    fun close()

    /** Records nothing. What the app uses when it has nowhere to write a log. */
    object None : TimingRecorder {
        override fun frameHandled(frameNanos: Long) = Unit

        override fun frameSent(frameNanos: Long) = Unit

        override fun frameDropped(frameNanos: Long) = Unit

        override fun pathReceived(frameNanos: Long) = Unit

        override fun pathDrawn(frameNanos: Long) = Unit

        override fun close() = Unit
    }
}
