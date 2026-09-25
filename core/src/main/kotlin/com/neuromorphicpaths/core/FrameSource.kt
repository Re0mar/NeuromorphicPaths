package com.neuromorphicpaths.core

import kotlinx.coroutines.flow.Flow

/**
 * The input contract. Anything that can produce frames: a camera, a recording, a network stream.
 *
 * The flow is cold. Collecting it starts the source, and cancelling the collector stops it.
 * A source emits as fast as it can and leaves dropping to the consumer.
 */
interface FrameSource : AutoCloseable {
    /** Short name for logs and the screen, such as "camera" or the file name. */
    val name: String

    fun frames(): Flow<Frame>
}
