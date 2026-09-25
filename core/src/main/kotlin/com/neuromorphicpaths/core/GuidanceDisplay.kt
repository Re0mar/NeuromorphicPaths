package com.neuromorphicpaths.core

/**
 * The output contract. Receives one update per processed frame.
 *
 * Suspending so a display can do its own conversion work off the pipeline's thread. The pipeline
 * calls displays in order and waits for each, so a slow display slows everything behind it.
 */
interface GuidanceDisplay : AutoCloseable {
    val name: String

    suspend fun show(update: GuidanceUpdate)
}
