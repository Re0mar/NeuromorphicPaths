package com.neuromorphicpaths.output

/**
 * Decides whether an update with no detections replaces what is on screen. It waits until the
 * last update with detections has been shown for the hold time, so one empty frame doesn't blink
 * the boxes and the arrow off and back on.
 */
class OverlayHold(private val holdMillis: Long = DEFAULT_HOLD_MILLIS) {
    private var lastDetectionsShownAtMillis: Long? = null

    fun shouldShow(hasDetections: Boolean, nowMillis: Long): Boolean {
        if (hasDetections) {
            lastDetectionsShownAtMillis = nowMillis
            return true
        }
        val lastDetections = lastDetectionsShownAtMillis ?: return true
        return nowMillis - lastDetections >= holdMillis
    }

    fun reset() {
        lastDetectionsShownAtMillis = null
    }

    companion object {
        // Two detector periods at the 640 model's 2.4 frames a second on the phone. One or two
        // empty results are covered, and something that really left the view is gone within a second.
        const val DEFAULT_HOLD_MILLIS = 800L
    }
}
