package com.example.sidewalkvision

// The first app's value. Long enough to cover one or two empty detections, short enough that a
// path that really disappeared stops showing within half a second.
private const val OVERLAY_HOLD_MS = 400L

/**
 * Decides whether a new detection replaces the one on screen. A result with no path waits until
 * the last path has been shown for the hold time, so one empty frame doesn't blink the overlay off.
 */
class OverlayHold(private val holdMillis: Long = OVERLAY_HOLD_MS) {
    private var lastPathShownAtMillis: Long? = null

    fun shouldShow(hasPath: Boolean, nowMillis: Long): Boolean {
        if (hasPath) {
            lastPathShownAtMillis = nowMillis
            return true
        }
        val lastPath = lastPathShownAtMillis ?: return true
        return nowMillis - lastPath >= holdMillis
    }

    fun reset() {
        lastPathShownAtMillis = null
    }
}
