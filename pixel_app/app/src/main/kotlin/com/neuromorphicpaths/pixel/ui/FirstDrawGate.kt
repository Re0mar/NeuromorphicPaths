package com.neuromorphicpaths.pixel.ui

/**
 * Says when a draw pass is the first one to use a given item.
 *
 * The overlay redraws far more often than paths arrive: every recomposition, every age tick. The
 * frame-to-arrow measurement wants the first of those draws per path, so this compares by
 * identity. Two paths with identical numbers are still two arrivals.
 */
class FirstDrawGate<T : Any> {
    private var lastDrawn: T? = null

    fun isFirstDraw(item: T?): Boolean {
        if (item == null || item === lastDrawn) return false
        lastDrawn = item
        return true
    }
}
