package com.example.neuromorphicpaths.display

import com.example.neuromorphicpaths.steering.DirectionCommand

class FeedbackRouter(
    private val displayRenderer: GlassesDisplayRenderer?,
    private val audioFeedbackManager: AudioFeedbackManager?,
    var isDisplayEnabled: Boolean = true,
    var isAudioEnabled: Boolean = true,
    private val minFeedbackIntervalMs: Long = 2500L
) {
    private var lastRoutedCommand: DirectionCommand? = null
    private var lastFeedbackTimeMs: Long = 0L

    fun routeFeedback(command: DirectionCommand, timestampMs: Long) {
        val isCommandChanged = command != lastRoutedCommand
        val isIntervalPassed = (timestampMs - lastFeedbackTimeMs) >= minFeedbackIntervalMs

        if (isCommandChanged || isIntervalPassed) {
            if (isDisplayEnabled) {
                displayRenderer?.renderCommand(command)
            }
            if (isAudioEnabled) {
                audioFeedbackManager?.speakCommand(command)
            }
            lastRoutedCommand = command
            lastFeedbackTimeMs = timestampMs
        }
    }
}
