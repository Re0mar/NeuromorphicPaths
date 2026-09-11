package com.example.neuromorphicpaths.steering

data class DebouncerConfig(
    val minDwellTimeMs: Long = 1000L,
    val consecutiveRequired: Int = 3
)

class DirectionDebouncer(private val config: DebouncerConfig = DebouncerConfig()) {
    private var currentCommand: DirectionCommand = DirectionCommand.STRAIGHT
    private var candidateCommand: DirectionCommand? = null
    private var consecutiveCount: Int = 0
    private var lastCommandChangeTimeMs: Long = 0L

    fun debounce(detectedCommand: DirectionCommand, timestampMs: Long): DirectionCommand {
        if (lastCommandChangeTimeMs == 0L) {
            lastCommandChangeTimeMs = timestampMs
            currentCommand = detectedCommand
            return currentCommand
        }

        if (detectedCommand == currentCommand) {
            candidateCommand = null
            consecutiveCount = 0
            return currentCommand
        }

        if (detectedCommand == candidateCommand) {
            consecutiveCount++
        } else {
            candidateCommand = detectedCommand
            consecutiveCount = 1
        }

        val timeSinceLastChange = timestampMs - lastCommandChangeTimeMs
        if (consecutiveCount >= config.consecutiveRequired && timeSinceLastChange >= config.minDwellTimeMs) {
            currentCommand = detectedCommand
            lastCommandChangeTimeMs = timestampMs
            candidateCommand = null
            consecutiveCount = 0
        }

        return currentCommand
    }

    fun reset() {
        currentCommand = DirectionCommand.STRAIGHT
        candidateCommand = null
        consecutiveCount = 0
        lastCommandChangeTimeMs = 0L
    }
}
