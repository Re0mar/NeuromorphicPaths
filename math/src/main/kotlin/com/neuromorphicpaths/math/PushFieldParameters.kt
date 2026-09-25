package com.neuromorphicpaths.math

/**
 * The tuning knobs of the push field. Every one has a physical meaning, so a value can be
 * argued about in meters and seconds rather than found by trial.
 *
 * Clearance is how far an object has to sit from the walker's line for the walker to be
 * comfortable. Reference time is the time to contact at which a dead-ahead object is worth
 * about 0.7 bits of surprise. Turn tolerance is how far the walker expects to deviate from
 * straight ahead before that alone feels wrong. Default speed stands in until something
 * measures the walker.
 */
data class PushFieldParameters(
    val clearanceMeters: Double = 0.5,
    val referenceTimeSeconds: Double = 2.0,
    val minimumTimeToContactSeconds: Double = 0.2,
    val turnToleranceRadians: Double = Math.toRadians(30.0),
    val defaultWalkerSpeedMetersPerSecond: Double = 1.4,
    val maxHeadingRadians: Double = Math.toRadians(60.0),
    val headingStepRadians: Double = Math.toRadians(1.0),
) {
    init {
        require(clearanceMeters > 0.0) { "clearanceMeters must be positive" }
        require(referenceTimeSeconds > 0.0) { "referenceTimeSeconds must be positive" }
        require(minimumTimeToContactSeconds > 0.0) { "minimumTimeToContactSeconds must be positive" }
        require(turnToleranceRadians > 0.0) { "turnToleranceRadians must be positive" }
        require(defaultWalkerSpeedMetersPerSecond > 0.0) { "defaultWalkerSpeedMetersPerSecond must be positive" }
        require(maxHeadingRadians > 0.0 && headingStepRadians > 0.0) { "heading search range and step must be positive" }
    }
}
