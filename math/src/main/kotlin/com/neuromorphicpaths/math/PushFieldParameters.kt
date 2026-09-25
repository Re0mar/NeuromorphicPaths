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
 *
 * The turn tolerance can instead follow the walker's measured heading wobble, as the wobble times
 * a ratio, clamped between a floor and the search range. That is off by default on purpose. On
 * the first outdoor recording the 1 s wobble has a median of 4 degrees, which as a tolerance
 * would charge a 20 degree turn about 4 bits, against 0.7 bits for an obstacle two seconds
 * ahead, so the field would hold its line into most obstacles. Whether the mapping, the window
 * or the ratio is wrong is an open question, and the switch is here so it can be tried.
 */
data class PushFieldParameters(
    val clearanceMeters: Double = 0.5,
    val referenceTimeSeconds: Double = 2.0,
    val minimumTimeToContactSeconds: Double = 0.2,
    val turnToleranceRadians: Double = Math.toRadians(30.0),
    val turnToleranceFromWobble: Boolean = false,
    val wobbleToToleranceRatio: Double = WELFORD_BAND_Z,
    val minimumTurnToleranceRadians: Double = Math.toRadians(5.0),
    val defaultWalkerSpeedMetersPerSecond: Double = 1.4,
    val maxHeadingRadians: Double = Math.toRadians(60.0),
    val headingStepRadians: Double = Math.toRadians(1.0),
) {
    init {
        require(clearanceMeters > 0.0) { "clearanceMeters must be positive" }
        require(referenceTimeSeconds > 0.0) { "referenceTimeSeconds must be positive" }
        require(minimumTimeToContactSeconds > 0.0) { "minimumTimeToContactSeconds must be positive" }
        require(turnToleranceRadians > 0.0) { "turnToleranceRadians must be positive" }
        require(wobbleToToleranceRatio > 0.0) { "wobbleToToleranceRatio must be positive" }
        require(minimumTurnToleranceRadians > 0.0) { "minimumTurnToleranceRadians must be positive" }
        require(defaultWalkerSpeedMetersPerSecond > 0.0) { "defaultWalkerSpeedMetersPerSecond must be positive" }
        require(maxHeadingRadians > 0.0 && headingStepRadians > 0.0) { "heading search range and step must be positive" }
    }

    companion object {
        /** Welford's convention: 96 percent of a walker's headings fall within this many spreads. */
        const val WELFORD_BAND_Z = 2.07
    }
}
