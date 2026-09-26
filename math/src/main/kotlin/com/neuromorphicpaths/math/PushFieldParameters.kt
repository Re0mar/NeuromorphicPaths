package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.ObstacleClass
import com.neuromorphicpaths.core.SurfaceClass

/**
 * The tuning knobs of the push field. Every one has a physical meaning, so a value can be
 * argued about in meters and seconds rather than found by trial.
 *
 * What each class of obstacle means is in [profiles], one [ObstacleProfile] per class: the
 * berth it needs, how far ahead in time it starts to count, and whether touching it is fine.
 * Turn tolerance is the spread of the prior on the heading, how far the walker expects to
 * deviate from straight ahead before that alone feels wrong. Default speed stands in until
 * something measures the walker, and a measured speed is floored at the minimum, so a walker
 * standing still is treated as barely moving: standing objects then fade to nothing through
 * their time to contact, while a person walking toward the walker keeps their own closing
 * speed, and the arrow still points somewhere on the first step.
 *
 * A detector's score is not a probability, so it is calibrated before it enters the field: a
 * score at or above [confidenceForCertainty] counts as an object that certainly exists, and
 * below it the probability falls linearly, so the detector's own 0.25 threshold reads as an
 * even chance. Taken literally, a cone a meter ahead scored at 0.45 could never be worth more
 * than a bit, however close it came.
 *
 * The walker's measured heading wobble is reported but never sets the tolerance. That was tried
 * on the first outdoor recording: the sway-derived 8 degrees made no calls at all and missed the
 * cone the walker swerved around, and every mapping with a floor landed between the fixed
 * values with more noise. Sway and tolerance are different quantities.
 *
 * The ground ahead costs [surfaceCostBitsPerMeter] for every meter of the next
 * [surfaceLookaheadSeconds] spent on each surface, sampled every [surfaceStepMeters]. Pavement
 * and unknown ground are free, so grass is crossed to get away from a wall and left alone
 * otherwise. A cost of 0.1 bits per meter says each meter of grass is fine with probability 0.93.
 *
 * The projected path rolls the field forward [pathSteps] times, [pathStepMeters] each, so with
 * the defaults it reaches 3 m, about two seconds of walking and the same distance the surface
 * term looks ahead.
 */
data class PushFieldParameters(
    val profiles: Map<ObstacleClass, ObstacleProfile> = ObstacleProfile.DEFAULTS,
    val confidenceForCertainty: Double = 0.5,
    val minimumTimeToContactSeconds: Double = 0.2,
    val turnToleranceRadians: Double = Math.toRadians(30.0),
    val defaultWalkerSpeedMetersPerSecond: Double = 1.4,
    val minimumWalkerSpeedMetersPerSecond: Double = 0.3,
    val maxHeadingRadians: Double = Math.toRadians(60.0),
    val headingStepRadians: Double = Math.toRadians(1.0),
    val surfaceCostBitsPerMeter: Map<SurfaceClass, Double> = DEFAULT_SURFACE_COSTS,
    val surfaceLookaheadSeconds: Double = 2.0,
    val surfaceStepMeters: Double = 0.25,
    val pathSteps: Int = 6,
    val pathStepMeters: Double = 0.5,
) {
    init {
        // A class without a profile would surface as a lookup failure deep inside the search.
        val missingProfiles = ObstacleClass.entries.filterNot { it in profiles }
        require(missingProfiles.isEmpty()) { "profiles is missing $missingProfiles" }
        val missingSurfaces = SurfaceClass.entries.filterNot { it in surfaceCostBitsPerMeter }
        require(missingSurfaces.isEmpty()) { "surfaceCostBitsPerMeter is missing $missingSurfaces" }
        require(surfaceCostBitsPerMeter.values.all { it >= 0.0 }) { "surface costs cannot be negative" }
        require(confidenceForCertainty > 0.0 && confidenceForCertainty <= 1.0) { "confidenceForCertainty must be in (0, 1]" }
        require(minimumTimeToContactSeconds > 0.0) { "minimumTimeToContactSeconds must be positive" }
        require(turnToleranceRadians > 0.0) { "turnToleranceRadians must be positive" }
        require(defaultWalkerSpeedMetersPerSecond > 0.0) { "defaultWalkerSpeedMetersPerSecond must be positive" }
        require(minimumWalkerSpeedMetersPerSecond > 0.0) { "minimumWalkerSpeedMetersPerSecond must be positive" }
        require(maxHeadingRadians > 0.0 && headingStepRadians > 0.0) { "heading search range and step must be positive" }
        require(surfaceLookaheadSeconds > 0.0 && surfaceStepMeters > 0.0) { "surface lookahead and step must be positive" }
        require(pathSteps >= 0 && pathStepMeters > 0.0) { "path steps cannot be negative and the step must be positive" }
    }

    fun profileOf(obstacleClass: ObstacleClass): ObstacleProfile = profiles.getValue(obstacleClass)

    fun surfaceCostOf(surfaceClass: SurfaceClass): Double = surfaceCostBitsPerMeter.getValue(surfaceClass)

    companion object {
        /**
         * Bits per meter on each surface. Grass and dirt are a mild preference against, a road is
         * a place not to be, and nothing is charged for ground nobody has classified.
         */
        val DEFAULT_SURFACE_COSTS: Map<SurfaceClass, Double> = mapOf(
            SurfaceClass.PAVEMENT to 0.0,
            SurfaceClass.GRASS to 0.1,
            SurfaceClass.DIRT to 0.15,
            SurfaceClass.ROAD to 0.6,
            SurfaceClass.UNKNOWN to 0.0,
        )
    }
}
