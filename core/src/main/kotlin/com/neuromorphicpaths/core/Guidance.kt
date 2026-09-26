package com.neuromorphicpaths.core

/**
 * One obstacle's push on the walker, in the walker's frame. Lateral is positive to the right,
 * forward is positive ahead. Unitless. Only the direction and the relative sizes matter.
 */
data class Push(
    val lateral: Double,
    val forward: Double,
)

/** What one obstacle contributed: its own surprise and the push that came out of it. */
data class ObstacleSurprise(
    val obstacle: Obstacle,
    val surpriseBits: Double,
    val push: Push,
)

/**
 * The field's answer for one frame.
 *
 * Desired heading is radians relative to the camera's forward axis, positive to the right.
 * Overall surprise is how improbable the walker's actual heading is given the field, in bits.
 * The per-obstacle list is for the display and the log, not for adding up. Walker wobble is the
 * running spread of the walker's own heading, null until measured. Turn tolerance is the value
 * the field charged turns against on this frame, null for a field that has no such term.
 * Heading entropy is how spread out the field's belief over headings is, in bits: near zero
 * when one heading is clearly best, near the log of the number of candidates when they are
 * all about as good. Null for a field that holds no such belief. Heading information is how
 * far the scene moved that belief away from the walker's own prior, in bits, the divergence of
 * the posterior from the prior: zero with nothing in view, however spread the belief is, and
 * large when something in view reshapes where the walker would go. Null for a field with no
 * prior to move from. The projected path is where the field would send the walker over the
 * next few meters, rolled forward a step at a time, empty for a field that does not project.
 */
data class Guidance(
    val desiredHeadingRadians: Double,
    val overallSurpriseBits: Double,
    val perObstacle: List<ObstacleSurprise>,
    val walkerWobbleRadians: Double? = null,
    val turnToleranceRadians: Double? = null,
    val headingEntropyBits: Double? = null,
    val headingInformationBits: Double? = null,
    val projectedPath: List<PathPoint> = emptyList(),
)

/**
 * One step of the projected path: where the virtual walker stands after it, in the walker's
 * frame, which heading the field chose there, and how much the scene reshaped the belief at
 * that step, in bits. A display fades a step by its information and colors the path by the
 * frame's surprise, two different things kept on two different channels.
 */
data class PathPoint(
    val forwardMeters: Double,
    val rightMeters: Double,
    val headingRadians: Double,
    val informationBits: Double,
)
