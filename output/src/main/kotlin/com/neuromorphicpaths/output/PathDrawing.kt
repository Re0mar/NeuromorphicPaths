package com.neuromorphicpaths.output

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.clipRect
import com.neuromorphicpaths.core.GroundPlaneProjection
import com.neuromorphicpaths.core.GroundPoint
import com.neuromorphicpaths.core.GuidanceUpdate
import kotlin.math.max
import kotlin.math.min

// The projected path's ribbon, shared by the app's own screen and the full-screen floating
// overlay, so both draw exactly the same shape and colors.

/** Maps normalized image coordinates onto the rectangle [ContentScale.Fit] gives the image. */
internal class FittedImage(imageWidth: Int, imageHeight: Int, canvas: Size) {
    private val scale = min(canvas.width / imageWidth, canvas.height / imageHeight)
    val drawnWidth = imageWidth * scale
    val drawnHeight = imageHeight * scale
    private val offsetX = (canvas.width - drawnWidth) / 2f
    private val offsetY = (canvas.height - drawnHeight) / 2f

    fun point(normalizedX: Double, normalizedY: Double): Offset =
        Offset(offsetX + (normalizedX * drawnWidth).toFloat(), offsetY + (normalizedY * drawnHeight).toFloat())
}

/**
 * Draws the projected path as a ribbon on the ground, from the walker's feet through each
 * step's position projected into the frame.
 *
 * The ribbon is the prominent thing on screen, a fifth of the frame's width where it leaves
 * the bottom edge and narrowing with height, the way a strip of ground does, to a few pixels
 * at the top of the frame. Two channels, kept apart on purpose. Color is handed in, from
 * [pathColor], blue when there is nothing to worry about and red as surprise rises, since
 * that is what says an obstacle is about to be hit. Opacity follows the information at each
 * step, how far the scene moved the field's belief off the walker's own prior: a stretch
 * where nothing in view shaped the choice is drawn faint, a stretch where a wall or a gap
 * decided it is drawn solid, and the fade runs smoothly between the steps. The floor is high,
 * seven tenths, because the ribbon is meant to be prominent even on open ground. The
 * full-screen overlay turns this channel off with [fadeByInformation], since its window
 * opacity already sets how strong the ribbon is there. Whatever the
 * information says, the ribbon also fades with distance to nothing at its far end, so it is
 * always seen disappearing rather than stopping.
 */
internal fun DrawScope.drawProjectedPath(update: GuidanceUpdate, fit: FittedImage, color: Color, fadeByInformation: Boolean = true) {
    val path = update.guidance.projectedPath
    if (path.isEmpty()) return
    val frame = update.frame
    val projection = GroundPlaneProjection(frame.pose, frame.intrinsics, frame.width, frame.height)
    // The walker's feet sit below the bottom edge of the frame, so the ribbon starts where the
    // arrow starts, at the bottom center, and runs through every step that projects inside it.
    val centers = mutableListOf(fit.point(0.5, 1.0))
    val informationAtCenter = mutableListOf(path.first().informationBits)
    for (point in path) {
        val framePoint = projection.frameAt(GroundPoint(point.forwardMeters, point.rightMeters)) ?: continue
        centers += fit.point(framePoint.x, framePoint.y)
        informationAtCenter += point.informationBits
    }
    if (centers.size < 2) return

    val bottomY = fit.point(0.5, 1.0).y
    val topY = fit.point(0.5, 0.0).y
    val widthAtBottom = fit.drawnWidth * PATH_WIDTH_AT_BOTTOM_FRACTION
    fun halfWidthAt(y: Float): Float {
        val heightFraction = ((bottomY - y) / (bottomY - topY)).coerceIn(0f, 1f)
        return (widthAtBottom + (PATH_WIDTH_AT_TOP_PX - widthAtBottom) * heightFraction) / 2f
    }

    // The distance fade runs from full at the walker's feet to nothing at the last step.
    val fades = centers.indices.map { index -> 1f - index.toFloat() / (centers.size - 1) }.toMutableList()
    // The ribbon's square end lifts a corner off the bottom edge when the first segment turns.
    // Extending one width further back along the same direction hides that corner below the
    // frame, and the clip cuts the ribbon flat at the edge.
    val firstDirectionX = centers[1].x - centers[0].x
    val firstDirectionY = centers[1].y - centers[0].y
    val firstLength = kotlin.math.hypot(firstDirectionX, firstDirectionY).takeIf { it > 0f } ?: 1f
    centers.add(0, Offset(centers[0].x - firstDirectionX / firstLength * widthAtBottom, centers[0].y - firstDirectionY / firstLength * widthAtBottom))
    informationAtCenter.add(0, informationAtCenter[0])
    fades.add(0, 1f)

    // Each center gets an edge point either side, across the ribbon's direction there. The
    // direction at a point is the average of its two segments, so the joints do not kink.
    val leftEdge = ArrayList<Offset>(centers.size)
    val rightEdge = ArrayList<Offset>(centers.size)
    for (index in centers.indices) {
        val before = centers[max(index - 1, 0)]
        val after = centers[min(index + 1, centers.size - 1)]
        val directionX = after.x - before.x
        val directionY = after.y - before.y
        val length = kotlin.math.hypot(directionX, directionY).takeIf { it > 0f } ?: 1f
        val normalX = -directionY / length
        val normalY = directionX / length
        val halfWidth = halfWidthAt(centers[index].y)
        leftEdge += Offset(centers[index].x + normalX * halfWidth, centers[index].y + normalY * halfWidth)
        rightEdge += Offset(centers[index].x - normalX * halfWidth, centers[index].y - normalY * halfWidth)
    }
    val ribbon = Path().apply {
        moveTo(leftEdge.first().x, leftEdge.first().y)
        for (point in leftEdge.drop(1)) lineTo(point.x, point.y)
        for (point in rightEdge.asReversed()) lineTo(point.x, point.y)
        close()
    }

    // One color for the whole ribbon. Opacity follows each step's information as a vertical
    // gradient (the ribbon climbs the frame as steps go further). It also fades from full at
    // the feet to nothing at the far end, so it trails off rather than ending in an edge.
    val gradientTop = centers.minOf { it.y }
    val gradientBottom = centers.maxOf { it.y }
    val span = (gradientBottom - gradientTop).takeIf { it > 0f } ?: 1f
    val stops = centers.indices
        .map { index ->
            ((centers[index].y - gradientTop) / span) to color.copy(alpha = (if (fadeByInformation) pathOpacity(informationAtCenter[index]) else 1f) * fades[index])
        }
        .sortedBy { it.first }
        .toTypedArray()
    // The wide end's corners reach past the bottom of the frame, so the ribbon is clipped to
    // the image rather than spilling onto whatever the screen shows around it.
    val imageTopLeft = fit.point(0.0, 0.0)
    val imageBottomRight = fit.point(1.0, 1.0)
    clipRect(imageTopLeft.x, imageTopLeft.y, imageBottomRight.x, imageBottomRight.y) {
        drawPath(ribbon, brush = Brush.verticalGradient(colorStops = stops, startY = gradientTop, endY = gradientBottom))
    }
}

/** Opacity for a step's information: the floor at zero, solid from [PATH_INFORMATION_FOR_SOLID_BITS] up. */
private fun pathOpacity(informationBits: Double): Float =
    PATH_OPACITY_FLOOR + (1f - PATH_OPACITY_FLOOR) * min(1.0, informationBits / PATH_INFORMATION_FOR_SOLID_BITS).toFloat()

/** Blue at an alert fraction of 0, red at 1, mixed in between. The fraction comes from [GuidanceDisplayTuning.alertFraction]. */
internal fun pathColor(alertFraction: Float): Color {
    val fraction = alertFraction.coerceIn(0f, 1f)
    return Color(
        red = PATH_CALM_COLOR.red + (PATH_ALERT_COLOR.red - PATH_CALM_COLOR.red) * fraction,
        green = PATH_CALM_COLOR.green + (PATH_ALERT_COLOR.green - PATH_CALM_COLOR.green) * fraction,
        blue = PATH_CALM_COLOR.blue + (PATH_ALERT_COLOR.blue - PATH_CALM_COLOR.blue) * fraction,
    )
}

// TUNABLE. The path's two channels. One bit of information is a wall along the path or a
// barrier a meter ahead doubling the odds of the chosen direction, which is solid. Where red
// starts is GuidanceDisplayTuning.redSurpriseBits.
private val PATH_CALM_COLOR = Color(0xFF2979FF)
private val PATH_ALERT_COLOR = Color(0xFFFF1744)

// TUNABLE. A fifth of the frame where the ribbon leaves the bottom edge, narrowing linearly with
// height to twenty pixels at the top edge, so it reads as a strip of ground in perspective.
private const val PATH_WIDTH_AT_BOTTOM_FRACTION = 0.2f
private const val PATH_WIDTH_AT_TOP_PX = 20f

// TUNABLE. The ribbon is meant to be prominent, so even a step the scene had nothing to say about
// draws at seven tenths, and the information lifts it from there to solid.
private const val PATH_OPACITY_FLOOR = 0.7f
private const val PATH_INFORMATION_FOR_SOLID_BITS = 1.0
