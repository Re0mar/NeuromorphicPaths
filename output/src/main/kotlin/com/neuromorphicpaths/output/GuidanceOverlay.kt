package com.neuromorphicpaths.output

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import com.neuromorphicpaths.core.GroundPlaneProjection
import com.neuromorphicpaths.core.GroundPoint
import com.neuromorphicpaths.core.Guidance
import com.neuromorphicpaths.core.GuidanceUpdate
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.SceneClass
import com.neuromorphicpaths.core.SceneClassMap
import java.util.Locale
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Draws the newest frame with its detections, the desired heading as an arrow from the bottom
 * center, and the numbers in a corner. The screen-only feedback and the debugging view in one.
 */
@Composable
fun GuidanceOverlay(display: ScreenOverlayDisplay, modifier: Modifier = Modifier) {
    val state by display.latest.collectAsState()
    val current = state
    if (current == null) {
        Box(modifier, contentAlignment = Alignment.Center) { Text("Waiting for frames") }
        return
    }
    Box(modifier) {
        Image(
            bitmap = current.image,
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Fit,
        )
        Canvas(Modifier.fillMaxSize()) {
            val fit = FittedImage(current.image.width, current.image.height, size)
            current.update.sceneMap?.let { drawSceneMap(it, fit) }
            drawProjectedPath(current.update, fit)
            for (detection in current.update.detections) {
                val topLeft = fit.point(detection.box.left, detection.box.top)
                val bottomRight = fit.point(detection.box.right, detection.box.bottom)
                drawRect(
                    color = BOX_COLOR,
                    topLeft = topLeft,
                    size = Size(bottomRight.x - topLeft.x, bottomRight.y - topLeft.y),
                    style = Stroke(width = BOX_STROKE_PX),
                )
            }
            drawHeadingArrow(current.update.guidance.desiredHeadingRadians, fit)
        }
        Column(
            Modifier
                .align(Alignment.TopStart)
                .padding(8.dp)
                .background(Color.Black.copy(alpha = 0.55f))
                .padding(8.dp),
        ) {
            val guidance = current.update.guidance
            Text(formatLine("heading %+.0f deg", Math.toDegrees(guidance.desiredHeadingRadians)), color = Color.White)
            Text(formatLine("surprise %.2f bits, %s", guidance.overallSurpriseBits, describeEntropy(guidance)), color = Color.White)
            Text(formatLine("pitch %+.0f deg, fov %.0f deg", Math.toDegrees(current.update.frame.pose.pitchRadians), Math.toDegrees(current.update.frame.intrinsics.horizontalFovRadians)), color = Color.White)
            Text(describeWalker(current.update), color = Color.White)
            Text("${current.update.obstacles.size} obstacles", color = Color.White)
            for (obstacle in current.update.obstacles) {
                Text(describe(obstacle), color = Color.White)
            }
        }
    }
}

private fun describe(obstacle: Obstacle): String {
    val closing = obstacle.closingSpeedMetersPerSecond?.let { formatLine(", closing %.1f m/s", it) } ?: ""
    val track = obstacle.detection.trackId?.let { "#$it " } ?: ""
    return formatLine(
        "%s%s %.1f m at %+.0f deg%s",
        track,
        obstacle.obstacleClass.name.lowercase().replace('_', ' '),
        obstacle.rangeMeters,
        Math.toDegrees(obstacle.bearingRadians),
        closing,
    )
}

/** How spread the field's belief is and how far the scene moved it, with a dash for a field that holds neither. */
private fun describeEntropy(guidance: Guidance): String {
    val entropy = guidance.headingEntropyBits?.let { formatLine("entropy %.2f bits", it) } ?: "entropy - bits"
    val information = guidance.headingInformationBits?.let { formatLine("information %.2f bits", it) } ?: "information - bits"
    return "$entropy, $information"
}

/** Speed, wobble and the tolerance in force, with a dash for whatever nothing has measured yet. */
private fun describeWalker(update: GuidanceUpdate): String {
    val speed = update.walker.speedMetersPerSecond?.let { formatLine("%.1f m/s", it) } ?: "- m/s"
    val wobble = update.guidance.walkerWobbleRadians?.let { formatLine("%.0f deg", Math.toDegrees(it)) } ?: "- deg"
    val tolerance = update.guidance.turnToleranceRadians?.let { formatLine("%.0f deg", Math.toDegrees(it)) } ?: "- deg"
    return "speed $speed, wobble $wobble, tolerance $tolerance"
}

private fun formatLine(pattern: String, vararg values: Any): String = String.format(Locale.US, pattern, *values)

/** Maps normalized image coordinates onto the rectangle [ContentScale.Fit] gives the image. */
private class FittedImage(imageWidth: Int, imageHeight: Int, canvas: Size) {
    private val scale = min(canvas.width / imageWidth, canvas.height / imageHeight)
    val drawnWidth = imageWidth * scale
    val drawnHeight = imageHeight * scale
    private val offsetX = (canvas.width - drawnWidth) / 2f
    private val offsetY = (canvas.height - drawnHeight) / 2f

    fun point(normalizedX: Double, normalizedY: Double): Offset =
        Offset(offsetX + (normalizedX * drawnWidth).toFloat(), offsetY + (normalizedY * drawnHeight).toFloat())
}

/**
 * Tints each cell of the scene map that the field acts on, so a person can see what the
 * segmenter called ground and what it called a wall. Sky and everything else stay clear.
 */
private fun DrawScope.drawSceneMap(map: SceneClassMap, fit: FittedImage) {
    val cellWidth = 1.0 / map.width
    val cellHeight = 1.0 / map.height
    for (row in 0 until map.height) {
        for (column in 0 until map.width) {
            val color = SCENE_TINTS[map.classAt(column, row)] ?: continue
            val topLeft = fit.point(column * cellWidth, row * cellHeight)
            val bottomRight = fit.point((column + 1) * cellWidth, (row + 1) * cellHeight)
            drawRect(color, topLeft, Size(bottomRight.x - topLeft.x, bottomRight.y - topLeft.y))
        }
    }
}

/**
 * Draws the projected path as a ribbon on the ground, from the walker's feet through each
 * step's position projected into the frame.
 *
 * The ribbon is the prominent thing on screen, a fifth of the frame's width where it leaves
 * the bottom edge and narrowing with height, the way a strip of ground does, to a few pixels
 * at the top of the frame. Two channels, kept apart on purpose. Color follows the frame's
 * surprise, blue when the walker's line is near the best one and red as the gap opens, since
 * that is what says an obstacle is about to be hit. Opacity follows the information at each
 * step, how far the scene moved the field's belief off the walker's own prior: a stretch
 * where nothing in view shaped the choice is drawn faint, a stretch where a wall or a gap
 * decided it is drawn solid, and the fade runs smoothly between the steps. The floor is high,
 * seven tenths, because the ribbon is meant to be prominent even on open ground. Whatever the
 * information says, the ribbon also fades with distance to nothing at its far end, so it is
 * always seen disappearing rather than stopping.
 */
private fun DrawScope.drawProjectedPath(update: GuidanceUpdate, fit: FittedImage) {
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
    // The ribbon's end is square to its first segment, so when that segment turns, one corner
    // of the end lifts off the bottom edge and leaves a wedge of frame showing. Starting the
    // ribbon one bottom-width further back along the same direction, below the frame, puts
    // that corner out of sight and the clip cuts the ribbon flat at the edge.
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

    // One color for the whole ribbon, and an opacity that follows each step's information as a
    // vertical gradient, since the ribbon climbs the frame as the steps go further out. On top
    // of that the ribbon always fades with distance, from full at the walker's feet to nothing
    // at its far end, so it never stops with an edge: the eye reads it as disappearing.
    val color = pathColor(update.guidance.overallSurpriseBits)
    val gradientTop = centers.minOf { it.y }
    val gradientBottom = centers.maxOf { it.y }
    val span = (gradientBottom - gradientTop).takeIf { it > 0f } ?: 1f
    val stops = centers.indices
        .map { index ->
            ((centers[index].y - gradientTop) / span) to color.copy(alpha = pathOpacity(informationAtCenter[index]) * fades[index])
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

/** Blue at no surprise, red from [PATH_SURPRISE_FOR_RED_BITS] up, mixed in between. */
private fun pathColor(surpriseBits: Double): Color {
    val fraction = min(1.0, surpriseBits / PATH_SURPRISE_FOR_RED_BITS).toFloat()
    return Color(
        red = PATH_CALM_COLOR.red + (PATH_ALERT_COLOR.red - PATH_CALM_COLOR.red) * fraction,
        green = PATH_CALM_COLOR.green + (PATH_ALERT_COLOR.green - PATH_CALM_COLOR.green) * fraction,
        blue = PATH_CALM_COLOR.blue + (PATH_ALERT_COLOR.blue - PATH_CALM_COLOR.blue) * fraction,
    )
}

private fun DrawScope.drawHeadingArrow(headingRadians: Double, fit: FittedImage) {
    val start = fit.point(0.5, 1.0)
    val length = fit.drawnHeight * ARROW_LENGTH_FRACTION
    val end = Offset(
        start.x + (length * sin(headingRadians)).toFloat(),
        start.y - (length * cos(headingRadians)).toFloat(),
    )
    drawLine(ARROW_COLOR, start, end, strokeWidth = ARROW_STROKE_PX)
    // Two short barbs angled back from the tip.
    val barbLength = length * ARROW_BARB_FRACTION
    for (side in listOf(-1.0, 1.0)) {
        val barbAngle = headingRadians + Math.PI + side * ARROW_BARB_ANGLE_RADIANS
        val barbEnd = Offset(
            end.x + (barbLength * sin(barbAngle)).toFloat(),
            end.y - (barbLength * cos(barbAngle)).toFloat(),
        )
        drawLine(ARROW_COLOR, end, barbEnd, strokeWidth = ARROW_STROKE_PX)
    }
}

private val BOX_COLOR = Color(0xFFFFC107)
private val ARROW_COLOR = Color(0xFF00E5FF)

// The path's two channels. One bit of information is a wall along the path or a barrier a
// meter ahead doubling the odds of the chosen direction, which is solid. Three bits of
// surprise is the cone at its worst on the outdoor walk, which is red.
private val PATH_CALM_COLOR = Color(0xFF2979FF)
private val PATH_ALERT_COLOR = Color(0xFFFF1744)

// A fifth of the frame where the ribbon leaves the bottom edge, narrowing linearly with
// height to twenty pixels at the top edge, so it reads as a strip of ground in perspective.
private const val PATH_WIDTH_AT_BOTTOM_FRACTION = 0.2f
private const val PATH_WIDTH_AT_TOP_PX = 20f

// The ribbon is meant to be prominent, so even a step the scene had nothing to say about
// draws at seven tenths, and the information lifts it from there to solid.
private const val PATH_OPACITY_FLOOR = 0.7f
private const val PATH_INFORMATION_FOR_SOLID_BITS = 1.0
private const val PATH_SURPRISE_FOR_RED_BITS = 3.0

// Translucent, so the frame stays readable under the tint. Ground is cool, structures are warm.
private val SCENE_TINTS: Map<SceneClass, Color> = mapOf(
    SceneClass.PAVEMENT to Color(0x4000E5FF),
    SceneClass.GRASS to Color(0x5000C853),
    SceneClass.DIRT to Color(0x50795548),
    SceneClass.ROAD to Color(0x506A1B9A),
    SceneClass.BUILDING to Color(0x60E53935),
    SceneClass.WALL to Color(0x60FF6D00),
    SceneClass.STAIRS to Color(0x60FF4081),
)
private const val BOX_STROKE_PX = 4f
private const val ARROW_STROKE_PX = 8f
private const val ARROW_LENGTH_FRACTION = 0.3f
private const val ARROW_BARB_FRACTION = 0.2f
private const val ARROW_BARB_ANGLE_RADIANS = 0.5
