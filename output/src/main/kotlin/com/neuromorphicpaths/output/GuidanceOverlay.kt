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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import com.neuromorphicpaths.core.Guidance
import com.neuromorphicpaths.core.GuidanceUpdate
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.SceneClass
import com.neuromorphicpaths.core.SceneClassMap
import java.util.Locale
import kotlin.math.cos
import kotlin.math.sin

/**
 * Draws the newest frame with its detections, the desired heading as an arrow from the bottom
 * center, and the numbers in a corner. The screen-only feedback and the debugging view in one.
 */
@Composable
fun GuidanceOverlay(display: ScreenOverlayDisplay, modifier: Modifier = Modifier, tuning: GuidanceDisplayTuning = GuidanceDisplayTuning()) {
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
            drawProjectedPath(current.update, fit, pathColor(tuning.alertFraction(current.update.guidance)))
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
