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
import com.neuromorphicpaths.core.Obstacle
import java.util.Locale
import kotlin.math.cos
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
            Text(formatLine("surprise %.2f bits", guidance.overallSurpriseBits), color = Color.White)
            Text(formatLine("pitch %+.0f deg, fov %.0f deg", Math.toDegrees(current.update.frame.pose.pitchRadians), Math.toDegrees(current.update.frame.intrinsics.horizontalFovRadians)), color = Color.White)
            Text("${current.update.obstacles.size} obstacles", color = Color.White)
            for (obstacle in current.update.obstacles) {
                Text(describe(obstacle), color = Color.White)
            }
        }
    }
}

private fun describe(obstacle: Obstacle): String = formatLine(
    "%s %.1f m at %+.0f deg",
    obstacle.obstacleClass.name.lowercase().replace('_', ' '),
    obstacle.rangeMeters,
    Math.toDegrees(obstacle.bearingRadians),
)

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
private const val BOX_STROKE_PX = 4f
private const val ARROW_STROKE_PX = 8f
private const val ARROW_LENGTH_FRACTION = 0.3f
private const val ARROW_BARB_FRACTION = 0.2f
private const val ARROW_BARB_ANGLE_RADIANS = 0.5
