package com.example.sidewalkvision

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope

private val STRAIGHT_EDGE_COLOR = Color(0xFF4CAF50)
private val REJECTED_EDGE_COLOR = Color(0xFFFF9800)
private val HORIZON_COLOR = Color.White.copy(alpha = 0.7f)

/**
 * Where the pose's frame sits on screen. Frame pixels are those the pose was estimated in, the
 * detection's mask bitmap. The image is drawn scaled and offset to fit the view.
 */
data class FrameOnScreen(
    val frameWidthPx: Int,
    val frameHeightPx: Int,
    val left: Float,
    val top: Float,
    val scaledWidth: Float,
    val scaledHeight: Float,
) {
    fun toScreen(xPx: Double, yPx: Double): Offset = Offset(
        left + (xPx / frameWidthPx * scaledWidth).toFloat(),
        top + (yPx / frameHeightPx * scaledHeight).toFloat(),
    )
}

/**
 * Draws the pose's working for debugging: each fitted edge line, green if straight enough to
 * trust and orange if not, and the vanishing point with the horizon through it. The lines run
 * from the vanishing point down to the bottom of the frame when there is one, and over the rows
 * they were fitted to otherwise.
 */
fun DrawScope.drawPoseWorking(pose: PoseEstimate, frame: FrameOnScreen) {
    val vanishingPoint = pose.vanishingPointPx
    for (fit in listOfNotNull(pose.left, pose.right)) {
        val fromY = vanishingPoint?.second?.coerceAtLeast(0.0) ?: fit.topPx
        val toY = if (vanishingPoint != null) frame.frameHeightPx.toDouble() else fit.bottomPx
        drawLine(
            color = if (fit.straight) STRAIGHT_EDGE_COLOR else REJECTED_EDGE_COLOR,
            start = frame.toScreen(fit.xAt(fromY), fromY),
            end = frame.toScreen(fit.xAt(toY), toY),
            strokeWidth = 6f,
        )
    }
    if (vanishingPoint != null && pose.status == PoseStatus.OK) {
        val (vanishingX, vanishingY) = vanishingPoint
        // With no roll the horizon is level, so it's the row through the vanishing point.
        if (vanishingY in 0.0..frame.frameHeightPx.toDouble()) {
            drawLine(
                color = HORIZON_COLOR,
                start = frame.toScreen(0.0, vanishingY),
                end = frame.toScreen(frame.frameWidthPx.toDouble(), vanishingY),
                strokeWidth = 3f,
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(20f, 12f)),
            )
        }
        drawCircle(color = Color.White, radius = 12f, center = frame.toScreen(vanishingX, vanishingY))
    }
}
