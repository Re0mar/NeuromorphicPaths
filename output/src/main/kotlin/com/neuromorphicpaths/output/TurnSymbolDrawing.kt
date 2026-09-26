package com.neuromorphicpaths.output

import android.graphics.Paint
import android.graphics.Typeface
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

/**
 * Draws one of the small overlay's symbols filling the draw scope, in [color]: a thick arrow
 * straight up, bent 45 or 90 degrees, a stop sign, or a U-turn.
 *
 * Shapes are laid out on a unit square and scaled, so the symbol reads the same at any size.
 * Everything is thick on purpose. It has to be read at a glance while walking.
 */
internal fun DrawScope.drawTurnSymbol(symbol: TurnSymbol, color: Color) {
    val side = min(size.width, size.height)
    val left = (size.width - side) / 2f
    val top = (size.height - side) / 2f
    fun at(unitX: Float, unitY: Float) = Offset(left + unitX * side, top + unitY * side)
    val shaftWidth = side * SHAFT_WIDTH_FRACTION

    when (symbol) {
        TurnSymbol.FORWARD -> drawBentArrow(side, ::at, bendDegrees = 0f, color = color, shaftWidth = shaftWidth)
        TurnSymbol.BEAR_RIGHT -> drawBentArrow(side, ::at, bendDegrees = 45f, color = color, shaftWidth = shaftWidth)
        TurnSymbol.BEAR_LEFT -> drawBentArrow(side, ::at, bendDegrees = -45f, color = color, shaftWidth = shaftWidth)
        TurnSymbol.HARD_RIGHT -> drawBentArrow(side, ::at, bendDegrees = 90f, color = color, shaftWidth = shaftWidth)
        TurnSymbol.HARD_LEFT -> drawBentArrow(side, ::at, bendDegrees = -90f, color = color, shaftWidth = shaftWidth)
        TurnSymbol.STOP -> drawStopSign(side, ::at, color)
        TurnSymbol.U_TURN -> drawUTurn(side, ::at, color, shaftWidth)
    }
}

// A shaft up from the bottom to a bend point, then on at the bend angle, then a head. The whole
// shape shifts against the bend so a hard turn still sits in the middle of the square.
private fun DrawScope.drawBentArrow(side: Float, at: (Float, Float) -> Offset, bendDegrees: Float, color: Color, shaftWidth: Float) {
    val angle = Math.toRadians(bendDegrees.toDouble())
    val directionX = sin(angle).toFloat()
    val directionY = -cos(angle).toFloat()
    val shift = -0.22f * directionX
    val bendY = if (bendDegrees == 0f) 0.34f else 0.56f
    val segmentLength = if (bendDegrees == 0f) 0f else 0.2f
    val start = at(0.5f + shift, 0.92f)
    val bend = at(0.5f + shift, bendY)
    val end = Offset(bend.x + directionX * segmentLength * side, bend.y + directionY * segmentLength * side)
    val shaft = Path().apply {
        moveTo(start.x, start.y)
        lineTo(bend.x, bend.y)
        lineTo(end.x, end.y)
    }
    drawPath(shaft, color, style = Stroke(width = shaftWidth, cap = StrokeCap.Butt, join = StrokeJoin.Round))
    drawArrowHead(end, directionX, directionY, side, color)
}

private fun DrawScope.drawArrowHead(base: Offset, directionX: Float, directionY: Float, side: Float, color: Color) {
    val length = side * HEAD_LENGTH_FRACTION
    val halfWidth = side * HEAD_WIDTH_FRACTION / 2f
    // The perpendicular to the direction, for the head's two back corners.
    val normalX = -directionY
    val normalY = directionX
    val head = Path().apply {
        moveTo(base.x + directionX * length, base.y + directionY * length)
        lineTo(base.x + normalX * halfWidth, base.y + normalY * halfWidth)
        lineTo(base.x - normalX * halfWidth, base.y - normalY * halfWidth)
        close()
    }
    drawPath(head, color)
}

private fun DrawScope.drawStopSign(side: Float, at: (Float, Float) -> Offset, color: Color) {
    val center = at(0.5f, 0.5f)
    val radius = side * 0.46f
    val octagon = Path().apply {
        for (corner in 0 until 8) {
            val angle = Math.PI / 8 + corner * Math.PI / 4
            val x = center.x + (radius * cos(angle)).toFloat()
            val y = center.y + (radius * sin(angle)).toFloat()
            if (corner == 0) moveTo(x, y) else lineTo(x, y)
        }
        close()
    }
    drawPath(octagon, color)
    val paint = Paint().apply {
        this.color = Color.White.toArgb()
        textSize = side * 0.26f
        typeface = Typeface.DEFAULT_BOLD
        textAlign = Paint.Align.CENTER
        isAntiAlias = true
    }
    // Baseline a third of the text size below center puts the capitals in the middle.
    drawContext.canvas.nativeCanvas.drawText("STOP", center.x, center.y + paint.textSize / 3f, paint)
}

// Up the right side, over the top in a half circle, down the left with the head pointing back.
private fun DrawScope.drawUTurn(side: Float, at: (Float, Float) -> Offset, color: Color, shaftWidth: Float) {
    val rightBottom = at(0.7f, 0.92f)
    val rightTop = at(0.7f, 0.42f)
    val leftTop = at(0.3f, 0.42f)
    val leftEnd = at(0.3f, 0.6f)
    val radius = (rightTop.x - leftTop.x) / 2f
    val arcCenter = Offset((rightTop.x + leftTop.x) / 2f, rightTop.y)
    val shape = Path().apply {
        moveTo(rightBottom.x, rightBottom.y)
        lineTo(rightTop.x, rightTop.y)
        arcTo(Rect(arcCenter, radius), startAngleDegrees = 0f, sweepAngleDegrees = -180f, forceMoveTo = false)
        lineTo(leftEnd.x, leftEnd.y)
    }
    drawPath(shape, color, style = Stroke(width = shaftWidth, cap = StrokeCap.Butt, join = StrokeJoin.Round))
    drawArrowHead(leftEnd, 0f, 1f, side, color)
}

// TUNABLE. Proportions of the symbols against their square: a thick shaft and a wide head.
private const val SHAFT_WIDTH_FRACTION = 0.16f
private const val HEAD_LENGTH_FRACTION = 0.22f
private const val HEAD_WIDTH_FRACTION = 0.4f
