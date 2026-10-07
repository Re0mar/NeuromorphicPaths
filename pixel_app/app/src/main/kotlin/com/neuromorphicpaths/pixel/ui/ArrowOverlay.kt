package com.neuromorphicpaths.pixel.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.unit.dp
import com.neuromorphicpaths.pixel.wire.ReceivedPath
import java.util.Locale
import kotlin.math.cos
import kotlin.math.sin
import kotlinx.coroutines.delay

/** Where an arrow's tip lands on the screen, in pixels, y growing downward. */
data class ArrowTip(val x: Float, val y: Float)

/**
 * The arithmetic behind the overlay, kept out of the composable so it has JVM tests.
 *
 * The sign convention is the laptop's: a positive heading points right. The web page and the
 * OpenCV window draw the same way, so the three displays agree on which way to step.
 */
object ArrowGeometry {
    /** An arrow this old is drawn as stale. The laptop plans at frame rate, so a second of silence is a problem. */
    const val STALE_AFTER_MILLIS = 1_000L

    /** The tip of an arrow of [lengthPixels] from ([baseX], [baseY]), pointing [headingRadians] right of straight up. */
    fun headingToArrowTip(headingRadians: Double, lengthPixels: Float, baseX: Float, baseY: Float): ArrowTip =
        ArrowTip(
            x = baseX + lengthPixels * sin(headingRadians).toFloat(),
            y = baseY - lengthPixels * cos(headingRadians).toFloat(),
        )

    /** `+12 deg`, sign always shown, so a near-zero heading reads as a direction rather than a number. */
    fun headingText(headingRadians: Double): String = String.format(Locale.US, "%+.0f deg", Math.toDegrees(headingRadians))

    /** `0.3 s ago`, and `stale` once a second has passed without a path. */
    fun ageText(ageMillis: Long): String {
        val seconds = ageMillis.coerceAtLeast(0L) / 1000.0
        val base = String.format(Locale.US, "%.1f s ago", seconds)
        return if (ageMillis > STALE_AFTER_MILLIS) "$base, stale" else base
    }
}

/**
 * The arrow over the camera picture: where the newest path says to go, red on alarm, with the
 * heading in degrees and how old the path is.
 *
 * Age is measured from when the path arrived on this phone, with the same clock the connection
 * stamped it with, and the overlay ticks on its own so a frozen arrow visibly ages.
 */
@Composable
fun ArrowOverlay(received: ReceivedPath?, nowMillis: () -> Long, modifier: Modifier = Modifier) {
    var now by remember { mutableLongStateOf(nowMillis()) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(AGE_TICK_MILLIS)
            now = nowMillis()
        }
    }
    val message = received?.message
    val color = when {
        message == null -> Color.Gray
        message.alarm -> Color.Red
        else -> ARROW_GREEN
    }

    Box(modifier = modifier) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val baseX = size.width / 2f
            val baseY = size.height * BASE_HEIGHT_FRACTION
            val length = size.height * LENGTH_FRACTION
            val stroke = size.minDimension * STROKE_FRACTION
            val tip = ArrowGeometry.headingToArrowTip(message?.lookaheadHeadingRadians ?: 0.0, length, baseX, baseY)
            val tipOffset = Offset(tip.x, tip.y)
            drawLine(color, Offset(baseX, baseY), tipOffset, strokeWidth = stroke, cap = StrokeCap.Round)
            // The head: two strokes back from the tip, half a radian either side of the shaft.
            val shaftAngle = kotlin.math.atan2((tip.y - baseY).toDouble(), (tip.x - baseX).toDouble())
            val headLength = length * HEAD_FRACTION
            for (side in doubleArrayOf(-HEAD_HALF_ANGLE_RADIANS, HEAD_HALF_ANGLE_RADIANS)) {
                val end = Offset(
                    tip.x - headLength * cos(shaftAngle + side).toFloat(),
                    tip.y - headLength * sin(shaftAngle + side).toFloat(),
                )
                drawLine(color, tipOffset, end, strokeWidth = stroke, cap = StrokeCap.Round)
            }
        }
        Column(modifier = Modifier.align(Alignment.BottomCenter).padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            if (message == null) {
                Text("No path yet", color = Color.White, style = MaterialTheme.typography.titleLarge)
            } else {
                Text(ArrowGeometry.headingText(message.lookaheadHeadingRadians), color = Color.White, style = MaterialTheme.typography.titleLarge)
                if (message.alarm) Text("ALARM", color = Color.Red, style = MaterialTheme.typography.titleLarge)
                Text(ArrowGeometry.ageText(now - received.receivedAtMillis), color = Color.White)
            }
        }
    }
}

private const val AGE_TICK_MILLIS = 200L
private const val BASE_HEIGHT_FRACTION = 0.8f
private const val LENGTH_FRACTION = 0.35f
private const val STROKE_FRACTION = 0.03f
private const val HEAD_FRACTION = 0.25f
private const val HEAD_HALF_ANGLE_RADIANS = 0.5
private val ARROW_GREEN = Color(0xFF3FDF5F)
