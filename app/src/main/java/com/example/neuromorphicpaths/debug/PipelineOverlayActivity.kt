package com.example.neuromorphicpaths.debug

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Paint
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.neuromorphicpaths.pipeline.PipelineFrame
import com.example.neuromorphicpaths.ui.theme.NeuromorphicPathsTheme
import java.util.Locale

class PipelineOverlayActivity : ComponentActivity() {
    private val viewModel: PipelineOverlayViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        viewModel.startPipeline()
        
        setContent {
            NeuromorphicPathsTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    DebugView(viewModel)
                }
            }
        }
    }
}

@Composable
fun DebugView(viewModel: PipelineOverlayViewModel) {
    val frame by viewModel.latestFrame.collectAsState()
    val isLogging by viewModel.isLogging.collectAsState()

    Box(modifier = Modifier.fillMaxSize()) {
        frame?.let { f ->
            // Camera Preview
            Image(
                bitmap = f.bitmap.asImageBitmap(),
                contentDescription = "Camera Preview",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit
            )

            // Overlays (Mask and Lines)
            PipelineOverlay(f)
            
            // HUD
            TelemetryHud(f, isLogging, onToggleLogging = { viewModel.toggleLogging() })
        } ?: Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator()
            Text("Waiting for frames...", modifier = Modifier.padding(top = 64.dp))
        }
    }
}

@Composable
fun PipelineOverlay(frame: PipelineFrame) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val scaleX = size.width / frame.rawWidth
        val scaleY = size.height / frame.rawHeight

        // Draw Mask
        frame.segmentationMask?.let { mask ->
            drawIntoCanvas { canvas ->
                val paint = Paint().apply {
                    alpha = 0.4f
                }
                canvas.drawImageRect(
                    image = mask.asImageBitmap(),
                    dstOffset = IntOffset(0, 0),
                    dstSize = IntSize(size.width.toInt(), size.height.toInt()),
                    paint = paint
                )
            }
        }

        // Draw Left Edge
        frame.leftEdge?.let { line ->
            drawLine(
                color = Color.Red,
                start = Offset(line.x1 * scaleX, line.y1 * scaleY),
                end = Offset(line.x2 * scaleX, line.y2 * scaleY),
                strokeWidth = 5f
            )
        }

        // Draw Right Edge
        frame.rightEdge?.let { line ->
            drawLine(
                color = Color.Blue,
                start = Offset(line.x1 * scaleX, line.y1 * scaleY),
                end = Offset(line.x2 * scaleX, line.y2 * scaleY),
                strokeWidth = 5f
            )
        }
    }
}

@Composable
fun TelemetryHud(
    frame: PipelineFrame,
    isLogging: Boolean,
    onToggleLogging: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .padding(top = 32.dp),
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        // Top Stats
        Column(
            modifier = Modifier
                .background(Color.Black.copy(alpha = 0.6f))
                .padding(8.dp)
        ) {
            val pos = frame.sidewalkPosition
            Text("Left Offset: ${String.format(Locale.US, "%.2f m", pos?.leftEdgeOffsetM ?: 0.0)}", color = Color.White, fontSize = 14.sp, fontFamily = FontFamily.Monospace)
            Text("Right Offset: ${String.format(Locale.US, "%.2f m", pos?.rightEdgeOffsetM ?: 0.0)}", color = Color.White, fontSize = 14.sp, fontFamily = FontFamily.Monospace)
            Text("Heading Dev: ${String.format(Locale.US, "%.1f deg", frame.steeringResult?.recommendedHeadingDeg ?: 0.0)}", color = Color.White, fontSize = 14.sp, fontFamily = FontFamily.Monospace)
            Text("Command: ${frame.stabilizedCommand?.name ?: "NONE"}", color = Color.Yellow, fontSize = 16.sp, fontFamily = FontFamily.Monospace)
        }

        // Bottom Controls
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End
        ) {
            Button(
                onClick = onToggleLogging,
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isLogging) Color.Red else MaterialTheme.colorScheme.primary
                )
            ) {
                Text(if (isLogging) "STOP LOGGING" else "START LOGGING")
            }
        }
    }
}
