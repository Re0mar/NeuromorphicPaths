package com.example.neuromorphicpaths.pipeline

import android.graphics.Bitmap
import com.example.neuromorphicpaths.geometry.Line
import com.example.neuromorphicpaths.geometry.SidewalkPosition
import com.example.neuromorphicpaths.steering.DirectionCommand
import com.example.neuromorphicpaths.steering.SteeringResult

data class PipelineFrame(
    val bitmap: Bitmap,
    val segmentationMask: Bitmap? = null,
    val timestampMs: Long,
    val rawWidth: Int,
    val rawHeight: Int,
    val leftEdge: Line? = null,
    val rightEdge: Line? = null,
    val sidewalkPosition: SidewalkPosition? = null,
    val steeringResult: SteeringResult? = null,
    val smoothedHeadingDeg: Float? = null,
    val stabilizedCommand: DirectionCommand? = null
)
