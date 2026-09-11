package com.example.neuromorphicpaths.vision

data class SidewalkPosition(
    val leftEdgeOffsetM: Float,
    val rightEdgeOffsetM: Float,
    val headingDeviationDeg: Float,
    val confidence: Float
)

enum class DirectionCommand { HARD_LEFT, LEFT, STRAIGHT, RIGHT, HARD_RIGHT, UNKNOWN }
