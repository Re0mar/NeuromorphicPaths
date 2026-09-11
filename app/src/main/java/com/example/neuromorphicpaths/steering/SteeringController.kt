package com.example.neuromorphicpaths.steering

import com.example.neuromorphicpaths.geometry.SidewalkPosition
import kotlin.math.abs

data class SteeringResult(
    val lateralPushM: Float,
    val recommendedHeadingDeg: Float,
    val command: DirectionCommand
)

class SteeringController {
    private val safetyMargin = 0.8f // meters
    private val repulsionGain = 0.4f
    private val attractionGain = 0.2f // Attraction to centerline

    fun computeSteering(position: SidewalkPosition): SteeringResult {
        val distLeft = abs(position.leftEdgeOffsetM)
        val distRight = abs(position.rightEdgeOffsetM)
        
        val leftRepulsion = BarrierFunctions.repulsiveGradient(distLeft, safetyMargin, repulsionGain)
        val rightRepulsion = BarrierFunctions.repulsiveGradient(distRight, safetyMargin, repulsionGain)
        
        // Net lateral push: positive means push right (away from left edge)
        var lateralPush = leftRepulsion - rightRepulsion
        
        // Attraction to centerline: position is relative to user. Center is at (left + right)/2
        val centerlineOffset = (position.leftEdgeOffsetM + position.rightEdgeOffsetM) / 2.0f
        lateralPush += centerlineOffset * attractionGain
        
        // Recommended heading combines heading deviation and lateral push
        val recommendedHeading = position.headingDeviationDeg + (lateralPush * 15f)

        return SteeringResult(
            lateralPushM = lateralPush,
            recommendedHeadingDeg = recommendedHeading,
            command = mapToCommand(recommendedHeading)
        )
    }

    fun mapToCommand(heading: Float): DirectionCommand {
        return when {
            heading > 15f -> DirectionCommand.HARD_RIGHT
            heading > 5f -> DirectionCommand.SLIGHT_RIGHT
            heading < -15f -> DirectionCommand.HARD_LEFT
            heading < -5f -> DirectionCommand.SLIGHT_LEFT
            else -> DirectionCommand.STRAIGHT
        }
    }
}
