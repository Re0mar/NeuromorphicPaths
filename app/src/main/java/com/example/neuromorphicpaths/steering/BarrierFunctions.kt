package com.example.neuromorphicpaths.steering

object BarrierFunctions {
    /**
     * Implements a repulsive gradient that increases as the user approaches an edge.
     * 
     * @param distanceToEdgeM Distance to the edge in meters (must be positive).
     * @param safetyMarginM Distance at which the force starts to be significant.
     * @param gain Scaling factor for the force.
     * @return Repulsive force magnitude.
     */
    fun repulsiveGradient(distanceToEdgeM: Float, safetyMarginM: Float, gain: Float): Float {
        if (distanceToEdgeM <= 0.05f) return gain * 20.0f
        if (distanceToEdgeM >= safetyMarginM) return 0f
        
        return gain * (1f / distanceToEdgeM - 1f / safetyMarginM)
    }
}
