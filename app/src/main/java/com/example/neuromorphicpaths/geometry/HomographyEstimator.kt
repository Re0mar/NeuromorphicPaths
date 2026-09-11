package com.example.neuromorphicpaths.geometry

import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc

class HomographyEstimator {

    /**
     * Computes the homography matrix H that transforms image coordinates (u, v)
     * to ground-plane coordinates (x, y) in meters.
     */
    fun computeHomography(imageWidth: Int, imageHeight: Int): Mat {
        // Default calibration points for glasses at ~1.6m height
        // Mapping a trapezoid in the image to a rectangle on the ground.
        
        // Image points (trapezoid on sidewalk)
        val srcPoints = arrayOf(
            Point(imageWidth * 0.15, imageHeight.toDouble()),    // Bottom-left
            Point(imageWidth * 0.85, imageHeight.toDouble()),    // Bottom-right
            Point(imageWidth * 0.65, imageHeight * 0.5),         // Top-right
            Point(imageWidth * 0.35, imageHeight * 0.5)          // Top-left
        )

        // Ground points in meters relative to the user (x=0 is centerline, y=0 is feet)
        // x-axis: negative is left, positive is right.
        // y-axis: distance forward from feet.
        val dstPoints = arrayOf(
            Point(-1.0, 1.0), // Bottom-left (1m ahead, 1m left)
            Point(1.0, 1.0),  // Bottom-right (1m ahead, 1m right)
            Point(1.0, 8.0),  // Top-right (8m ahead, 1m right)
            Point(-1.0, 8.0)  // Top-left (8m ahead, 1m left)
        )

        val srcMat = MatOfPoint2f(*srcPoints)
        val dstMat = MatOfPoint2f(*dstPoints)

        return Imgproc.getPerspectiveTransform(srcMat, dstMat)
    }
}
