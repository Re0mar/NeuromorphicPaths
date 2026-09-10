package com.example.neuromorphicpaths.vision

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * PathDetector handles the loading and execution of a TensorFlow Lite model
 * for detecting walkways and their boundaries in an image.
 */
class PathDetector(context: Context) {
    companion object {
        private const val TAG = "PathDetector"
        private const val MODEL_PATH = "path_segmentation.tflite"
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var inputImageWidth = 0
    private var inputImageHeight = 0

    init {
        try {
            val options = Interpreter.Options()
            // Use GPU delegate for faster inference if available
            gpuDelegate = GpuDelegate()
            options.addDelegate(gpuDelegate)

            val modelBuffer = FileUtil.loadMappedFile(context, MODEL_PATH)
            interpreter = Interpreter(modelBuffer, options)

            // Get input shape from the model
            val inputShape = interpreter!!.getInputTensor(0).shape() // {1, height, width, 3}
            inputImageHeight = inputShape[1]
            inputImageWidth = inputShape[2]
            
            Log.d(TAG, "PathDetector initialized with model $MODEL_PATH ($inputImageWidth x $inputImageHeight)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize TFLite interpreter. Ensure $MODEL_PATH exists in assets.", e)
        }
    }

    fun detectPath(bitmap: Bitmap): PathResult? {
        val interpreter = interpreter ?: return null

        // 1. Pre-process the image
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputImageHeight, inputImageWidth, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        var tensorImage = TensorImage(interpreter.getInputTensor(0).dataType())
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)

        // 2. Prepare output buffer
        // Assuming the model outputs a segmentation mask of shape {1, height, width, num_classes}
        val outputShape = interpreter.getOutputTensor(0).shape()
        val outputBuffer = ByteBuffer.allocateDirect(outputShape.fold(1) { acc, i -> acc * i } * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        // 3. Run inference
        interpreter.run(tensorImage.buffer, outputBuffer)

        // 4. Post-process output to find boundaries
        // This is highly dependent on the model output format.
        // For a segmentation mask, we would find the contours of the 'walkway' class.
        return processMask(outputBuffer, outputShape)
    }

    private fun processMask(buffer: ByteBuffer, shape: IntArray): PathResult {
        val height = shape[1]
        val width = shape[2]
        val boundaries = mutableListOf<List<Point>>()

        // 1. Create a 2D array for the binary mask
        val mask = Array(height) { BooleanArray(width) }
        buffer.rewind()
        for (y in 0 until height) {
            for (x in 0 until width) {
                // If model has multiple output channels, we take the one corresponding to 'walkway'
                // Here we assume a single channel sigmoid output
                val probability = buffer.float
                mask[y][x] = probability > 0.5f
            }
        }

        // 2. Simple Boundary Extraction (Scanning for edges)
        val leftPathPoints = mutableListOf<Point>()
        val rightPathPoints = mutableListOf<Point>()

        for (y in 0 until height step 5) { // Sample every 5 rows
            var leftEdge = -1
            var rightEdge = -1
            for (x in 0 until width) {
                if (mask[y][x]) {
                    if (leftEdge == -1) leftEdge = x
                    rightEdge = x
                }
            }
            if (leftEdge != -1) {
                leftPathPoints.add(Point(leftEdge.toFloat() / width, y.toFloat() / height))
                rightPathPoints.add(Point(rightEdge.toFloat() / width, y.toFloat() / height))
            }
        }

        // Combine into a single list of points forming a boundary
        if (leftPathPoints.isNotEmpty()) {
            // Concatenate left edge (top down) and right edge (bottom up) to form a polygon
            val fullPath = mutableListOf<Point>()
            fullPath.addAll(leftPathPoints)
            fullPath.addAll(rightPathPoints.reversed())
            boundaries.add(fullPath)
        }

        return PathResult(boundaries)
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
    }
}

data class PathResult(
    val boundaries: List<List<Point>>
)

data class Point(val x: Float, val y: Float)
