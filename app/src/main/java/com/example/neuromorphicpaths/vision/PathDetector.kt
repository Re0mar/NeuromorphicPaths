package com.example.neuromorphicpaths.vision

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
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
        private const val MODEL_PATH = "best_int8.tflite"
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var inputImageWidth = 320
    private var inputImageHeight = 320

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
        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape() // e.g., {1, 320, 320, 1}
        val dataType = outputTensor.dataType()
        
        // Allocate buffer based on data type
        val outputBuffer = ByteBuffer.allocateDirect(outputTensor.numBytes())
        outputBuffer.order(ByteOrder.nativeOrder())

        // 3. Run inference
        interpreter.run(tensorImage.buffer, outputBuffer)

        // 4. Post-process output to find boundaries
        return processOutput(outputBuffer, outputShape, dataType)
    }

    private fun processOutput(buffer: ByteBuffer, shape: IntArray, dataType: DataType): PathResult {
        val height = shape[1]
        val width = shape[2]
        val numClasses = if (shape.size > 3) shape[3] else 1
        
        buffer.rewind()
        val mask = Array(height) { BooleanArray(width) }

        for (y in 0 until height) {
            for (x in 0 until width) {
                val probability = when (dataType) {
                    DataType.FLOAT32 -> buffer.float
                    DataType.UINT8, DataType.INT8 -> {
                        val value = buffer.get().toInt() and 0xFF
                        value.toFloat() / 255.0f
                    }
                    else -> 0f
                }
                // Skip other classes if multi-class, assuming class 0 or class 1 is path
                if (numClasses > 1) {
                    for (c in 1 until numClasses) {
                        when (dataType) {
                            DataType.FLOAT32 -> buffer.float
                            else -> buffer.get()
                        }
                    }
                }
                mask[y][x] = probability > 0.5f
            }
        }

        val boundaries = mutableListOf<List<Point>>()
        val leftPathPoints = mutableListOf<Point>()
        val rightPathPoints = mutableListOf<Point>()

        // Scan from bottom to top for better path tracking
        for (y in height - 1 downTo 0 step 8) {
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

        if (leftPathPoints.isNotEmpty()) {
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
