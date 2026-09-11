package com.example.neuromorphicpaths.vision.segmentation

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class SidewalkSegmenter(private val context: Context) {
    companion object {
        private const val TAG = "SidewalkSegmenter"
        private const val MODEL_PATH = "path_segmentation.tflite"
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var inputWidth = 256
    private var inputHeight = 256

    init {
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed.")
        }

        try {
            val options = Interpreter.Options()
            gpuDelegate = GpuDelegate()
            options.addDelegate(gpuDelegate)

            val modelBuffer = loadModelFile()
            if (modelBuffer != null) {
                interpreter = Interpreter(modelBuffer, options)
                val inputShape = interpreter!!.getInputTensor(0).shape() // {1, height, width, 3}
                inputHeight = inputShape[1]
                inputWidth = inputShape[2]
                Log.d(TAG, "Interpreter initialized successfully.")
            } else {
                Log.e(TAG, "Model file not found: $MODEL_PATH. Running in stub mode.")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing TFLite: ${e.message}")
        }
    }

    private fun loadModelFile(): ByteBuffer? {
        return try {
            val fileDescriptor = context.assets.openFd(MODEL_PATH)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        } catch (e: Exception) {
            null
        }
    }

    fun segment(bitmap: Bitmap): Mat {
        val mask = Mat(bitmap.height, bitmap.width, CvType.CV_8UC1)
        
        if (interpreter == null) {
            // Stub: Return a mock mask (e.g., a triangle representing a path)
            mask.setTo(Scalar(0.0))
            val points = listOf(
                org.opencv.core.Point(bitmap.width * 0.2, bitmap.height.toDouble()),
                org.opencv.core.Point(bitmap.width * 0.8, bitmap.height.toDouble()),
                org.opencv.core.Point(bitmap.width * 0.5, bitmap.height * 0.4)
            )
            val matOfPoint = org.opencv.core.MatOfPoint()
            matOfPoint.fromList(points)
            Imgproc.fillConvexPoly(mask, matOfPoint, Scalar(255.0))
            return mask
        }

        // 1. Preprocess
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

        // 2. Prepare Output
        val outputBuffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        // 3. Run Inference
        interpreter?.run(inputBuffer, outputBuffer)

        // 4. Post-process to OpenCV Mat
        outputBuffer.rewind()
        val maskData = ByteArray(inputWidth * inputHeight)
        for (i in 0 until inputWidth * inputHeight) {
            val prob = outputBuffer.float
            maskData[i] = if (prob > 0.5f) 255.toByte() else 0.toByte()
        }

        val smallMask = Mat(inputHeight, inputWidth, CvType.CV_8UC1)
        smallMask.put(0, 0, maskData)

        // Resize back to original size
        Imgproc.resize(smallMask, mask, mask.size())
        
        return mask
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(inputWidth * inputHeight)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (pixelValue in intValues) {
            byteBuffer.putFloat(((pixelValue shr 16 and 0xFF) / 255.0f))
            byteBuffer.putFloat(((pixelValue shr 8 and 0xFF) / 255.0f))
            byteBuffer.putFloat(((pixelValue and 0xFF) / 255.0f))
        }
        return byteBuffer
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
    }
}
