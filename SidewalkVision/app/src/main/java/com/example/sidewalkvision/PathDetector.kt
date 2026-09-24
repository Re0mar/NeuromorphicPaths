package com.example.sidewalkvision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

enum class SidewalkEdgeSide {
    LEFT, RIGHT, NEAR, FAR
}

data class Point2D(
    val x: Float,
    val y: Float
)

data class SidewalkEdgePoint(
    val position: Point2D,
    val side: SidewalkEdgeSide,
    val estimatedDistanceMeters: Float?
)

data class DetectionResult(
    val score: Float,
    val box: FloatArray?, // [l, t, r, b] normalized [0, 1]
    val maskBitmap: Bitmap?,
    val edgePoints: List<SidewalkEdgePoint> = emptyList()
)

class PathDetector(context: Context) {
    private var interpreter: Interpreter? = null
    private var inputShape: IntArray = intArrayOf()
    private var inputDataType: DataType = DataType.FLOAT32
    private var inputQuantScale: Float = 1f
    private var inputQuantZeroPoint: Int = 0
    private var isNchw: Boolean = false
    private var inWidth: Int = 320
    private var inHeight: Int = 320

    private val outputDetails = mutableListOf<Tensor>()
    private val outputQuantScales = mutableMapOf<Int, Float>()
    private val outputQuantZeroPoints = mutableMapOf<Int, Int>()

    init {
        try {
            val modelBuffer = loadModelFile(context, "best_int8.tflite")
            val options = Interpreter.Options().apply {
                numThreads = 4
            }
            val interp = Interpreter(modelBuffer, options)
            interp.allocateTensors()
            interpreter = interp

            val inpDetails = interp.getInputTensor(0)
            inputShape = inpDetails.shape()
            inputDataType = inpDetails.dataType()
            val quantParams = inpDetails.quantizationParams()
            inputQuantScale = quantParams.scale
            inputQuantZeroPoint = quantParams.zeroPoint

            isNchw = inputShape.size == 4 && (inputShape[1] == 1 || inputShape[1] == 3)
            if (isNchw) {
                inHeight = inputShape[2]
                inWidth = inputShape[3]
            } else if (inputShape.size == 4) {
                inHeight = inputShape[1]
                inWidth = inputShape[2]
            }

            for (i in 0 until interp.outputTensorCount) {
                val outTensor = interp.getOutputTensor(i)
                outputDetails.add(outTensor)
                val qp = outTensor.quantizationParams()
                outputQuantScales[i] = qp.scale
                outputQuantZeroPoints[i] = qp.zeroPoint
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun loadModelFile(context: Context, assetName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(assetName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun detect(bitmap: Bitmap): DetectionResult {
        val interp = interpreter ?: return DetectionResult(0f, null, null)

        val origWidth = bitmap.width
        val origHeight = bitmap.height

        // Letterbox
        val scale = min(inWidth.toFloat() / origWidth, inHeight.toFloat() / origHeight)
        val newWidth = max(1, round(origWidth * scale).toInt())
        val newHeight = max(1, round(origHeight * scale).toInt())
        val padX = (inWidth - newWidth) / 2f
        val padY = (inHeight - newHeight) / 2f

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
        val canvasBitmap = Bitmap.createBitmap(inWidth, inHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(canvasBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.drawBitmap(scaledBitmap, padX, padY, null)

        val intValues = IntArray(inWidth * inHeight)
        canvasBitmap.getPixels(intValues, 0, inWidth, 0, 0, inWidth, inHeight)

        val elementSize = when (inputDataType) {
            DataType.FLOAT32 -> 4
            DataType.INT32 -> 4
            else -> 1 // INT8, UINT8
        }
        val inputBuffer = ByteBuffer.allocateDirect(1 * inHeight * inWidth * 3 * elementSize)
        inputBuffer.order(ByteOrder.nativeOrder())

        if (isNchw) {
            val chwPixels = FloatArray(3 * inWidth * inHeight)
            var idx = 0
            for (c in 0 until 3) {
                for (h in 0 until inHeight) {
                    for (w in 0 until inWidth) {
                        val p = intValues[h * inWidth + w]
                        val v = when (c) {
                            0 -> ((p shr 16) and 0xFF).toFloat()
                            1 -> ((p shr 8) and 0xFF).toFloat()
                            else -> (p and 0xFF).toFloat()
                        }
                        chwPixels[idx++] = v
                    }
                }
            }
            if (inputDataType == DataType.FLOAT32) {
                for (v in chwPixels) inputBuffer.putFloat(v / 255f)
            } else {
                val expects0To255 = inputQuantScale < 0.01f
                for (v in chwPixels) {
                    val realVal = if (expects0To255) v else (v / 255f)
                    val q = round(realVal / inputQuantScale + inputQuantZeroPoint).toInt()
                    if (inputDataType == DataType.INT8) {
                        inputBuffer.put(q.coerceIn(-128, 127).toByte())
                    } else if (inputDataType == DataType.UINT8) {
                        inputBuffer.put(q.coerceIn(0, 255).toByte())
                    } else {
                        inputBuffer.put(q.coerceIn(-128, 127).toByte())
                    }
                }
            }
        } else {
            val hwcPixels = FloatArray(inWidth * inHeight * 3)
            var idx = 0
            for (i in intValues.indices) {
                val p = intValues[i]
                hwcPixels[idx++] = ((p shr 16) and 0xFF).toFloat()
                hwcPixels[idx++] = ((p shr 8) and 0xFF).toFloat()
                hwcPixels[idx++] = (p and 0xFF).toFloat()
            }
            if (inputDataType == DataType.FLOAT32) {
                for (v in hwcPixels) inputBuffer.putFloat(v / 255f)
            } else {
                val expects0To255 = inputQuantScale < 0.01f
                for (v in hwcPixels) {
                    val realVal = if (expects0To255) v else (v / 255f)
                    val q = round(realVal / inputQuantScale + inputQuantZeroPoint).toInt()
                    if (inputDataType == DataType.INT8) {
                        inputBuffer.put(q.coerceIn(-128, 127).toByte())
                    } else if (inputDataType == DataType.UINT8) {
                        inputBuffer.put(q.coerceIn(0, 255).toByte())
                    } else {
                        inputBuffer.put(q.coerceIn(-128, 127).toByte())
                    }
                }
            }
        }
        inputBuffer.rewind()

        val outputs = mutableMapOf<Int, Any>()
        for (i in 0 until interp.outputTensorCount) {
            val outTensor = interp.getOutputTensor(i)
            val shape = outTensor.shape()
            val totalSize = shape.fold(1) { acc, elem -> acc * elem }
            val outBuffer = if (outTensor.dataType() == DataType.FLOAT32) {
                ByteBuffer.allocateDirect(totalSize * 4).order(ByteOrder.nativeOrder())
            } else {
                ByteBuffer.allocateDirect(totalSize).order(ByteOrder.nativeOrder())
            }
            outputs[i] = outBuffer
        }

        interp.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

        val outputArrays = mutableMapOf<Int, FloatArray>()
        for (i in 0 until interp.outputTensorCount) {
            val outTensor = interp.getOutputTensor(i)
            val shape = outTensor.shape()
            val totalSize = shape.fold(1) { acc, elem -> acc * elem }
            val buf = outputs[i] as ByteBuffer
            buf.rewind()

            val floatArray = FloatArray(totalSize)
            val scale = outputQuantScales[i] ?: 1f
            val zp = outputQuantZeroPoints[i] ?: 0

            if (outTensor.dataType() == DataType.FLOAT32) {
                for (j in 0 until totalSize) {
                    floatArray[j] = buf.float
                }
            } else {
                for (j in 0 until totalSize) {
                    val quantizedVal = buf.get().toInt()
                    floatArray[j] = (quantizedVal - zp) * scale
                }
            }
            outputArrays[i] = floatArray
        }

        var det: Array<FloatArray>? = null
        var proto: Array<Array<FloatArray>>? = null

        for (i in 0 until interp.outputTensorCount) {
            val shape = interp.getOutputTensor(i).shape()
            val data = outputArrays[i]!!

            if (shape.size == 4) {
                val isNchwProto = shape[1] <= 64 && shape[1] < shape[2] && shape[1] < shape[3]
                if (isNchwProto) {
                    val C = shape[1]
                    val H = shape[2]
                    val W = shape[3]
                    proto = Array(H) { h -> Array(W) { FloatArray(C) } }
                    var idx = 0
                    for (c in 0 until C) {
                        for (h in 0 until H) {
                            for (w in 0 until W) {
                                proto[h][w][c] = data[idx++]
                            }
                        }
                    }
                } else {
                    val H = shape[1]
                    val W = shape[2]
                    val C = shape[3]
                    proto = Array(H) { h -> Array(W) { FloatArray(C) } }
                    var idx = 0
                    for (h in 0 until H) {
                        for (w in 0 until W) {
                            for (c in 0 until C) {
                                proto[h][w][c] = data[idx++]
                            }
                        }
                    }
                }
            } else if (shape.size == 3) {
                val dim1 = shape[1]
                val dim2 = shape[2]
                val isTransposed = dim1 < dim2
                val anchors = if (isTransposed) dim2 else dim1
                val channels = if (isTransposed) dim1 else dim2

                // If channels is small (e.g. <= 128) and anchors is large (e.g. >= 100), it's detection
                if (channels <= 128 && anchors >= 100) {
                    det = Array(anchors) { FloatArray(channels) }
                    for (a in 0 until anchors) {
                        for (c in 0 until channels) {
                            val srcIdx = if (isTransposed) c * anchors + a else a * channels + c
                            det[a][c] = data[srcIdx]
                        }
                    }
                }
            }
        }

        if (det == null || proto == null) {
            return DetectionResult(0f, null, null, emptyList())
        }

        var bestIdx = -1
        var bestScore = -1f
        for (a in det.indices) {
            val score = det[a][4]
            if (score > bestScore) {
                bestScore = score
                bestIdx = a
            }
        }

        val CONF_THRESHOLD = 0.001f // TODO: consider 0.25. At 0.001, 2 of 3 no-sidewalk frames got a false path. testimage.png scores 0.89.
        if (bestIdx == -1 || bestScore < CONF_THRESHOLD) {
            return DetectionResult(bestScore, null, null, emptyList())
        }

        val bestAnchor = det[bestIdx]
        val maxVal = maxOf(bestAnchor[0], bestAnchor[1], bestAnchor[2], bestAnchor[3])
        val isPixelCoords = maxVal > 1.0f
        val cx = if (isPixelCoords) bestAnchor[0] / inWidth.toFloat() else bestAnchor[0]
        val cy = if (isPixelCoords) bestAnchor[1] / inHeight.toFloat() else bestAnchor[1]
        val w = if (isPixelCoords) bestAnchor[2] / inWidth.toFloat() else bestAnchor[2]
        val h = if (isPixelCoords) bestAnchor[3] / inHeight.toFloat() else bestAnchor[3]

        val l = max(0f, min(1f, cx - w / 2f))
        val t = max(0f, min(1f, cy - h / 2f))
        val r = max(0f, min(1f, cx + w / 2f))
        val b = max(0f, min(1f, cy + h / 2f))

        val ph = proto.size
        val pw = proto[0].size
        val pc = proto[0][0].size
        val coeffs = FloatArray(pc) { c -> bestAnchor[5 + c] }

        val maskPixels = Array(ph) { r -> BooleanArray(pw) }
        for (mh in 0 until ph) {
            val ys = (mh + 0.5f) / ph
            for (mw in 0 until pw) {
                val xs = (mw + 0.5f) / pw
                var dot = 0f
                val protoPixel = proto[mh][mw]
                for (c in 0 until pc) {
                    dot += protoPixel[c] * coeffs[c]
                }
                val sigmoid = 1f / (1f + exp(-dot))
                val inBox = ys >= (t - 0.1f) && ys <= (b + 0.1f) && xs >= (l - 0.1f) && xs <= (r + 0.1f)
                maskPixels[mh][mw] = sigmoid > 0.35f && inBox // TODO: consider 0.5 and no box margin, see extractSidewalkEdges.
            }
        }

        val lbMaskBitmap = Bitmap.createBitmap(pw, ph, Bitmap.Config.ARGB_8888)
        for (mh in 0 until ph) {
            for (mw in 0 until pw) {
                if (maskPixels[mh][mw]) {
                    lbMaskBitmap.setPixel(mw, mh, Color.argb(128, 0, 255, 0))
                } else {
                    lbMaskBitmap.setPixel(mw, mh, Color.TRANSPARENT)
                }
            }
        }

        val maxMaskDim = 1920
        val maskWidth = if (origWidth > maxMaskDim || origHeight > maxMaskDim) {
            val s = min(maxMaskDim.toFloat() / origWidth, maxMaskDim.toFloat() / origHeight)
            round(origWidth * s).toInt()
        } else {
            origWidth
        }
        val maskHeight = if (origWidth > maxMaskDim || origHeight > maxMaskDim) {
            val s = min(maxMaskDim.toFloat() / origWidth, maxMaskDim.toFloat() / origHeight)
            round(origHeight * s).toInt()
        } else {
            origHeight
        }

        val origMaskBitmap = Bitmap.createBitmap(maskWidth, maskHeight, Bitmap.Config.ARGB_8888)
        val maskCanvas = Canvas(origMaskBitmap)
        val srcRect = Rect(
            round(padX * pw / inWidth).toInt(),
            round(padY * ph / inHeight).toInt(),
            round((padX + newWidth) * pw / inWidth).toInt(),
            round((padY + newHeight) * ph / inHeight).toInt()
        )
        val destRect = Rect(0, 0, maskWidth, maskHeight)
        maskCanvas.drawBitmap(lbMaskBitmap, srcRect, destRect, null)

        val uxl = max(0f, min(1f, (l * inWidth - padX) / newWidth))
        val uyt = max(0f, min(1f, (t * inHeight - padY) / newHeight))
        val uxr = max(0f, min(1f, (r * inWidth - padX) / newWidth))
        val uyb = max(0f, min(1f, (b * inHeight - padY) / newHeight))

        val box = floatArrayOf(uxl, uyt, uxr, uyb)
        val edgePoints = extractSidewalkEdges(origMaskBitmap)

        return DetectionResult(bestScore, box, origMaskBitmap, edgePoints)
    }

    // Taking the first and last mask pixel per row spans the road when both sidewalks are in view.
    // With the 0.35 threshold and 0.1 box margin the mask also spills past the path, and the box
    // cuts it straight, which reads as a clean false edge. Both measured with analysis/ on this model.
    // TODO: consider tracing the run under the bottom center upward, as the first app did.
    private fun extractSidewalkEdges(maskBitmap: Bitmap, cameraHeightMeters: Float = 1.2f): List<SidewalkEdgePoint> {
        val width = maskBitmap.width
        val height = maskBitmap.height
        val pixels = IntArray(width * height)
        maskBitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        var minX = width; var maxX = 0; var minY = height; var maxY = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val alpha = (pixels[y * width + x] ushr 24) and 0xFF
                if (alpha > 0) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }

        if (minX > maxX || minY > maxY) return emptyList()

        val centerX = (minX + maxX) / 2f
        val edgePoints = mutableListOf<SidewalkEdgePoint>()

        for (y in minY..maxY) {
            var leftEdgeX = -1
            for (x in minX..maxX) {
                if (((pixels[y * width + x] ushr 24) and 0xFF) > 0) {
                    leftEdgeX = x
                    break
                }
            }

            var rightEdgeX = -1
            for (x in maxX downTo minX) {
                if (((pixels[y * width + x] ushr 24) and 0xFF) > 0) {
                    rightEdgeX = x
                    break
                }
            }

            if (leftEdgeX != -1) {
                val normX = leftEdgeX.toFloat() / width
                val normY = y.toFloat() / height
                val side = if (leftEdgeX < centerX) SidewalkEdgeSide.LEFT else SidewalkEdgeSide.RIGHT
                val distance = estimateDistance(normY, cameraHeightMeters)
                edgePoints.add(SidewalkEdgePoint(Point2D(normX, normY), side, distance))
            }

            if (rightEdgeX != -1 && rightEdgeX != leftEdgeX) {
                val normX = rightEdgeX.toFloat() / width
                val normY = y.toFloat() / height
                val side = if (rightEdgeX < centerX) SidewalkEdgeSide.LEFT else SidewalkEdgeSide.RIGHT
                val distance = estimateDistance(normY, cameraHeightMeters)
                edgePoints.add(SidewalkEdgePoint(Point2D(normX, normY), side, distance))
            }
        }

        return edgePoints
    }

    private fun estimateDistance(normalizedY: Float, cameraHeightMeters: Float): Float {
        val horizonY = 0.4f
        if (normalizedY <= horizonY) return Float.MAX_VALUE
        val effectiveY = normalizedY - horizonY
        val focalFactor = 1.5f
        return (cameraHeightMeters * focalFactor) / effectiveY
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
