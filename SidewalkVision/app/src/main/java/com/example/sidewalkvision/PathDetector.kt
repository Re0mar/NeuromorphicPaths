package com.example.sidewalkvision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Locale
import kotlin.math.abs
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
    val estimatedDistanceMeters: Float?,
    // True when the mask runs off the side of the image here, so this point is the frame's border
    // and not the sidewalk's edge. Anything measuring position or fitting edge lines must skip it.
    val clipped: Boolean = false
)

data class DetectionResult(
    val score: Float,
    val box: FloatArray?, // [l, t, r, b] normalized [0, 1]
    val maskBitmap: Bitmap?,
    val edgePoints: List<SidewalkEdgePoint> = emptyList(),
    // When the camera captured the frame, from CameraX, in nanoseconds on the camera's clock. Time
    // to the edge needs the real gap between analyzed frames, which varies as frames are dropped.
    // Null for still images.
    val captureTimeNanos: Long? = null,
    // Left, top, right, bottom of the region the mask was allowed into: the box plus its margin,
    // normalized. A mask edge running along one of its sides is the box, not the sidewalk.
    val maskBounds: FloatArray? = null,
    // Camera pose from the sidewalk's edges. Null when nothing was detected.
    val pose: PoseEstimate? = null
)

class PathDetector(context: Context) {
    companion object {
        private const val TAG = "PathDetector"

        // Least score to accept a detection. At 0.001 a path was found in 2 of 3 frames with
        // none, while real sidewalks score around 0.89. The screens' labels use it too.
        const val CONFIDENCE_THRESHOLD = 0.25f

        // Within this many mask cells of the image's side, an edge point counts as clipped.
        private const val CLIP_MARGIN_CELLS = 1

        // How far past the detection box mask cells are still kept, in fractions of the model input.
        // At 0.1 a mask spilling onto the next surface ran on to the margin and stopped in a straight
        // line, which the pose read as a perfect path edge.
        private const val MASK_BOX_MARGIN = 0f

        // A cell is sidewalk when the model thinks it more likely than not. At 0.35 the mask crept
        // onto surfaces that look alike, a red bike lane or driveway tiles.
        private const val MASK_PROBABILITY = 0.5f

        // Detections low and central in the frame are where the ground ahead of the user is. A
        // score is weighted down by up to these fractions toward the sides and the top.
        private const val GROUND_PREFERENCE_X = 0.3f
        private const val GROUND_PREFERENCE_Y = 0.3f

        // Empty grid rows tolerated while tracing the sidewalk upward before it counts as ended.
        private const val MAX_GAP_ROWS = 2

        // Box values above this are pixels, at or below it normalized. See isPixelCoords.
        private const val NORMALIZED_COORDINATE_LIMIT = 2f

        // Saves what the detector received and what the model saw, for labeling and scoring
        // offline. The camera screen analyzes about 15 frames a second, so every 30th is one
        // frame every 2 seconds. Set DEBUG_DUMP_MAX to 0 to turn it off.
        const val DEBUG_DUMP_FOLDER = "pathdetector_debug"
        private const val DEBUG_DUMP_EVERY = 30
        private const val DEBUG_DUMP_MAX = 60
        private const val DEBUG_DUMP_JPEG_QUALITY = 95
    }

    private val appContext = context.applicationContext
    // Frames from different app sessions get different file names instead of overwriting.
    private val dumpSessionId = System.currentTimeMillis()
    private var framesSeen = 0
    private var framesDumped = 0

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

    /**
     * Detects the sidewalk and estimates the camera's pose from its edges.
     *
     * @param intrinsics the camera's focal length, for pitch, heading, height and edge distances.
     * Null for images from an unknown camera, which then get position across the path only.
     */
    fun detect(bitmap: Bitmap, intrinsics: CameraIntrinsics? = null): DetectionResult {
        val interp = interpreter ?: return DetectionResult(0f, null, null)

        val origWidth = bitmap.width
        val origHeight = bitmap.height

        // Letterbox
        val scale = min(inWidth.toFloat() / origWidth, inHeight.toFloat() / origHeight)
        val newWidth = max(1, round(origWidth * scale).toInt())
        val newHeight = max(1, round(origHeight * scale).toInt())
        val padX = (inWidth - newWidth) / 2f
        val padY = (inHeight - newHeight) / 2f

        val scaledBitmap = shrinkSmoothly(bitmap, newWidth, newHeight)
        val canvasBitmap = Bitmap.createBitmap(inWidth, inHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(canvasBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.drawBitmap(scaledBitmap, padX, padY, null)
        maybeDumpDebugFrames(bitmap, canvasBitmap)

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
        var bestScore = 0f
        var bestWeighted = 0f
        var topScore = 0f
        for (a in det.indices) {
            val score = det[a][4]
            topScore = max(topScore, score)
            if (score < CONFIDENCE_THRESHOLD) continue
            val anchorIsPixels = maxOf(det[a][0], det[a][1], det[a][2], det[a][3]) > NORMALIZED_COORDINATE_LIMIT
            val anchorCx = if (anchorIsPixels) det[a][0] / inWidth else det[a][0]
            val anchorCy = if (anchorIsPixels) det[a][1] / inHeight else det[a][1]
            val imageCx = (anchorCx * inWidth - padX) / newWidth
            val imageCy = (anchorCy * inHeight - padY) / newHeight
            // A box centered on the gray padding isn't looking at the picture at all.
            if (imageCx !in 0f..1f || imageCy !in 0f..1f) continue
            val weight = (1f - GROUND_PREFERENCE_X * abs(imageCx - 0.5f) * 2f) *
                (1f - GROUND_PREFERENCE_Y * (1f - imageCy))
            if (score * weight > bestWeighted) {
                bestWeighted = score * weight
                bestScore = score
                bestIdx = a
            }
        }

        if (bestIdx == -1) {
            return DetectionResult(topScore, null, null, emptyList())
        }

        val bestAnchor = det[bestIdx]
        val maxVal = maxOf(bestAnchor[0], bestAnchor[1], bestAnchor[2], bestAnchor[3])
        // Normalized boxes run slightly past 1.0 when the sidewalk fills the frame's width, for
        // example w = 1.0011 on testimage.png. Testing against 1.0 read those as pixels, divided
        // them by the input size, and left the mask empty. Pixel boxes run up to the input size.
        val isPixelCoords = maxVal > NORMALIZED_COORDINATE_LIMIT
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
                val inBox = ys >= (t - MASK_BOX_MARGIN) && ys <= (b + MASK_BOX_MARGIN) &&
                    xs >= (l - MASK_BOX_MARGIN) && xs <= (r + MASK_BOX_MARGIN)
                maskPixels[mh][mw] = sigmoid > MASK_PROBABILITY && inBox
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
        // One mask cell, in pixels of the full-size mask. The model's box usually stops a little
        // inside the frame, so a sidewalk running off the side often ends one cell short of it.
        val maskCellPx = maskWidth.toFloat() / (newWidth * pw.toFloat() / inWidth)
        val edgesWithoutDistance = traceSidewalkEdges(maskPixels, padX, padY, newWidth, newHeight)

        // The region the mask was allowed into: the box plus its margin, in image fractions.
        val maskBounds = floatArrayOf(
            max(0f, min(1f, ((l - MASK_BOX_MARGIN) * inWidth - padX) / newWidth)),
            max(0f, min(1f, ((t - MASK_BOX_MARGIN) * inHeight - padY) / newHeight)),
            max(0f, min(1f, ((r + MASK_BOX_MARGIN) * inWidth - padX) / newWidth)),
            max(0f, min(1f, ((b + MASK_BOX_MARGIN) * inHeight - padY) / newHeight))
        )
        val maskCellFraction = maskCellPx.toDouble() / maskWidth
        val pose = estimatePose(edgesWithoutDistance, maskWidth, maskHeight, maskCellFraction, maskBounds, intrinsics)
        // Distances only from a pose the edges support, instead of a fixed horizon and camera height.
        val edgePoints = edgesWithoutDistance.map { point ->
            val distance = groundDistanceMeters(pose, point.position.y.toDouble() * maskHeight, maskHeight)
            point.copy(estimatedDistanceMeters = distance?.toFloat())
        }

        return DetectionResult(bestScore, box, origMaskBitmap, edgePoints, maskBounds = maskBounds, pose = pose)
    }

    /**
     * The sidewalk's left and right edge on each grid row, traced upward from the bottom center.
     *
     * On the first row with mask cells, the run containing the center column wins, or else the
     * nearest run. On each row above, only a run overlapping the one below counts, and the largest
     * overlap wins. Other patches of mask, a sidewalk across the road or a spill onto a driveway,
     * are never joined to the one the user stands on. The same rule as trace_path in analysis/.
     */
    private fun traceSidewalkEdges(
        maskPixels: Array<BooleanArray>,
        padX: Float,
        padY: Float,
        newWidth: Int,
        newHeight: Int
    ): List<SidewalkEdgePoint> {
        val gridHeight = maskPixels.size
        val gridWidth = maskPixels[0].size
        fun imageX(gridX: Float) = max(0f, min(1f, (gridX / gridWidth * inWidth - padX) / newWidth))
        fun imageY(gridY: Float) = max(0f, min(1f, (gridY / gridHeight * inHeight - padY) / newHeight))
        // Columns whose center falls on the picture rather than the gray padding.
        val contentColumns = (0 until gridWidth).filter { column ->
            val centerPx = (column + 0.5f) / gridWidth * inWidth
            centerPx >= padX && centerPx <= padX + newWidth
        }
        val firstContentColumn = contentColumns.first()
        val lastContentColumn = contentColumns.last()

        val centerColumn = gridWidth / 2
        var previousLeft = -1
        var previousRight = -1
        var gapRows = 0
        val edgePoints = mutableListOf<SidewalkEdgePoint>()

        for (gridY in gridHeight - 1 downTo 0) {
            val row = maskPixels[gridY]
            var chosenLeft = -1
            var chosenRight = -1
            var chosenRank = Int.MIN_VALUE
            var x = 0
            while (x < gridWidth) {
                if (!row[x]) {
                    x++
                    continue
                }
                val start = x
                while (x < gridWidth && row[x]) x++
                val end = x - 1
                val rank = if (previousLeft == -1) {
                    if (centerColumn in start..end) 0 else -minOf(abs(start - centerColumn), abs(end - centerColumn))
                } else {
                    minOf(end, previousRight) - maxOf(start, previousLeft) + 1
                }
                if (previousLeft != -1 && rank <= 0) continue
                if (rank > chosenRank) {
                    chosenRank = rank
                    chosenLeft = start
                    chosenRight = end
                }
            }

            if (chosenLeft == -1) {
                if (previousLeft != -1 && ++gapRows > MAX_GAP_ROWS) break
                continue
            }
            gapRows = 0
            previousLeft = chosenLeft
            previousRight = chosenRight

            val y = imageY(gridY + 0.5f)
            val leftClipped = chosenLeft <= firstContentColumn + CLIP_MARGIN_CELLS
            val rightClipped = chosenRight >= lastContentColumn - CLIP_MARGIN_CELLS
            edgePoints.add(SidewalkEdgePoint(Point2D(imageX(chosenLeft.toFloat()), y), SidewalkEdgeSide.LEFT, null, leftClipped))
            edgePoints.add(SidewalkEdgePoint(Point2D(imageX(chosenRight + 1f), y), SidewalkEdgeSide.RIGHT, null, rightClipped))
        }
        return edgePoints
    }

    /**
     * Shrinks by halving until one last step of at most 2x remains. A single bilinear step from a
     * large frame samples only a few source pixels per output pixel and aliases, so fine texture
     * such as paving joints turns into noise unlike anything the model saw in training.
     */
    private fun shrinkSmoothly(source: Bitmap, width: Int, height: Int): Bitmap {
        var current = source
        while (current.width >= width * 2 && current.height >= height * 2) {
            val half = Bitmap.createScaledBitmap(current, maxOf(width, current.width / 2), maxOf(height, current.height / 2), true)
            if (current !== source) current.recycle()
            current = half
        }
        val result = Bitmap.createScaledBitmap(current, width, height, true)
        if (current !== source && current !== result) current.recycle()
        return result
    }

    private fun maybeDumpDebugFrames(raw: Bitmap, letterboxed: Bitmap) {
        val frameNumber = framesSeen++
        if (DEBUG_DUMP_MAX <= 0 || framesDumped >= DEBUG_DUMP_MAX || frameNumber % DEBUG_DUMP_EVERY != 0) return
        val directory = appContext.getExternalFilesDir(DEBUG_DUMP_FOLDER) ?: return
        try {
            for ((suffix, image) in listOf("raw" to raw, "input" to letterboxed)) {
                File(directory, "session_${dumpSessionId}_frame_%03d_$suffix.jpg".format(Locale.US, framesDumped))
                    .outputStream()
                    .use { stream -> image.compress(Bitmap.CompressFormat.JPEG, DEBUG_DUMP_JPEG_QUALITY, stream) }
            }
            framesDumped++
        } catch (ioException: IOException) {
            // A full or missing storage folder costs a debug frame, never a detection.
            Log.w(TAG, "Could not save debug frame: ${ioException.message}")
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
