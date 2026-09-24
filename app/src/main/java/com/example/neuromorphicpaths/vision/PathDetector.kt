package com.example.neuromorphicpaths.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * PathDetector handles the loading and execution of a TensorFlow Lite model
 * (specifically Ultralytics YOLO segmentation model, e.g. YOLOv8-seg / YOLOv11-seg)
 * for detecting walkways and their boundaries in an image.
 *
 * All coordinates in the returned [PathResult] are normalized [0..1] relative to the
 * ORIGINAL bitmap passed to [detectPath] (letterbox padding is removed again).
 */
class PathDetector(context: Context) {
    companion object {
        private const val TAG = "PathDetector"
        private const val MODEL_PATH = "best_int8.tflite"
        private const val MAX_CPU_THREADS = 6

        private const val CONF_THRESHOLD = 0.25f

        private const val BOXES_NORMALIZED = true

        private const val PAD_GRAY = 114

        // ---- Ground-region prior -------------------------------------------------------------
        // The walkway is in front of / below the wearer, so detections are only accepted if, in
        // original-image coordinates (0 = top, 1 = bottom), the box centre is at least this far
        // down and the box reaches at least this far down. Also rejects boxes whose centre falls
        // in the gray letterbox padding. Set both to 0f to disable.
        private const val MIN_CENTER_Y = 0f
        private const val MIN_BOTTOM_Y = 0f

        // Among the accepted detections, score is scaled down the further the box centre is from
        // the bottom-centre of the frame (0 = ignore position, 1 = strongest pull).
        private const val PRIOR_X = 0.30f
        private const val PRIOR_Y = 0.30f

        // While tracing the walkway upwards from the bottom centre, tolerate this many rows
        // without an overlapping mask run before stopping.
        private const val MAX_GAP_ROWS = 2

        // Debug: save raw + letterboxed frames (every Nth frame, up to MAX of them) to
        // <app external files>/pathdetector_debug so you can see exactly what the model sees.
        // Set DEBUG_DUMP_MAX to 0 to turn off.
        private const val DEBUG_DUMP_MAX = 6
        private const val DEBUG_DUMP_EVERY = 10

        private const val MODE_FLOAT32 = 0
        private const val MODE_INT8 = 1
        private const val MODE_UINT8 = 2
        private const val MODE_UNSUPPORTED = -1

        private fun modeOf(type: DataType): Int = when (type) {
            DataType.FLOAT32 -> MODE_FLOAT32
            DataType.INT8 -> MODE_INT8
            DataType.UINT8 -> MODE_UINT8
            else -> MODE_UNSUPPORTED
        }
    }

    private val appContext = context.applicationContext
    private var interpreter: Interpreter? = null
    private var frameCounter = 0
    private var dumpedFrames = 0

    // Input shape parameters
    private var inputWidth = 320
    private var inputHeight = 320
    private var isInputNCHW = true

    // Which output tensor is which (resolved from shapes, not assumed)
    private var detOutputIndex = 0
    private var protoOutputIndex = 1

    // Output "detections": [1, 37, 2100] or [1, 2100, 37]
    private var numAnchors = 2100
    private var numOutput0Channels = 37 // 4 box + 1 class + 32 mask coeffs
    private var isOutput0ChannelsFirst = true

    // Output "protos": [1, 32, 80, 80] or [1, 80, 80, 32]
    private var numProtoMasks = 32
    private var protoHeight = 80
    private var protoWidth = 80
    private var isOutput1ChannelsFirst = true

    // Tensor I/O types. The model may use FLOAT32 or quantized INT8/UINT8 tensors (e.g. best_int8.tflite).
    // Quantized tensors are converted exactly like the Python reference:
    //   input : q = round(x / scale + zeroPoint), clamped to the type range
    //   output: x = (q - zeroPoint) * scale
    private var ioSupported = false
    private var warnedScoreRange = false

    private var inputMode = MODE_FLOAT32
    private var inputScale = 1f
    private var inputZeroPoint = 0
    private val inputLut = ByteArray(256) // 0..255 pixel value -> quantized byte

    private var out0Mode = MODE_FLOAT32
    private var out0Scale = 1f
    private var out0ZeroPoint = 0

    private var out1Mode = MODE_FLOAT32
    private var out1Scale = 1f
    private var out1ZeroPoint = 0

    // Reusable direct buffers for zero-allocation performance at 24 FPS
    private var inputBuffer: ByteBuffer? = null
    private var output0Buffer: ByteBuffer? = null
    private var output1Buffer: ByteBuffer? = null

    // Reusable arrays
    private var pixelArray = IntArray(320 * 320)
    private var maskCoeffs = FloatArray(32)
    private var maskGrid = Array(80) { BooleanArray(80) }

    // Reusable letterbox surface
    private var letterboxBitmap: Bitmap? = null
    private var letterboxCanvas: Canvas? = null
    private val letterboxPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val letterboxDst = RectF()

    // Letterbox geometry of the current frame (in model-input pixels)
    private var lbPadX = 0f
    private var lbPadY = 0f
    private var lbNewW = 1f
    private var lbNewH = 1f
    private var lbSrcW = 0
    private var lbSrcH = 0

    /** Set to false to skip building [PathResult.edgeBlocks] (saves a few allocations per frame). */
    @Volatile
    var computeEdgeBlocks = true

    init {
        try {
            val options = Interpreter.Options().apply {
                setNumThreads(
                    Runtime.getRuntime().availableProcessors().coerceIn(1, MAX_CPU_THREADS)
                )
            }

            val modelBuffer = FileUtil.loadMappedFile(context, MODEL_PATH)
            val interp = Interpreter(modelBuffer, options)
            interpreter = interp

            // 1. Inspect Input Tensor
            val inputTensor = interp.getInputTensor(0)
            logTensor("IN", inputTensor)
            inputMode = modeOf(inputTensor.dataType())
            inputScale = inputTensor.quantizationParams().scale
            inputZeroPoint = inputTensor.quantizationParams().zeroPoint
            if (inputMode == MODE_INT8 || inputMode == MODE_UINT8) buildInputLut()

            val inputShape = inputTensor.shape() // e.g. [1, 3, 320, 320] or [1, 320, 320, 3]
            if (inputShape.size == 4) {
                if (inputShape[1] == 3 || inputShape[1] == 1) {
                    isInputNCHW = true
                    inputHeight = inputShape[2]
                    inputWidth = inputShape[3]
                } else {
                    isInputNCHW = false
                    inputHeight = inputShape[1]
                    inputWidth = inputShape[2]
                }
            }

            // Allocate input buffer sized by the tensor (4 bytes/elem for float32, 1 for int8/uint8)
            inputBuffer = ByteBuffer.allocateDirect(inputTensor.numBytes()).apply {
                order(ByteOrder.nativeOrder())
            }
            pixelArray = IntArray(inputWidth * inputHeight)

            letterboxBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
            letterboxCanvas = Canvas(letterboxBitmap!!)

            // 2. Inspect Output Tensors
            val numOutputs = interp.outputTensorCount
            Log.d(TAG, "TFLite Model outputs count: $numOutputs")
            for (i in 0 until numOutputs) {
                val t = interp.getOutputTensor(i)
                logTensor("OUT$i", t)
            }

            if (numOutputs >= 2) {
                // Identify tensors by rank: rank 3 = detections, rank 4 = prototype masks.
                var det = -1
                var proto = -1
                for (i in 0 until numOutputs) {
                    when (interp.getOutputTensor(i).shape().size) {
                        3 -> if (det == -1) det = i
                        4 -> if (proto == -1) proto = i
                    }
                }
                if (det == -1 || proto == -1) {
                    Log.e(TAG, "Could not identify detection (rank 3) and proto (rank 4) outputs; " +
                            "falling back to order 0/1")
                    det = 0
                    proto = 1
                }
                detOutputIndex = det
                protoOutputIndex = proto

                // Detections & coefficients
                val shape0 = interp.getOutputTensor(detOutputIndex).shape()
                if (shape0.size == 3) {
                    if (shape0[1] < shape0[2]) { // e.g. [1, 37, 2100]
                        isOutput0ChannelsFirst = true
                        numOutput0Channels = shape0[1]
                        numAnchors = shape0[2]
                    } else { // e.g. [1, 2100, 37]
                        isOutput0ChannelsFirst = false
                        numAnchors = shape0[1]
                        numOutput0Channels = shape0[2]
                    }
                }

                // Prototype masks
                val shape1 = interp.getOutputTensor(protoOutputIndex).shape()
                if (shape1.size == 4) {
                    if (shape1[1] == 32 || shape1[1] < shape1[2]) { // e.g. [1, 32, 80, 80]
                        isOutput1ChannelsFirst = true
                        numProtoMasks = shape1[1]
                        protoHeight = shape1[2]
                        protoWidth = shape1[3]
                    } else { // e.g. [1, 80, 80, 32]
                        isOutput1ChannelsFirst = false
                        protoHeight = shape1[1]
                        protoWidth = shape1[2]
                        numProtoMasks = shape1[3]
                    }
                }

                maskGrid = Array(protoHeight) { BooleanArray(protoWidth) }
                maskCoeffs = FloatArray(numProtoMasks)

                val detTensor = interp.getOutputTensor(detOutputIndex)
                val protoTensor = interp.getOutputTensor(protoOutputIndex)

                out0Mode = modeOf(detTensor.dataType())
                out0Scale = detTensor.quantizationParams().scale
                out0ZeroPoint = detTensor.quantizationParams().zeroPoint
                out1Mode = modeOf(protoTensor.dataType())
                out1Scale = protoTensor.quantizationParams().scale
                out1ZeroPoint = protoTensor.quantizationParams().zeroPoint

                ioSupported = inputMode != MODE_UNSUPPORTED &&
                        out0Mode != MODE_UNSUPPORTED &&
                        out1Mode != MODE_UNSUPPORTED
                if (!ioSupported) {
                    Log.e(
                        TAG,
                        "Unsupported tensor types (in=${inputTensor.dataType()}, " +
                                "det=${detTensor.dataType()}, proto=${protoTensor.dataType()}). " +
                                "Only FLOAT32, INT8 and UINT8 are handled; detection is disabled."
                    )
                }

                output0Buffer = ByteBuffer.allocateDirect(detTensor.numBytes()).apply {
                    order(ByteOrder.nativeOrder())
                }
                output1Buffer = ByteBuffer.allocateDirect(protoTensor.numBytes()).apply {
                    order(ByteOrder.nativeOrder())
                }

                Log.d(
                    TAG,
                    "PathDetector initialized YOLO-seg: Input (${inputWidth}x${inputHeight}, NCHW=$isInputNCHW), " +
                            "Det=out$detOutputIndex ($numOutput0Channels x $numAnchors, ChFirst=$isOutput0ChannelsFirst), " +
                            "Proto=out$protoOutputIndex ($numProtoMasks x ${protoWidth}x${protoHeight}, ChFirst=$isOutput1ChannelsFirst)"
                )
            } else {
                Log.w(TAG, "Model has only $numOutputs output tensor(s). Expected YOLO-seg 2 output tensors.")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize TFLite interpreter for $MODEL_PATH", e)
        }
    }

    private fun logTensor(label: String, t: Tensor) {
        val q = t.quantizationParams()
        Log.d(
            TAG,
            "$label: type=${t.dataType()} shape=${t.shape().toList()} quant(scale=${q.scale}, zeroPoint=${q.zeroPoint})"
        )
    }

    /**
     * Resizes [src] into the reusable letterbox bitmap (aspect preserved, gray padding),
     * exactly like Ultralytics preprocessing, and records the geometry so results can be
     * mapped back to the original image.
     */
    private fun letterbox(src: Bitmap): Bitmap? {
        val lb = letterboxBitmap ?: return null
        val canvas = letterboxCanvas ?: return null

        val scale = min(inputWidth.toFloat() / src.width, inputHeight.toFloat() / src.height)
        val newW = (src.width * scale).roundToInt().coerceIn(1, inputWidth)
        val newH = (src.height * scale).roundToInt().coerceIn(1, inputHeight)
        // Whole-pixel padding (the Python reference pastes at round(pad)); keeps the image
        // pixel-aligned and makes the un-letterbox mapping exact.
        val padX = ((inputWidth - newW) / 2f).roundToInt()
        val padY = ((inputHeight - newH) / 2f).roundToInt()

        lbSrcW = src.width
        lbSrcH = src.height
        lbNewW = newW.toFloat()
        lbNewH = newH.toFloat()
        lbPadX = padX.toFloat()
        lbPadY = padY.toFloat()

        // A single bilinear draw only samples 2x2 source pixels, so a big camera frame (e.g. 1080p
        // -> 320) gets aliased and looks different from what PIL produces (PIL's resize is
        // antialiased). Halve repeatedly first so the final step is at most a 2x reduction.
        var source = src
        while (source.width >= newW * 2 && source.height >= newH * 2) {
            val next = Bitmap.createScaledBitmap(
                source,
                maxOf(newW, source.width / 2),
                maxOf(newH, source.height / 2),
                true
            )
            if (source !== src) source.recycle()
            source = next
        }

        canvas.drawColor(Color.rgb(PAD_GRAY, PAD_GRAY, PAD_GRAY))
        letterboxDst.set(padX.toFloat(), padY.toFloat(), (padX + newW).toFloat(), (padY + newH).toFloat())
        canvas.drawBitmap(source, null, letterboxDst, letterboxPaint)
        if (source !== src) source.recycle()
        return lb
    }

    /** Pixel value 0..255 -> quantized input byte, matching `round(x/255 / scale + zeroPoint)` in Python. */
    private fun buildInputLut() {
        val minV = if (inputMode == MODE_INT8) -128 else 0
        val maxV = if (inputMode == MODE_INT8) 127 else 255
        for (v in 0..255) {
            val q = ((v / 255.0f) / inputScale + inputZeroPoint).roundToInt().coerceIn(minV, maxV)
            inputLut[v] = q.toByte()
        }
    }

    /** Writes one 0..255 colour sample into the input buffer in the tensor's native type. */
    private fun putSample(buf: ByteBuffer, v: Int) {
        if (inputMode == MODE_FLOAT32) buf.putFloat(v / 255.0f) else buf.put(inputLut[v])
    }

    /** Reads element [index] of an output buffer as a real (dequantized) float. */
    private fun readValue(buf: ByteBuffer, index: Int, mode: Int, scale: Float, zeroPoint: Int): Float =
        when (mode) {
            MODE_FLOAT32 -> buf.getFloat(index * 4)
            MODE_INT8 -> (buf.get(index).toInt() - zeroPoint) * scale
            else -> ((buf.get(index).toInt() and 0xFF) - zeroPoint) * scale
        }

    // Letterboxed-input normalized coordinate -> original-image normalized coordinate
    private fun unletterboxX(nx: Float): Float = ((nx * inputWidth - lbPadX) / lbNewW).coerceIn(0f, 1f)
    private fun unletterboxY(ny: Float): Float = ((ny * inputHeight - lbPadY) / lbNewH).coerceIn(0f, 1f)

    private fun maybeDumpDebugFrames(raw: Bitmap, input: Bitmap) {
        val n = frameCounter++
        if (n == 0) {
            Log.d(TAG, "First frame: ${raw.width}x${raw.height} -> letterbox ${inputWidth}x$inputHeight " +
                    "(content ${lbNewW.toInt()}x${lbNewH.toInt()}, pad ${lbPadX.toInt()},${lbPadY.toInt()})")
        }
        if (DEBUG_DUMP_MAX <= 0 || dumpedFrames >= DEBUG_DUMP_MAX || n % DEBUG_DUMP_EVERY != 0) return
        try {
            val dir = appContext.getExternalFilesDir("pathdetector_debug") ?: return
            dir.mkdirs()
            for ((suffix, bmp) in listOf("raw" to raw, "input" to input)) {
                File(dir, "frame_${dumpedFrames}_$suffix.jpg").outputStream().use {
                    bmp.compress(Bitmap.CompressFormat.JPEG, 90, it)
                }
            }
            Log.d(TAG, "Saved debug frame $dumpedFrames to ${dir.absolutePath}")
            dumpedFrames++
        } catch (e: Exception) {
            Log.w(TAG, "Could not save debug frame: ${e.message}")
        }
    }

    @Synchronized
    fun detectPath(bitmap: Bitmap): PathResult? {
        val interp = interpreter ?: return null
        val inBuffer = inputBuffer ?: return null
        val out0Buffer = output0Buffer ?: return null
        val out1Buffer = output1Buffer ?: return null
        if (!ioSupported) return null

        // 1. Preprocess: letterbox -> input buffer (float [0,1], or quantized int8/uint8)
        val prepared = letterbox(bitmap) ?: return null
        maybeDumpDebugFrames(bitmap, prepared)

        prepared.getPixels(pixelArray, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        inBuffer.rewind()
        val numPixels = inputWidth * inputHeight

        if (isInputNCHW) {
            for (i in 0 until numPixels) putSample(inBuffer, (pixelArray[i] shr 16) and 0xFF) // R
            for (i in 0 until numPixels) putSample(inBuffer, (pixelArray[i] shr 8) and 0xFF)  // G
            for (i in 0 until numPixels) putSample(inBuffer, pixelArray[i] and 0xFF)          // B
        } else {
            for (i in 0 until numPixels) {
                val pixel = pixelArray[i]
                putSample(inBuffer, (pixel shr 16) and 0xFF)
                putSample(inBuffer, (pixel shr 8) and 0xFF)
                putSample(inBuffer, pixel and 0xFF)
            }
        }

        inBuffer.rewind()
        out0Buffer.rewind()
        out1Buffer.rewind()

        // 2. Run Multi-Output Inference (output indices resolved from tensor shapes)
        val outputsMap = mapOf(
            detOutputIndex to out0Buffer,
            protoOutputIndex to out1Buffer
        )
        try {
            interp.runForMultipleInputsOutputs(arrayOf(inBuffer), outputsMap)
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}", e)
            return null
        }

        out0Buffer.rewind()
        out1Buffer.rewind()

        // 3. Process YOLO-seg Output
        return processYoloOutputs(out0Buffer, out1Buffer)
    }

    private fun processYoloOutputs(
        out0: ByteBuffer,
        out1: ByteBuffer
    ): PathResult {
        var bestAnchorIndex = -1
        var bestScore = 0.0f      // raw score of the chosen anchor
        var bestAdjusted = 0.0f   // score after the ground-region prior
        var topRawScore = 0.0f    // best raw score anywhere (diagnostics only)
        var topRawAnchor = -1
        var rejected = 0          // confident detections dropped by the ground-region filter
        var minRaw = Float.MAX_VALUE

        val numCoeffs = minOf(numProtoMasks, numOutput0Channels - 5)

        for (i in 0 until numAnchors) {
            // Already a probability in the TFLite export - no sigmoid here.
            val score = getOutput0Value(out0, 4, i)
            if (score < minRaw) minRaw = score
            if (score > topRawScore) {
                topRawScore = score
                topRawAnchor = i
            }
            if (score < CONF_THRESHOLD) continue

            // Where does this box sit in the ORIGINAL image? (0,0 = top-left, 1,1 = bottom-right)
            val bcx = getOutput0Value(out0, 0, i)
            val bcy = getOutput0Value(out0, 1, i)
            val bh = getOutput0Value(out0, 3, i)
            val ncy = if (BOXES_NORMALIZED) bcy else bcy / inputHeight
            val nh = if (BOXES_NORMALIZED) bh else bh / inputHeight
            val ncx = if (BOXES_NORMALIZED) bcx else bcx / inputWidth
            val cxImg = (ncx * inputWidth - lbPadX) / lbNewW
            val cyImg = (ncy * inputHeight - lbPadY) / lbNewH
            val bottomImg = ((ncy + nh / 2f) * inputHeight - lbPadY) / lbNewH

            if (cxImg < 0f || cxImg > 1f || cyImg < MIN_CENTER_Y || cyImg > 1f || bottomImg < MIN_BOTTOM_Y) {
                rejected++
                continue
            }

            // Prefer boxes centred low and in the middle of the frame.
            val prior = (1f - PRIOR_X * abs(cxImg - 0.5f) * 2f) * (1f - PRIOR_Y * (1f - cyImg))
            val adjusted = score * prior
            if (adjusted > bestAdjusted) {
                bestAdjusted = adjusted
                bestScore = score
                bestAnchorIndex = i
            }
        }

        // Probabilities must lie in [0, 1]. Values outside mean this export emits raw logits.
        if (!warnedScoreRange && (topRawScore > 1.0f || minRaw < 0.0f)) {
            warnedScoreRange = true
            Log.w(
                TAG,
                "Class scores outside [0,1] (min=$minRaw, max=$topRawScore): this export likely " +
                        "outputs raw logits. Apply sigmoid to the score in processYoloOutputs."
            )
        }

        if (bestAnchorIndex == -1) {
            val debugMsg = if (topRawAnchor == -1) {
                "No anchors scored"
            } else {
                String.format(
                    Locale.US,
                    "No walkway (top score %.1f%% @anchor %d, %d detection(s) rejected outside ground region)",
                    topRawScore * 100f, topRawAnchor, rejected
                )
            }
            return PathResult(emptyList(), topScore = topRawScore, debugInfo = debugMsg)
        }

        // Extract bounding box for the chosen candidate (in letterboxed input space)
        val cx = getOutput0Value(out0, 0, bestAnchorIndex)
        val cy = getOutput0Value(out0, 1, bestAnchorIndex)
        val w = getOutput0Value(out0, 2, bestAnchorIndex)
        val h = getOutput0Value(out0, 3, bestAnchorIndex)

        // Normalize box coordinates to [0.0, 1.0] of the letterboxed input (decided once, by constant)
        val normCx = if (BOXES_NORMALIZED) cx else cx / inputWidth
        val normCy = if (BOXES_NORMALIZED) cy else cy / inputHeight
        val normW = if (BOXES_NORMALIZED) w else w / inputWidth
        val normH = if (BOXES_NORMALIZED) h else h / inputHeight

        val lbLeft = (normCx - normW / 2f).coerceIn(0f, 1f)
        val lbTop = (normCy - normH / 2f).coerceIn(0f, 1f)
        val lbRight = (normCx + normW / 2f).coerceIn(0f, 1f)
        val lbBottom = (normCy + normH / 2f).coerceIn(0f, 1f)

        // Same box mapped back to the original image (this is what callers get)
        val topBox = RectF(
            unletterboxX(lbLeft),
            unletterboxY(lbTop),
            unletterboxX(lbRight),
            unletterboxY(lbBottom)
        )

        // Extract mask coefficients for selected anchor
        for (c in 0 until numCoeffs) {
            maskCoeffs[c] = getOutput0Value(out0, 5 + c, bestAnchorIndex)
        }

        // Clear mask grid
        for (y in 0 until protoHeight) {
            maskGrid[y].fill(false)
        }

        // Multiply mask coefficients by prototype masks, cropped to the (letterbox-space) box.
        // Cell centres are used so the grid lines up with the input image.
        var positiveMaskCount = 0
        for (y in 0 until protoHeight) {
            val normY = (y + 0.5f) / protoHeight
            if (normY < lbTop || normY > lbBottom) continue

            for (x in 0 until protoWidth) {
                val normX = (x + 0.5f) / protoWidth
                if (normX < lbLeft || normX > lbRight) continue

                var sum = 0.0f
                for (c in 0 until numCoeffs) {
                    sum += maskCoeffs[c] * getOutput1Value(out1, c, y, x)
                }
                // sum > 0.0 corresponds to sigmoid > 0.5
                if (sum > 0.0f) {
                    maskGrid[y][x] = true
                    positiveMaskCount++
                }
            }
        }

        val edgeBlocks = if (computeEdgeBlocks) extractEdgeBlocks() else emptyList()

        // Trace the walkway upwards starting from the bottom centre of the mask: on the first row
        // that has mask pixels take the run nearest the centre column, then on each row above keep
        // only the run that overlaps the previous one. Disconnected blobs (e.g. a patch of grass
        // far to one side) are ignored instead of stretching the outline across the frame.
        val leftPathPoints = mutableListOf<Point>()
        val rightPathPoints = mutableListOf<Point>()

        val centerX = protoWidth / 2
        var prevL = -1
        var prevR = -1
        var gapRows = 0

        for (y in protoHeight - 1 downTo 0) {
            val row = maskGrid[y]
            var chosenL = -1
            var chosenR = -1
            var chosenRank = Int.MIN_VALUE

            var x = 0
            while (x < protoWidth) {
                if (!row[x]) {
                    x++
                    continue
                }
                val start = x
                while (x < protoWidth && row[x]) x++
                val end = x - 1

                val rank: Int
                if (prevL == -1) {
                    // Starting row: closest to the centre column wins (0 = contains it).
                    rank = if (centerX in start..end) 0 else -minOf(abs(start - centerX), abs(end - centerX))
                } else {
                    // Later rows: must overlap the run below; larger overlap wins.
                    rank = minOf(end, prevR) - maxOf(start, prevL) + 1
                    if (rank <= 0) continue
                }
                if (rank > chosenRank) {
                    chosenRank = rank
                    chosenL = start
                    chosenR = end
                }
            }

            if (chosenL == -1) {
                if (prevL != -1 && ++gapRows > MAX_GAP_ROWS) break
                continue
            }
            gapRows = 0
            prevL = chosenL
            prevR = chosenR

            val normY = (y + 0.5f) / protoHeight
            val normLeftX = chosenL.toFloat() / protoWidth
            val normRightX = (chosenR + 1).toFloat() / protoWidth

            val oy = unletterboxY(normY)
            leftPathPoints.add(Point(unletterboxX(normLeftX), oy))
            rightPathPoints.add(Point(unletterboxX(normRightX), oy))
        }

        val debugMsg = String.format(
            Locale.US,
            "Walkway: %.1f%% @anchor %d | Mask: %d px | Box: [%.2f, %.2f, %.2f, %.2f]",
            bestScore * 100f,
            bestAnchorIndex,
            positiveMaskCount,
            topBox.left, topBox.top, topBox.right, topBox.bottom
        )

        // If mask inside bounding box has no positive pixels, construct boundary from bounding box itself
        if (leftPathPoints.isEmpty()) {
            val boxBoundary = listOf(
                Point(topBox.left, topBox.top),
                Point(topBox.right, topBox.top),
                Point(topBox.right, topBox.bottom),
                Point(topBox.left, topBox.bottom)
            )
            return PathResult(
                boundaries = listOf(boxBoundary),
                topScore = bestScore,
                topBox = topBox,
                debugInfo = debugMsg,
                edgeBlocks = edgeBlocks,
                imageWidth = lbSrcW,
                imageHeight = lbSrcH
            )
        }

        val fullBoundary = mutableListOf<Point>()
        fullBoundary.addAll(leftPathPoints)
        fullBoundary.addAll(rightPathPoints.reversed())

        return PathResult(
            boundaries = listOf(fullBoundary),
            topScore = bestScore,
            topBox = topBox,
            debugInfo = debugMsg,
            edgeBlocks = edgeBlocks,
            imageWidth = lbSrcW,
            imageHeight = lbSrcH
        )
    }

    /**
     * Returns every mask cell that touches a non-walkway cell (4-neighbourhood), i.e. the blocks
     * forming the outline of the inferred sidewalk. Rows run top to bottom, left to right.
     * Neighbours outside the image (letterbox padding / outside the grid) are not treated as
     * "empty", so the frame border is not reported as a sidewalk edge.
     */
    private fun extractEdgeBlocks(): List<EdgeBlock> {
        val blocks = ArrayList<EdgeBlock>()
        for (y in 0 until protoHeight) {
            for (x in 0 until protoWidth) {
                if (!maskGrid[y][x]) continue
                val left = isEmptyImageCell(x - 1, y)
                val right = isEmptyImageCell(x + 1, y)
                val top = isEmptyImageCell(x, y - 1)
                val bottom = isEmptyImageCell(x, y + 1)
                if (!(left || right || top || bottom)) continue

                val l = unletterboxX(x.toFloat() / protoWidth)
                val r = unletterboxX((x + 1).toFloat() / protoWidth)
                val t = unletterboxY(y.toFloat() / protoHeight)
                val b = unletterboxY((y + 1).toFloat() / protoHeight)
                blocks.add(
                    EdgeBlock(
                        gridX = x,
                        gridY = y,
                        rect = RectF(l, t, r, b),
                        center = Point((l + r) / 2f, (t + b) / 2f),
                        left = left,
                        right = right,
                        top = top,
                        bottom = bottom
                    )
                )
            }
        }
        return blocks
    }

    /** True if the cell is inside the real image area and is NOT part of the walkway mask. */
    private fun isEmptyImageCell(x: Int, y: Int): Boolean {
        if (x < 0 || y < 0 || x >= protoWidth || y >= protoHeight) return false
        if (maskGrid[y][x]) return false
        val px = (x + 0.5f) / protoWidth * inputWidth
        val py = (y + 0.5f) / protoHeight * inputHeight
        return px >= lbPadX && px <= lbPadX + lbNewW && py >= lbPadY && py <= lbPadY + lbNewH
    }

    private fun getOutput0Value(buffer: ByteBuffer, channel: Int, anchor: Int): Float {
        val index = if (isOutput0ChannelsFirst) {
            channel * numAnchors + anchor
        } else {
            anchor * numOutput0Channels + channel
        }
        return readValue(buffer, index, out0Mode, out0Scale, out0ZeroPoint)
    }

    private fun getOutput1Value(buffer: ByteBuffer, channel: Int, y: Int, x: Int): Float {
        val index = if (isOutput1ChannelsFirst) {
            channel * (protoHeight * protoWidth) + y * protoWidth + x
        } else {
            (y * protoWidth + x) * numProtoMasks + channel
        }
        return readValue(buffer, index, out1Mode, out1Scale, out1ZeroPoint)
    }

    @Synchronized
    fun close() {
        interpreter?.close()
        interpreter = null
        letterboxBitmap?.recycle()
        letterboxBitmap = null
        letterboxCanvas = null
    }
}

data class PathResult(
    val boundaries: List<List<Point>>,
    val topScore: Float = 0f,
    val topBox: RectF? = null,
    val debugInfo: String = "",
    /** Mask blocks on the outline of the detected sidewalk; coordinates normalized [0..1] of the original bitmap. */
    val edgeBlocks: List<EdgeBlock> = emptyList(),
    /** Size in pixels of the bitmap that was analysed (needed for [edgeBlocksInView]). */
    val imageWidth: Int = 0,
    val imageHeight: Int = 0
) {
    /** Only the blocks where the sidewalk ends on the left (e.g. the left kerb / grass line). */
    val leftEdgeBlocks: List<EdgeBlock> get() = edgeBlocks.filter { it.left }

    /** Only the blocks where the sidewalk ends on the right. */
    val rightEdgeBlocks: List<EdgeBlock> get() = edgeBlocks.filter { it.right }

    /**
     * [edgeBlocks] converted to view pixels for a preview of size [viewWidth] x [viewHeight].
     *
     * @param fillCenter false = the image is fitted inside the view (letterboxed, PreviewView FIT_*
     * or ImageView fitCenter); true = the image fills the view and is cropped (PreviewView
     * FILL_*, the CameraX default, or ImageView centerCrop).
     */
    fun edgeBlocksInView(viewWidth: Float, viewHeight: Float, fillCenter: Boolean = false): List<EdgeBlock> {
        if (imageWidth <= 0 || imageHeight <= 0 || edgeBlocks.isEmpty()) return emptyList()
        val sx = viewWidth / imageWidth
        val sy = viewHeight / imageHeight
        val s = if (fillCenter) maxOf(sx, sy) else minOf(sx, sy)
        val offX = (viewWidth - imageWidth * s) / 2f
        val offY = (viewHeight - imageHeight * s) / 2f
        fun vx(n: Float) = n * imageWidth * s + offX
        fun vy(n: Float) = n * imageHeight * s + offY
        return edgeBlocks.map {
            it.copy(
                rect = RectF(vx(it.rect.left), vy(it.rect.top), vx(it.rect.right), vy(it.rect.bottom)),
                center = Point(vx(it.center.x), vy(it.center.y))
            )
        }
    }
}

/**
 * One cell of the segmentation mask grid that lies on the outline of the sidewalk.
 * [left]/[right]/[top]/[bottom] say on which side(s) the neighbouring cell is NOT sidewalk.
 * [rect] and [center] are normalized [0..1] of the original bitmap, unless the instance came
 * from [PathResult.edgeBlocksInView], in which case they are in view pixels.
 */
data class EdgeBlock(
    val gridX: Int,
    val gridY: Int,
    val rect: RectF,
    val center: Point,
    val left: Boolean,
    val right: Boolean,
    val top: Boolean,
    val bottom: Boolean
)

data class Point(val x: Float, val y: Float)