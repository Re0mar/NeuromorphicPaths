package com.neuromorphicpaths.model

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.ObstacleDetector
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * YOLO-World exported with this project's vocabulary, run through ONNX Runtime on the CPU.
 *
 * The model file is not committed. `model/tools/export_yolo_world.py` writes it into the assets
 * folder, and [isAvailable] says whether that has happened on this build. One frame goes in as
 * a letterboxed square, and the decoder maps the boxes back out.
 */
class OnnxYoloWorldDetector private constructor(
    private val environment: OrtEnvironment,
    private val session: OrtSession,
    private val decoder: YoloDecoder,
    private val inputSize: Int,
    private val dispatcher: CoroutineDispatcher,
) : ObstacleDetector {

    override val name: String = "yolo-world"

    private val inputName: String = session.inputNames.first()

    override suspend fun detect(frame: Frame): List<Detection> = withContext(dispatcher) {
        val letterbox = Letterbox.fit(frame.width, frame.height, inputSize)
        val input = letterboxedTensor(frame, letterbox)
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), longArrayOf(1, CHANNELS.toLong(), inputSize.toLong(), inputSize.toLong())).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val rows = (result.get(0).value as Array<Array<FloatArray>>)[0]
                val anchorCount = rows[0].size
                val flat = FloatArray(rows.size * anchorCount)
                for ((rowIndex, row) in rows.withIndex()) {
                    System.arraycopy(row, 0, flat, rowIndex * anchorCount, anchorCount)
                }
                decoder.decode(flat, anchorCount, letterbox, frame.width, frame.height)
            }
        }
    }

    override fun close() {
        session.close()
    }

    /** Scales the frame into the square, gray around it, and returns it as planar RGB in 0..1. */
    private fun letterboxedTensor(frame: Frame, letterbox: Letterbox): FloatArray {
        val source = Bitmap.createBitmap(frame.width, frame.height, Bitmap.Config.ARGB_8888)
        source.copyPixelsFromBuffer(ByteBuffer.wrap(frame.rgba))
        val square = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(square)
        canvas.drawColor(PADDING_COLOR)
        val target = RectF(
            letterbox.padX.toFloat(),
            letterbox.padY.toFloat(),
            (letterbox.padX + frame.width * letterbox.scale).toFloat(),
            (letterbox.padY + frame.height * letterbox.scale).toFloat(),
        )
        canvas.drawBitmap(source, null, target, Paint(Paint.FILTER_BITMAP_FLAG))
        source.recycle()

        val pixels = IntArray(inputSize * inputSize)
        square.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        square.recycle()
        val planeSize = inputSize * inputSize
        val tensor = FloatArray(CHANNELS * planeSize)
        for (index in 0 until planeSize) {
            val pixel = pixels[index]
            tensor[index] = Color.red(pixel) / MAX_CHANNEL
            tensor[planeSize + index] = Color.green(pixel) / MAX_CHANNEL
            tensor[2 * planeSize + index] = Color.blue(pixel) / MAX_CHANNEL
        }
        return tensor
    }

    companion object {
        const val MODEL_ASSET_PATH = "yolo_world/yolo_world.onnx"
        const val DEFAULT_INPUT_SIZE = 320
        private const val CHANNELS = 3
        private const val MAX_CHANNEL = 255f

        // The gray ultralytics pads with, so the model sees what it was exported against.
        private val PADDING_COLOR = Color.rgb(114, 114, 114)

        /** True when the exported model is bundled in this build. */
        fun isAvailable(context: Context): Boolean = try {
            context.assets.open(MODEL_ASSET_PATH).close()
            true
        } catch (missing: FileNotFoundException) {
            // Expected on a build where nobody ran the export script. Not an error.
            false
        }

        fun load(
            context: Context,
            inputSize: Int = DEFAULT_INPUT_SIZE,
            dispatcher: CoroutineDispatcher = Dispatchers.Default,
        ): OnnxYoloWorldDetector {
            val vocabulary = context.assets.open(YoloWorldVocabulary.ASSET_PATH).bufferedReader().use { reader ->
                YoloWorldVocabulary.parse(reader.readText())
            }
            val modelBytes = context.assets.open(MODEL_ASSET_PATH).use { stream -> stream.readBytes() }
            val environment = OrtEnvironment.getEnvironment()
            val session = environment.createSession(modelBytes, OrtSession.SessionOptions())
            return OnnxYoloWorldDetector(environment, session, YoloDecoder(vocabulary), inputSize, dispatcher)
        }
    }
}
