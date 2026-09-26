package com.neuromorphicpaths.model

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.SceneClassMap
import com.neuromorphicpaths.core.SurfaceSegmenter
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * SegFormer-B0 trained on ADE20K, exported by `model/tools/export_segformer.py`, run through ONNX
 * Runtime on the CPU.
 *
 * The frame is squashed into the model's square without letterboxing, which is how the model's
 * own image processor resizes, so the grid that comes back covers the whole frame and maps
 * onto it by plain stretching. The graph already normalizes and takes the argmax, so what comes
 * back is one class index per cell, translated through the class table into a [SceneClassMap].
 * The model file is not committed, and [isAvailable] says whether this build has it.
 */
class OnnxSegFormerSegmenter private constructor(
    private val environment: OrtEnvironment,
    private val session: OrtSession,
    private val classes: Ade20kClasses,
    private val inputSize: Int,
    private val dispatcher: CoroutineDispatcher,
) : SurfaceSegmenter {

    override val name: String = "segformer-b0-ade20k"

    private val inputName: String = session.inputNames.first()

    override suspend fun segment(frame: Frame): SceneClassMap = withContext(dispatcher) {
        val input = squashedTensor(frame)
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), longArrayOf(1, CHANNELS.toLong(), inputSize.toLong(), inputSize.toLong())).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val grid = (result.get(0).value as Array<Array<LongArray>>)[0]
                val height = grid.size
                val width = grid[0].size
                val ordinals = ByteArray(width * height)
                for (row in 0 until height) {
                    val cells = grid[row]
                    for (column in 0 until width) {
                        ordinals[row * width + column] = classes.sceneOrdinalAt(cells[column].toInt())
                    }
                }
                SceneClassMap(width, height, ordinals)
            }
        }
    }

    override fun close() {
        session.close()
    }

    /** Scales the whole frame into the square, aspect ignored, and returns it as planar RGB in 0..1. */
    private fun squashedTensor(frame: Frame): FloatArray {
        val source = Bitmap.createBitmap(frame.width, frame.height, Bitmap.Config.ARGB_8888)
        source.copyPixelsFromBuffer(ByteBuffer.wrap(frame.rgba))
        val square = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(square)
        canvas.drawBitmap(source, null, RectF(0f, 0f, inputSize.toFloat(), inputSize.toFloat()), Paint(Paint.FILTER_BITMAP_FLAG))
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
        const val MODEL_ASSET_PATH = "segformer/segformer_ade20k.onnx"
        private const val INPUT_DIMENSIONS = 4
        private const val CHANNELS = 3
        private const val MAX_CHANNEL = 255f

        /** True when the exported model is bundled in this build. */
        fun isAvailable(context: Context): Boolean = try {
            context.assets.open(MODEL_ASSET_PATH).close()
            true
        } catch (missing: FileNotFoundException) {
            // Expected on a build where nobody ran the export script. Not an error.
            false
        }

        /** The square side the model was exported with, read from its input tensor, as the detector does. */
        private fun inputSizeOf(session: OrtSession): Int {
            val info = session.inputInfo.values.first().info as TensorInfo
            val shape = info.shape
            require(shape.size == INPUT_DIMENSIONS && shape[2] == shape[3] && shape[3] > 0) {
                "Expected a [1, 3, size, size] input, got ${shape.toList()}"
            }
            return shape[3].toInt()
        }

        /**
         * Loads the model and its class table from the assets.
         *
         * With [useXnnpack] the session asks for the XNNPACK provider ahead of the default CPU
         * one, which is the comparison the timing runs need. Ops it does not cover fall back to
         * the CPU provider inside the runtime.
         */
        fun load(
            context: Context,
            useXnnpack: Boolean = false,
            dispatcher: CoroutineDispatcher = Dispatchers.Default,
        ): OnnxSegFormerSegmenter {
            val classes = context.assets.open(Ade20kClasses.ASSET_PATH).bufferedReader().use { reader ->
                Ade20kClasses.parse(reader.readText())
            }
            val modelBytes = context.assets.open(MODEL_ASSET_PATH).use { stream -> stream.readBytes() }
            val environment = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions()
            if (useXnnpack) options.addXnnpack(emptyMap())
            val session = environment.createSession(modelBytes, options)
            return OnnxSegFormerSegmenter(environment, session, classes, inputSizeOf(session), dispatcher)
        }
    }
}
