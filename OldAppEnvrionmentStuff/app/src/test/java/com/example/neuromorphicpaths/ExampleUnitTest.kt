package com.example.neuromorphicpaths

import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ExampleUnitTest {
    @Test
    fun parseTfliteDetails() {
        val file = File("src/main/assets/best_int8.tflite")
        val buf = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)

        // FlatBuffer root
        val rootPos = buf.getInt(0)
        
        fun getVtableOffset(tablePos: Int, fieldIndex: Int): Int {
            val vtablePos = tablePos - buf.getInt(tablePos)
            val vtableSize = buf.getShort(vtablePos).toInt() and 0xFFFF
            val fieldOffsetInVtable = 4 + fieldIndex * 2
            if (fieldOffsetInVtable >= vtableSize) return 0
            return buf.getShort(vtablePos + fieldOffsetInVtable).toInt() and 0xFFFF
        }

        // Subgraphs offset in Model (field 2 of Model)
        val subgraphsVtableOff = getVtableOffset(rootPos, 2)
        if (subgraphsVtableOff == 0) {
            println("No subgraphs field")
            return
        }
        val subgraphsVectorPos = rootPos + subgraphsVtableOff + buf.getInt(rootPos + subgraphsVtableOff)
        val subgraphsCount = buf.getInt(subgraphsVectorPos)
        println("Subgraphs count: $subgraphsCount")

        val subgraph0Pos = subgraphsVectorPos + 4 + buf.getInt(subgraphsVectorPos + 4)

        // Field 0 in SubGraph: tensors vector
        val tensorsVtableOff = getVtableOffset(subgraph0Pos, 0)
        val tensorsVectorPos = subgraph0Pos + tensorsVtableOff + buf.getInt(subgraph0Pos + tensorsVtableOff)
        val tensorsCount = buf.getInt(tensorsVectorPos)
        println("Tensors count: $tensorsCount")

        // Inputs & Outputs
        fun readIntVector(fieldIndex: Int): List<Int> {
            val off = getVtableOffset(subgraph0Pos, fieldIndex)
            if (off == 0) return emptyList()
            val vecPos = subgraph0Pos + off + buf.getInt(subgraph0Pos + off)
            val len = buf.getInt(vecPos)
            return (0 until len).map { buf.getInt(vecPos + 4 + it * 4) }
        }

        val inputs = readIntVector(1)
        val outputs = readIntVector(2)
        println("Inputs tensor indices: $inputs")
        println("Outputs tensor indices: $outputs")

        val typeNames = mapOf(
            0 to "FLOAT32", 1 to "FLOAT16", 2 to "INT32", 3 to "UINT8",
            4 to "INT64", 5 to "STRING", 6 to "BOOL", 7 to "INT16",
            8 to "COMPLEX64", 9 to "INT8"
        )

        fun printTensor(index: Int, label: String) {
            val tensorPos = tensorsVectorPos + 4 + index * 4 + buf.getInt(tensorsVectorPos + 4 + index * 4)

            // Tensor fields:
            // 0: shape (vector of int32)
            // 1: type (int8/enum TensorType)
            // 2: buffer (uint32)
            // 3: name (string)
            // 4: quantization (QuantizationParameters table)

            // Shape
            val shapeOff = getVtableOffset(tensorPos, 0)
            val shape = if (shapeOff != 0) {
                val vecPos = tensorPos + shapeOff + buf.getInt(tensorPos + shapeOff)
                val len = buf.getInt(vecPos)
                (0 until len).map { buf.getInt(vecPos + 4 + it * 4) }
            } else emptyList()

            // Type
            val typeOff = getVtableOffset(tensorPos, 1)
            val typeVal = if (typeOff != 0) buf.get(tensorPos + typeOff).toInt() else 0

            // Name
            val nameOff = getVtableOffset(tensorPos, 3)
            val name = if (nameOff != 0) {
                val strPos = tensorPos + nameOff + buf.getInt(tensorPos + nameOff)
                val len = buf.getInt(strPos)
                val bytes = ByteArray(len)
                val oldPos = buf.position()
                buf.position(strPos + 4)
                buf.get(bytes)
                buf.position(oldPos)
                String(bytes, Charsets.UTF_8)
            } else "unnamed"

            // Quantization
            var quantStr = "none"
            val quantOff = getVtableOffset(tensorPos, 4)
            if (quantOff != 0) {
                val quantPos = tensorPos + quantOff + buf.getInt(tensorPos + quantOff)
                // Field 0: min, 1: max, 2: scale (vector of float), 3: zero_point (vector of int64)
                val scaleOff = getVtableOffset(quantPos, 2)
                val zpOff = getVtableOffset(quantPos, 3)

                val scales = if (scaleOff != 0) {
                    val vecPos = quantPos + scaleOff + buf.getInt(quantPos + scaleOff)
                    val len = buf.getInt(vecPos)
                    (0 until len).map { buf.getFloat(vecPos + 4 + it * 4) }
                } else emptyList()

                val zps = if (zpOff != 0) {
                    val vecPos = quantPos + zpOff + buf.getInt(quantPos + zpOff)
                    val len = buf.getInt(vecPos)
                    (0 until len).map { buf.getLong(vecPos + 4 + it * 8) }
                } else emptyList()

                quantStr = "scales=$scales, zero_points=$zps"
            }

            println("Tensor [$index] ($label): name='$name', type=${typeNames[typeVal] ?: typeVal}, shape=$shape, quant=$quantStr")
        }

        inputs.forEach { printTensor(it, "INPUT") }
        outputs.forEachIndexed { i, idx -> printTensor(idx, "OUTPUT $i") }
    }
}


