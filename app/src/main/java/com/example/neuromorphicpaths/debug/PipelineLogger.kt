package com.example.neuromorphicpaths.debug

import android.content.Context
import android.util.Log
import com.example.neuromorphicpaths.pipeline.PipelineFrame
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class PipelineLogger(private val context: Context) {
    private var fileWriter: FileWriter? = null
    private val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)

    fun startLogging() {
        val timestamp = dateFormat.format(Date())
        val fileName = "pipeline_log_$timestamp.csv"
        val file = File(context.getExternalFilesDir(null), fileName)
        
        try {
            fileWriter = FileWriter(file)
            fileWriter?.write("timestampMs,leftOffsetM,rightOffsetM,headingDeviationDeg,command\n")
            Log.d("PipelineLogger", "Logging to ${file.absolutePath}")
        } catch (e: IOException) {
            Log.e("PipelineLogger", "Failed to start logging", e)
        }
    }

    fun logFrame(frame: PipelineFrame) {
        val writer = fileWriter ?: return
        val pos = frame.sidewalkPosition
        val steering = frame.steeringResult
        val cmd = frame.stabilizedCommand
        
        val line = String.format(
            Locale.US,
            "%d,%.3f,%.3f,%.3f,%s\n",
            frame.timestampMs,
            pos?.leftEdgeOffsetM ?: 0.0,
            pos?.rightEdgeOffsetM ?: 0.0,
            steering?.recommendedHeadingDeg ?: 0.0, // using recommended heading as deviation for now if that's what's intended
            cmd?.name ?: "NONE"
        )
        
        try {
            writer.write(line)
        } catch (e: IOException) {
            Log.e("PipelineLogger", "Failed to write log line", e)
        }
    }

    fun stopLogging() {
        try {
            fileWriter?.close()
        } catch (e: IOException) {
            Log.e("PipelineLogger", "Failed to close log file", e)
        }
        fileWriter = null
    }
}
