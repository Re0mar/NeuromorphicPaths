package com.neuromorphicpaths.pixel.timing

import android.util.Log
import java.io.BufferedWriter
import java.io.File
import java.io.IOException

/** Where a timing log's lines end up. A file on the phone, a buffer in the tests. */
interface TimingOutput {
    /** One line, without its newline. */
    fun writeLine(line: String)

    fun flush()

    fun close()
}

/** Lines to a UTF-8 file, each ended with a bare `\n` whatever the platform. */
class FileTimingOutput(file: File) : TimingOutput {
    private val writer: BufferedWriter = file.bufferedWriter(Charsets.UTF_8)

    override fun writeLine(line: String) {
        writer.write(line)
        writer.write("\n")
    }

    override fun flush() = writer.flush()

    override fun close() {
        try {
            writer.close()
        } catch (closing: IOException) {
            // Nothing left to do with a file that will not close. The lines already flushed are on disk.
            Log.w(TAG, "timing log did not close cleanly", closing)
        }
    }

    companion object {
        private const val TAG = "FileTimingOutput"

        /**
         * A new file in [directory] named for the session's start, never an existing one. Two
         * sessions inside one second get a numbered second file rather than sharing a log, because
         * a log holding two sessions reads as one and isn't.
         */
        fun createIn(directory: File, startedWall: String): FileTimingOutput {
            directory.mkdirs()
            val stem = "timing_" + startedWall.replace(':', '-')
            var candidate = File(directory, "$stem.jsonl")
            var attempt = 2
            while (!candidate.createNewFile()) {
                candidate = File(directory, "${stem}_$attempt.jsonl")
                attempt += 1
            }
            return FileTimingOutput(candidate)
        }
    }
}
