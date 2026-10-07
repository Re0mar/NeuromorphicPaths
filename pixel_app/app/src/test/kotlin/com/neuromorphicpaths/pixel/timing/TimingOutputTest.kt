package com.neuromorphicpaths.pixel.timing

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/** The file the phone actually writes, in a temporary folder. */
class TimingOutputTest {
    @Test
    fun linesEndWithABareNewline() {
        val directory = Files.createTempDirectory("timing").toFile()
        val output = FileTimingOutput.createIn(directory, "2026-10-04T10:12:03Z")
        output.writeLine("""{"type":"dropped","frame_ns":1}""")
        output.writeLine("""{"type":"dropped","frame_ns":2}""")
        output.close()

        val bytes = directory.listFiles()!!.single().readBytes()
        assertEquals(0, bytes.count { it == '\r'.code.toByte() }, "a carriage return reached the file")
        assertEquals(2, bytes.count { it == '\n'.code.toByte() })
    }

    @Test
    fun theFileIsNamedForTheSessionStartWithoutColons() {
        val directory = Files.createTempDirectory("timing").toFile()
        FileTimingOutput.createIn(directory, "2026-10-04T10:12:03Z").close()

        assertEquals("timing_2026-10-04T10-12-03Z.jsonl", directory.listFiles()!!.single().name)
    }

    @Test
    fun twoSessionsInOneSecondGetSeparateFiles() {
        // Sharing a file would make two sessions read back as one.
        val directory = Files.createTempDirectory("timing").toFile()
        val first = FileTimingOutput.createIn(directory, "2026-10-04T10:12:03Z")
        val second = FileTimingOutput.createIn(directory, "2026-10-04T10:12:03Z")
        first.writeLine("first")
        second.writeLine("second")
        first.close()
        second.close()

        val files = directory.listFiles()!!.sortedBy { it.name }
        assertEquals(2, files.size)
        assertNotEquals(files[0].readText(), files[1].readText())
        assertTrue(files.any { it.name == "timing_2026-10-04T10-12-03Z_2.jsonl" })
    }
}
