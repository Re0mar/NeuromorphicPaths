package com.example.sidewalkvision

import android.graphics.Bitmap
import android.media.ExifInterface
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

// A typical phone main camera. The photo is 4:3, like a phone's.
private const val EQUIVALENT_FOCAL_LENGTH_MM = 26
private const val PHOTO_WIDTH = 400
private const val PHOTO_HEIGHT = 300

@RunWith(AndroidJUnit4::class)
class PhotoIntrinsicsTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    private fun writePhoto(name: String, equivalentFocalLengthMm: Int?): File {
        val file = File(context.cacheDir, name)
        file.outputStream().use { stream ->
            Bitmap.createBitmap(PHOTO_WIDTH, PHOTO_HEIGHT, Bitmap.Config.ARGB_8888)
                .compress(Bitmap.CompressFormat.JPEG, 90, stream)
        }
        if (equivalentFocalLengthMm != null) {
            ExifInterface(file.absolutePath).apply {
                setAttribute(ExifInterface.TAG_FOCAL_LENGTH_IN_35MM_FILM, equivalentFocalLengthMm.toString())
                saveAttributes()
            }
        }
        return file
    }

    @Test
    fun theFocalLengthIsReadFromThePhotosExif() {
        val photo = writePhoto("with_exif.jpg", EQUIVALENT_FOCAL_LENGTH_MM)

        val intrinsics = photoIntrinsics(context.contentResolver, Uri.fromFile(photo), PHOTO_WIDTH, PHOTO_HEIGHT)
        photo.delete()

        assertNotNull(intrinsics)
        // A 400 x 300 image has a 500 px diagonal.
        assertEquals(26.0 * 500 / 43.27, intrinsics!!.focalLengthPx(PHOTO_WIDTH, PHOTO_HEIGHT), 1e-6)
    }

    @Test
    fun aPhotoWithoutTheTagGetsNoIntrinsics() {
        val photo = writePhoto("without_exif.jpg", equivalentFocalLengthMm = null)

        val intrinsics = photoIntrinsics(context.contentResolver, Uri.fromFile(photo), PHOTO_WIDTH, PHOTO_HEIGHT)
        photo.delete()

        assertNull(intrinsics)
    }
}
