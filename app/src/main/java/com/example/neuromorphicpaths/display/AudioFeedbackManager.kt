package com.example.neuromorphicpaths.display

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import com.example.neuromorphicpaths.steering.DirectionCommand
import java.util.Locale

class AudioFeedbackManager(context: Context) : TextToSpeech.OnInitListener {
    private var tts: TextToSpeech? = TextToSpeech(context.applicationContext, this)
    private var isInitialized = false

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("AudioFeedbackManager", "Language US is not supported or missing data")
            } else {
                isInitialized = true
                Log.d("AudioFeedbackManager", "TextToSpeech successfully initialized")
            }
        } else {
            Log.e("AudioFeedbackManager", "TextToSpeech initialization failed")
        }
    }

    fun speakCommand(command: DirectionCommand) {
        if (!isInitialized) {
            Log.w("AudioFeedbackManager", "TextToSpeech not initialized yet")
            return
        }
        val text = when (command) {
            DirectionCommand.HARD_LEFT -> "Hard left"
            DirectionCommand.SLIGHT_LEFT -> "Veer left"
            DirectionCommand.HARD_RIGHT -> "Hard right"
            DirectionCommand.SLIGHT_RIGHT -> "Veer right"
            DirectionCommand.STRAIGHT -> "Straight ahead"
        }
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
    }

    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        isInitialized = false
    }
}
