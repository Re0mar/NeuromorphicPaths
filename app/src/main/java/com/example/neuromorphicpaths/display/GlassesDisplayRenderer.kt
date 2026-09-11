package com.example.neuromorphicpaths.display

import android.util.Log
import com.example.neuromorphicpaths.steering.DirectionCommand
import com.meta.wearable.dat.core.session.DeviceSession
import com.meta.wearable.dat.display.Display
import com.meta.wearable.dat.display.addDisplay
import com.meta.wearable.dat.display.types.DisplayConfiguration
import com.meta.wearable.dat.display.views.Direction
import com.meta.wearable.dat.display.views.IconName
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class GlassesDisplayRenderer(
    private val scope: CoroutineScope,
    session: DeviceSession
) {
    private var display: Display? = null

    init {
        session.addDisplay(DisplayConfiguration()).fold(
            onSuccess = { addedDisplay ->
                display = addedDisplay
                Log.d("GlassesDisplayRenderer", "Display successfully added to session")
            },
            onFailure = { error, _ ->
                Log.e("GlassesDisplayRenderer", "Failed to add display: ${error.description}")
            }
        )
    }

    fun renderCommand(command: DirectionCommand) {
        val currentDisplay = display ?: return
        scope.launch(Dispatchers.IO) {
            val iconName = when (command) {
                DirectionCommand.HARD_LEFT, DirectionCommand.SLIGHT_LEFT -> IconName.ARROW_LEFT
                DirectionCommand.HARD_RIGHT, DirectionCommand.SLIGHT_RIGHT -> IconName.ARROW_RIGHT
                DirectionCommand.STRAIGHT -> IconName.ARROW_UP_SHALLOW_U
            }
            val textContent = when (command) {
                DirectionCommand.HARD_LEFT -> "Hard Left"
                DirectionCommand.SLIGHT_LEFT -> "Slight Left"
                DirectionCommand.HARD_RIGHT -> "Hard Right"
                DirectionCommand.SLIGHT_RIGHT -> "Slight Right"
                DirectionCommand.STRAIGHT -> "Straight Ahead"
            }

            currentDisplay.sendContent {
                flexBox(direction = Direction.COLUMN) {
                    icon(name = iconName)
                    text(content = textContent)
                }
            }
        }
    }

    fun stop() {
        display?.stop()
        display = null
    }
}
