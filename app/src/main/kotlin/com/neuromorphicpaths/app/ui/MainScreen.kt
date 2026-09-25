package com.neuromorphicpaths.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.neuromorphicpaths.output.GuidanceOverlay
import com.neuromorphicpaths.output.ScreenOverlayDisplay

/** A row of controls over the overlay. Everything the first version lets a person do. */
@Composable
fun MainScreen(
    display: ScreenOverlayDisplay,
    scriptedDetections: Boolean,
    onScriptedDetectionsChange: (Boolean) -> Unit,
    onStartCamera: () -> Unit,
    onOpenVideo: () -> Unit,
) {
    Column(Modifier.fillMaxSize()) {
        Row(
            Modifier
                .fillMaxWidth()
                .padding(8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Button(onClick = onStartCamera) { Text("Camera") }
            Button(onClick = onOpenVideo) { Text("Open video") }
            Spacer(Modifier.weight(1f))
            Text("Scripted")
            Switch(checked = scriptedDetections, onCheckedChange = onScriptedDetectionsChange)
        }
        GuidanceOverlay(
            display = display,
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
        )
    }
}
