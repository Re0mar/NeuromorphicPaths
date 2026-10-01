/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.displayaccess.ui

import androidx.activity.compose.LocalActivity
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.outlined.RemoveRedEye
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Snackbar
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.meta.wearable.dat.display.types.DisplayState
import com.example.neuromorphicpaths.BuildConfig
import com.example.neuromorphicpaths.R
import com.meta.wearable.dat.core.types.Permission
import com.meta.wearable.dat.core.types.PermissionStatus
import com.meta.wearable.dat.externalsampleapps.displayaccess.SampleApp
import com.meta.wearable.dat.externalsampleapps.displayaccess.display.DisplayViewModel
import com.example.neuromorphicpaths.wearables.WearablesViewModel
import com.example.neuromorphicpaths.ui.HomeScreen
import com.example.neuromorphicpaths.ui.CameraScreen
import com.example.neuromorphicpaths.ui.MockDeviceKitScreen
import com.example.neuromorphicpaths.ui.AppColor
import com.example.neuromorphicpaths.ui.SamplePlaceholderScreen
import com.example.neuromorphicpaths.ui.SamplesListScreen

object Routes {
  const val CAMERA = "camera"
  const val CONNECT = "connect"
  const val SAMPLES_LIST = "samples_list"
}

private val BackgroundColor = Color(0xFFF2F2F7)
private val TabContainerColor = Color(0xFFF6F6FA)
private val ActiveTabColor = Color(0xFFE7E7ED)
private val ActiveBlue = Color(0xFF3478F6)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppScaffold(
    wearablesViewModel: WearablesViewModel,
    onRequestWearablesPermission: suspend (Permission) -> PermissionStatus,
    onRequestRecordAudioPermission: suspend () -> Boolean,
    modifier: Modifier = Modifier,
) {
  val navController = rememberNavController()
  val wearablesState by wearablesViewModel.uiState.collectAsStateWithLifecycle()
  val displayViewModel: DisplayViewModel = viewModel()
  val displayState by displayViewModel.uiState.collectAsStateWithLifecycle()
  val isCapabilityReady = displayState.displayState == DisplayState.STARTED
  val lastKnownSessionActive = remember { mutableStateOf(displayState.isSessionActive) }
  val activity = LocalActivity.current
  val backStackEntry by navController.currentBackStackEntryAsState()
  val currentRoute = backStackEntry?.destination?.route ?: Routes.CONNECT

  val snackbarHostState = remember { SnackbarHostState() }
  val bottomSheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

  // Observe recent errors and show snackbar
  LaunchedEffect(wearablesState.recentError?.id) {
    wearablesState.recentError?.let { error ->
      snackbarHostState.showSnackbar(error.message)
      wearablesViewModel.clearRecentError(error.id)
    }
  }

  fun openCamera() {
    navController.navigate(Routes.CAMERA) {
      popUpTo(Routes.CONNECT)
      launchSingleTop = true
    }
  }

  fun openSamples() {
    navController.navigate(Routes.SAMPLES_LIST) {
      popUpTo(Routes.CONNECT)
      launchSingleTop = true
    }
  }

  fun openSettings() {
    navController.navigate(Routes.CONNECT) {
      popUpTo(Routes.CONNECT)
      launchSingleTop = true
    }
  }

  DisposableEffect(displayState.isSessionActive) {
    val sessionJustStarted = !lastKnownSessionActive.value && displayState.isSessionActive
    lastKnownSessionActive.value = displayState.isSessionActive
    if (sessionJustStarted && currentRoute == Routes.CONNECT) {
      openSamples()
    }
    onDispose {}
  }

  Scaffold(
      modifier = modifier,
      containerColor = BackgroundColor,
      floatingActionButton = {
        if (BuildConfig.DEBUG) {
          FloatingActionButton(
              onClick = { wearablesViewModel.showDebugMenu() },
              containerColor = AppColor.DeepBlue,
              contentColor = Color.White,
          ) {
            Icon(
                Icons.Default.BugReport,
                contentDescription = stringResource(R.string.debug_menu_description),
            )
          }
        }
      },
      snackbarHost = {
        SnackbarHost(
            hostState = snackbarHostState,
            modifier =
                Modifier.navigationBarsPadding()
                    .padding(horizontal = 16.dp, vertical = 32.dp),
            snackbar = { data ->
              Snackbar(
                  shape = RoundedCornerShape(24.dp),
                  containerColor = MaterialTheme.colorScheme.errorContainer,
                  contentColor = MaterialTheme.colorScheme.onErrorContainer,
              ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                  Icon(
                      imageVector = Icons.Default.Error,
                      contentDescription = stringResource(R.string.scaffold_error_icon_description),
                      tint = MaterialTheme.colorScheme.error,
                  )
                  Spacer(modifier = Modifier.width(8.dp))
                  Text(data.visuals.message)
                }
              }
            },
        )
      },
      bottomBar = {
        BottomTabBar(
            currentRoute = currentRoute,
            samplesEnabled = true,
            onOpenCamera = ::openCamera,
            onOpenSamples = ::openSamples,
            onOpenSettings = ::openSettings,
        )
      },
  ) { contentPadding ->
    Box(modifier = Modifier.fillMaxSize().padding(contentPadding)) {
      NavHost(
          navController = navController,
          startDestination = Routes.CONNECT,
          modifier = Modifier.fillMaxSize(),
      ) {
        composable(Routes.CAMERA) {
          if (wearablesState.isRegistered) {
            CameraScreen(
                wearablesViewModel = wearablesViewModel,
                onRequestWearablesPermission = onRequestWearablesPermission,
                onRequestRecordAudioPermission = onRequestRecordAudioPermission,
            )
          } else {
            HomeScreen(
                viewModel = wearablesViewModel,
            )
          }
        }

        composable(Routes.CONNECT) {
          ConnectScreen(
              uiState = wearablesState,
              selectedDisplayDeviceId = displayState.selectedDeviceId,
              isDisplayReady = isCapabilityReady,
              isPreparingDisplay = displayState.isPreparingDisplay,
              isDatAppUpdateRequired = displayState.isDatAppUpdateRequired,
              onRegister = { activity?.let { wearablesViewModel.startRegistration(it) } },
              onUnregister = {
                displayViewModel.stopSession()
                activity?.let { wearablesViewModel.startUnregistration(it) }
              },
              onOpenFirmwareUpdate = { activity?.let { wearablesViewModel.openFirmwareUpdate(it) } },
              onOpenDatAppUpdate = {
                activity?.let { wearablesViewModel.openDATGlassesAppUpdate(it) }
              },
              onSelectDevice = { deviceId -> displayViewModel.prepareDisplayConnection(deviceId) },
          )
        }

        composable(Routes.SAMPLES_LIST) {
            SamplesListScreen(
                isTryItEnabled = displayState.isSessionActive && isCapabilityReady,
                onSampleSelected = displayViewModel::sendSampleToDisplay,
            )
        }

        SampleApp.entries.forEach { sample ->
          composable(sample.route) {
              SamplePlaceholderScreen(
                  sample = sample,
                  onBack = { navController.popBackStack() },
              )
          }
        }
      }

      if (BuildConfig.DEBUG && wearablesState.isDebugMenuVisible) {
        ModalBottomSheet(
            onDismissRequest = { wearablesViewModel.hideDebugMenu() },
            sheetState = bottomSheetState,
            modifier = Modifier.fillMaxSize(),
        ) {
          MockDeviceKitScreen(modifier = Modifier.fillMaxSize())
        }
      }
    }
  }
}

@Composable
private fun BottomTabBar(
    currentRoute: String,
    samplesEnabled: Boolean,
    onOpenCamera: () -> Unit,
    onOpenSamples: () -> Unit,
    onOpenSettings: () -> Unit,
) {
  Row(
      modifier =
          Modifier.fillMaxWidth()
              .navigationBarsPadding()
              .padding(horizontal = 32.dp, vertical = 18.dp)
              .clip(RoundedCornerShape(999.dp))
              .background(TabContainerColor)
              .padding(8.dp),
      horizontalArrangement = Arrangement.spacedBy(8.dp),
  ) {
    BottomTab(
        label = "Camera",
        icon = {
          Icon(
              imageVector = Icons.Filled.PhotoCamera,
              contentDescription = null,
              modifier = Modifier.size(26.dp),
          )
        },
        selected = currentRoute == Routes.CAMERA,
        enabled = true,
        activeContentColor = Color.Black,
        inactiveContentColor = Color.Black.copy(alpha = 0.35f),
        onClick = onOpenCamera,
        modifier = Modifier.weight(1f),
    )
    BottomTab(
        label = stringResource(R.string.samples_tab),
        icon = {
          Icon(
              imageVector = Icons.Outlined.RemoveRedEye,
              contentDescription = null,
              modifier = Modifier.size(26.dp),
          )
        },
        selected = currentRoute == Routes.SAMPLES_LIST,
        enabled = samplesEnabled,
        activeContentColor = Color.Black,
        inactiveContentColor = Color.Black.copy(alpha = 0.35f),
        onClick = onOpenSamples,
        modifier = Modifier.weight(1f),
    )
    BottomTab(
        label = stringResource(R.string.settings_tab),
        icon = {
          Icon(
              imageVector = Icons.Filled.Settings,
              contentDescription = null,
              modifier = Modifier.size(28.dp),
          )
        },
        selected = currentRoute == Routes.CONNECT,
        enabled = true,
        activeContentColor = ActiveBlue,
        inactiveContentColor = Color.Black.copy(alpha = 0.35f),
        onClick = onOpenSettings,
        modifier = Modifier.weight(1f),
    )
  }
}

@Composable
private fun BottomTab(
    label: String,
    icon: @Composable () -> Unit,
    selected: Boolean,
    enabled: Boolean,
    activeContentColor: Color,
    inactiveContentColor: Color,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
  Column(
      modifier =
          modifier
              .clip(CircleShape)
              .background(if (selected) ActiveTabColor else Color.Transparent)
              .clickable(enabled = enabled, onClick = onClick)
              .padding(vertical = 14.dp),
      horizontalAlignment = Alignment.CenterHorizontally,
  ) {
    val contentColor =
        when {
          !enabled -> inactiveContentColor
          selected -> activeContentColor
          else -> inactiveContentColor
        }
    Row(verticalAlignment = Alignment.CenterVertically) {
        CompositionLocalProvider(
            LocalContentColor provides contentColor,
        ) {
          icon()
        }
    }
    Text(
        text = label,
        color = contentColor,
        style = MaterialTheme.typography.titleMedium,
        fontWeight = FontWeight.Medium,
        modifier = Modifier.padding(top = 6.dp),
    )
  }
}
