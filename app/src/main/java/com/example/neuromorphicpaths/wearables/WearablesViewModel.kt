/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// WearablesViewModel - Core DAT SDK Integration
//
// This ViewModel demonstrates the core DAT API patterns for:
// - Device registration and unregistration using the DAT SDK
// - Permission management for wearable devices
// - Device discovery and state management
// - Integration with MockDeviceKit for testing

package com.example.neuromorphicpaths.wearables

import android.app.Activity
import android.app.Application
import android.util.Log
import android.widget.Toast
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.selectors.AutoDeviceSelector
import com.meta.wearable.dat.core.selectors.DeviceSelector
import com.meta.wearable.dat.core.types.DeviceCompatibility
import com.meta.wearable.dat.core.types.DeviceIdentifier
import com.example.neuromorphicpaths.R
import com.meta.wearable.dat.core.types.RegistrationState
import com.meta.wearable.dat.externalsampleapps.displayaccess.wearables.WearablesRepository
import kotlinx.collections.immutable.toImmutableList
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class WearablesViewModel(private val application: Application) : AndroidViewModel(application) {

  private companion object {
    private const val TAG = "DisplaySampleWearablesVM"
  }

  private val repository: WearablesRepository = WearablesRepository.getInstance(application)

  private val _uiState = MutableStateFlow(WearablesUiState())
  val uiState: StateFlow<WearablesUiState> = _uiState.asStateFlow()

  // AutoDeviceSelector automatically selects the first available wearable device.
  val deviceSelector: DeviceSelector by lazy { AutoDeviceSelector() }
  private var deviceSelectorJob: Job? = null

  private var monitoringStarted = false
  private var recentErrorId = 0L
  private val deviceMonitoringJobs = mutableMapOf<DeviceIdentifier, Job>()
  private val deviceCompatibility = mutableMapOf<DeviceIdentifier, DeviceCompatibility>()

  private val collectionExceptionHandler = CoroutineExceptionHandler { _, throwable ->
    Log.e(TAG, "Wearables state observation failed", throwable)
  }
  private var observingStarted = false

  private fun startMonitoring() {
    if (monitoringStarted) {
      return
    }
    monitoringStarted = true

    // Monitor device selector for active device
    deviceSelectorJob = viewModelScope.launch {
      deviceSelector.activeDeviceFlow().collect { device ->
        _uiState.update { it.copy(hasActiveDevice = device != null) }
      }
    }

    // This allows the app to react to registration changes (registered, unregistered, etc.)
    viewModelScope.launch(collectionExceptionHandler) {
      Wearables.registrationState.collect { value ->
        _uiState.update { it.copy(registrationState = value) }
      }
    }
    // This automatically updates when devices are discovered, connected, or disconnected
    viewModelScope.launch(collectionExceptionHandler) {
      Wearables.devices.collect { value ->
        _uiState.update { it.copy(devices = value.toList().toImmutableList()) }
        // Monitor device metadata for compatibility issues
        monitorDeviceCompatibility(value)
      }
    }
  }

  private fun monitorDeviceCompatibility(devices: Set<DeviceIdentifier>) {
    // Cancel monitoring jobs for devices that are no longer in the list
    val removedDevices = deviceMonitoringJobs.keys - devices
    removedDevices.forEach { deviceId ->
      deviceMonitoringJobs[deviceId]?.cancel()
      deviceMonitoringJobs.remove(deviceId)
      deviceCompatibility.remove(deviceId)
    }
    updateFirmwareUpdateRequired()

    // Start monitoring jobs only for new devices (not already being monitored)
    val newDevices = devices - deviceMonitoringJobs.keys
    newDevices.forEach { deviceId ->
      val job = viewModelScope.launch(collectionExceptionHandler) {
        Wearables.devicesMetadata[deviceId]?.collect { metadata ->
          deviceCompatibility[deviceId] = metadata.compatibility
          updateFirmwareUpdateRequired()
          if (metadata.compatibility == DeviceCompatibility.DEVICE_UPDATE_REQUIRED) {
            val deviceName = metadata.name.ifEmpty { deviceId }
            setRecentError(application.getString(R.string.error_device_update_required, deviceName))
          }
        }
      }
      deviceMonitoringJobs[deviceId] = job
    }
  }

  fun startRegistration(activity: Activity) {
    if (repository.registrationState.value == RegistrationState.REGISTERED) {
      Toast.makeText(getApplication(), "Already connected", Toast.LENGTH_SHORT).show()
      return
    }
    repository.startRegistration(activity)
  }

  fun startUnregistration(activity: Activity) {
    if (repository.registrationState.value != RegistrationState.REGISTERED) {
      Toast.makeText(getApplication(), "Not connected", Toast.LENGTH_SHORT).show()
      return
    }
    repository.startUnregistration(activity)
  }

  fun openFirmwareUpdate(activity: Activity) {
    Wearables.openFirmwareUpdate(activity).onFailure { error, _ ->
      setRecentError(error.description)
    }
  }

  fun openDATGlassesAppUpdate(activity: Activity) {
    Wearables.openDATGlassesAppUpdate(activity).onFailure { error, _ ->
      Toast.makeText(getApplication(), error.description, Toast.LENGTH_SHORT).show()
    }
  }

  fun showDebugMenu() {
    _uiState.update { it.copy(isDebugMenuVisible = true) }
  }

  fun hideDebugMenu() {
    _uiState.update { it.copy(isDebugMenuVisible = false) }
  }

  fun clearRecentError(errorId: Long) {
    _uiState.update { state ->
      if (state.recentError?.id == errorId) state.copy(recentError = null) else state
    }
  }

  internal fun setRecentError(error: String) {
    recentErrorId += 1
    _uiState.update { it.copy(recentError = RecentError(recentErrorId, error)) }
  }

  fun onPermissionsResult(permissionsResult: Map<String, Boolean>, onAllGranted: () -> Unit) {
    val granted = permissionsResult.entries.all { it.value }
    _uiState.update { it.copy(canRegister = granted) }
    if (granted) {
      onAllGranted()
      startMonitoring()
      startObserving()
    } else {
      setRecentError(application.getString(R.string.error_permissions_required))
    }
  }

  override fun onCleared() {
    super.onCleared()
    // Cancel all device monitoring jobs when ViewModel is cleared
    deviceMonitoringJobs.values.forEach { it.cancel() }
    deviceMonitoringJobs.clear()
    deviceSelectorJob?.cancel()
  }

  private fun updateFirmwareUpdateRequired() {
    val isRequired =
        deviceCompatibility.values.any { it == DeviceCompatibility.DEVICE_UPDATE_REQUIRED }
    _uiState.update { it.copy(isFirmwareUpdateRequired = isRequired) }
  }

  fun startObserving() {
    if (observingStarted) return
    observingStarted = true

    repository.startMonitoring()

    viewModelScope.launch(collectionExceptionHandler) {
      repository.registrationState.collect { state ->
        _uiState.update { it.copy(registrationState = state) }
      }
    }
    viewModelScope.launch(collectionExceptionHandler) {
      repository.devices.collect { devices -> 
        _uiState.update { it.copy(devices = devices.toList().toImmutableList()) } 
      }
    }
    viewModelScope.launch(collectionExceptionHandler) {
      repository.devicesMetadata.collect { metadata ->
        _uiState.update {
          it.copy(
            devicesMetadata = metadata,
            isFirmwareUpdateRequired =
              metadata.values.any { device ->
                device.compatibility == DeviceCompatibility.DEVICE_UPDATE_REQUIRED
              },
          )
        }
      }
    }
  }
}
