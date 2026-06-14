package com.arhand.camera

import android.hardware.camera2.CameraManager

/**
 * Controls camera torch via CameraManager.setTorchMode().
 * Direct, instantaneous — replaces the fragile applyConstraints({torch}) approach from the HTML.
 */
class TorchController(private val cameraManager: CameraManager) {
    private var backCameraId: String? = null

    fun init() {
        // Find rear camera
        backCameraId = cameraManager.cameraIdList.firstOrNull { id ->
            val chars = cameraManager.getCameraCharacteristics(id)
            val facing = chars.get(android.hardware.camera2.CameraCharacteristics.LENS_FACING)
            facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_BACK
        }
    }

    fun setTorch(enabled: Boolean) {
        val id = backCameraId ?: return
        try {
            cameraManager.setTorchMode(id, enabled)
        } catch (e: Exception) {
            // Torch may not be available on all devices
        }
    }

    fun toggle(current: Boolean): Boolean {
        val next = !current
        setTorch(next)
        return next
    }
}
