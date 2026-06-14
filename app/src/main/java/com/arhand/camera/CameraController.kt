package com.arhand.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.WindowManager

/**
 * Camera2 pipeline.
 * Opens CameraDevice → creates CaptureSession → streams YUV_420_888 to ImageReader.
 * Mirrors getUserMedia + video stream logic from HTML prototype.
 */
class CameraController(
    private val context: Context,
    private val frameProvider: CameraFrameProvider
) {
    companion object {
        const val TARGET_WIDTH  = 640
        const val TARGET_HEIGHT = 480
        const val TARGET_FPS    = 30
    }

    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    val torchController = TorchController(cameraManager)

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null

    private var currentFacing = CameraCharacteristics.LENS_FACING_FRONT
    private var sensorOrientation = 0

    @SuppressLint("MissingPermission")
    fun start(facing: Int = CameraCharacteristics.LENS_FACING_FRONT) {
        currentFacing = facing
        torchController.init()

        cameraThread = HandlerThread("CameraThread").also { it.start() }
        cameraHandler = Handler(cameraThread!!.looper)

        val cameraId = cameraManager.cameraIdList.first { id ->
            val chars = cameraManager.getCameraCharacteristics(id)
            chars.get(CameraCharacteristics.LENS_FACING) == facing
        }
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)
        sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0

        // Display rotation compensates for the device's current physical orientation.
        // SENSOR_ORIENTATION alone is only correct when the device is in its natural
        // (portrait) orientation. When the user holds the device landscape or upside-down,
        // the frame arrives pre-rotated by the sensor but the display has also rotated —
        // we must add both angles so MediaPipe always receives an upright image.
        @Suppress("DEPRECATION")
        val displayRotationDeg = when (
            (context.getSystemService(Context.WINDOW_SERVICE) as WindowManager)
                .defaultDisplay.rotation
        ) {
            Surface.ROTATION_90  -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else                 -> 0
        }
        frameProvider.rotationDegrees = (sensorOrientation + displayRotationDeg) % 360

        imageReader = ImageReader.newInstance(TARGET_WIDTH, TARGET_HEIGHT, ImageFormat.YUV_420_888, 4)
        imageReader!!.setOnImageAvailableListener({ reader ->
            val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
            frameProvider.onImage(image)
        }, cameraHandler)

        cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                createCaptureSession()
            }
            override fun onDisconnected(camera: CameraDevice) { camera.close() }
            override fun onError(camera: CameraDevice, error: Int) { camera.close() }
        }, cameraHandler)
    }

    private fun createCaptureSession() {
        val surface = imageReader!!.surface
        cameraDevice!!.createCaptureSession(
            listOf(surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    val request = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                        addTarget(surface)
                        set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                        set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, android.util.Range(TARGET_FPS, TARGET_FPS))
                    }.build()
                    session.setRepeatingRequest(request, null, cameraHandler)
                }
                override fun onConfigureFailed(session: CameraCaptureSession) {}
            },
            cameraHandler
        )
    }

    fun switchCamera() {
        stop()
        val newFacing = if (currentFacing == CameraCharacteristics.LENS_FACING_FRONT)
            CameraCharacteristics.LENS_FACING_BACK
        else
            CameraCharacteristics.LENS_FACING_FRONT
        start(newFacing)
    }

    fun isFrontFacing() = currentFacing == CameraCharacteristics.LENS_FACING_FRONT

    fun stop() {
        captureSession?.close()
        cameraDevice?.close()
        imageReader?.close()
        cameraThread?.quitSafely()
        captureSession = null
        cameraDevice = null
        imageReader = null
        cameraThread = null
        cameraHandler = null
    }
}
