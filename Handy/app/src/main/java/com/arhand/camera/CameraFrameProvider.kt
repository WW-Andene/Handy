package com.arhand.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import java.io.ByteArrayOutputStream

/**
 * Converts Camera2 Image (YUV_420_888) to Bitmap for MediaPipe and CLAHE.
 * Emits frames via SharedFlow consumed by HandTracker and CLAHEAnalyzer.
 */
class CameraFrameProvider {
    private val _frames = MutableSharedFlow<Bitmap>(extraBufferCapacity = 2)
    val frames: SharedFlow<Bitmap> = _frames

    /**
     * Call from ImageReader.OnImageAvailableListener.
     * Converts YUV → NV21 → Bitmap and emits.
     */
    fun onImage(image: Image) {
        try {
            val bitmap = yuv420ToBitmap(image)
            _frames.tryEmit(bitmap)
        } finally {
            image.close()
        }
    }

    private fun yuv420ToBitmap(image: Image): Bitmap {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
}
