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
 *
 * NV21 packing accounts for per-plane rowStride and pixelStride.
 * On most devices the UV planes have pixelStride=2 (semi-planar / interleaved),
 * so naively calling buffer.get() on the raw plane bytes includes the interleaved
 * channel and produces corrupted colour. We extract every other byte explicitly.
 */
class CameraFrameProvider {
    private val _frames = MutableSharedFlow<Bitmap>(extraBufferCapacity = 2)
    val frames: SharedFlow<Bitmap> = _frames

    /**
     * Call from ImageReader.OnImageAvailableListener.
     * Converts YUV_420_888 → correct NV21 → Bitmap and emits.
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
        val width  = image.width
        val height = image.height

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride   // 1 = planar, 2 = semi-planar

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        // NV21 layout: full Y plane (width×height), then interleaved VU pairs (width×height/2 bytes)
        val nv21 = ByteArray(width * height * 3 / 2)

        // --- Y plane: copy row by row to strip any row-stride padding ---
        var yOffset = 0
        for (row in 0 until height) {
            yBuffer.position(row * yRowStride)
            yBuffer.get(nv21, yOffset, width)
            yOffset += width
        }

        // --- UV plane: extract V and U bytes, interleaved as VU in NV21 ---
        var uvOffset = width * height
        for (row in 0 until height / 2) {
            for (col in 0 until width / 2) {
                val vPos = row * uvRowStride + col * uvPixelStride
                val uPos = row * uvRowStride + col * uvPixelStride

                vBuffer.position(vPos)
                uBuffer.position(uPos)

                nv21[uvOffset++] = vBuffer.get()
                nv21[uvOffset++] = uBuffer.get()
            }
        }

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
}
