package com.arhand.scanner

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.min

/**
 * CLAHE (Contrast Limited Adaptive Histogram Equalization) — full port from HTML prototype.
 *
 * Applies tile-based equalization with bilinear interpolation between tile CLUTs.
 * Returns a contrast score [0,1]; the enhanced bitmap is never materialised.
 *
 * This version processes luminance only (Y from YUV), leaves Cr/Cb unchanged.
 * Runs on a downscaled version of the frame for performance (max 160×120).
 */
class CLAHEAnalyzer(
    private val tilesX: Int = 8,
    private val tilesY: Int = 8,
    private val clipLimit: Float = 3.0f
) {
    var lastContrastScore: Float = 1.0f
        private set

    /**
     * Process a Bitmap and return the contrast score [0,1].
     * Bitmap is scaled to 160×120 internally; no output Bitmap is allocated.
     */
    fun process(bitmap: Bitmap): Float {
        val w = 160; val h = 120
        val scaled = Bitmap.createScaledBitmap(bitmap, w, h, false)

        val pixels = IntArray(w * h)
        scaled.getPixels(pixels, 0, w, 0, 0, w, h)
        scaled.recycle()

        // Extract luminance
        val lum = IntArray(w * h) { i ->
            val r = Color.red(pixels[i])
            val g = Color.green(pixels[i])
            val b = Color.blue(pixels[i])
            (0.299f * r + 0.587f * g + 0.114f * b).toInt().coerceIn(0, 255)
        }

        val tileW = w / tilesX
        val tileH = h / tilesY

        // Build CLUT per tile
        val cluts = Array(tilesX * tilesY) { ti ->
            val tx = ti % tilesX; val ty = ti / tilesX
            val x0 = tx * tileW; val y0 = ty * tileH

            val hist = IntArray(256)
            for (y in y0 until min(y0 + tileH, h)) {
                for (x in x0 until min(x0 + tileW, w)) {
                    hist[lum[y * w + x]]++
                }
            }

            // Clip and redistribute
            val tilePixels = tileW * tileH
            val clipVal = (clipLimit * tilePixels / 256f).toInt().coerceAtLeast(1)
            var excess = 0
            for (v in 0..255) {
                if (hist[v] > clipVal) { excess += hist[v] - clipVal; hist[v] = clipVal }
            }
            val redistribute = excess / 256
            for (v in 0..255) hist[v] += redistribute

            // Build cumulative LUT
            val lut = IntArray(256)
            var cum = 0
            for (v in 0..255) { cum += hist[v]; lut[v] = (cum * 255 / tilePixels).coerceIn(0, 255) }
            lut
        }

        // Apply bilinear interpolation between tile CLUTs
        val outLum = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val txf = (x.toFloat() / tileW - 0.5f).coerceIn(0f, (tilesX - 1).toFloat())
                val tyf = (y.toFloat() / tileH - 0.5f).coerceIn(0f, (tilesY - 1).toFloat())
                val tx0 = txf.toInt().coerceIn(0, tilesX - 1)
                val ty0 = tyf.toInt().coerceIn(0, tilesY - 1)
                val tx1 = min(tx0 + 1, tilesX - 1)
                val ty1 = min(ty0 + 1, tilesY - 1)
                val fx = txf - tx0; val fy = tyf - ty0

                val src = lum[y * w + x]
                val v00 = cluts[ty0 * tilesX + tx0][src]
                val v10 = cluts[ty0 * tilesX + tx1][src]
                val v01 = cluts[ty1 * tilesX + tx0][src]
                val v11 = cluts[ty1 * tilesX + tx1][src]

                outLum[y * w + x] = (
                    v00 * (1 - fx) * (1 - fy) +
                    v10 * fx       * (1 - fy) +
                    v01 * (1 - fx) * fy       +
                    v11 * fx       * fy
                ).toInt().coerceIn(0, 255)
            }
        }

        // Contrast score: normalised stddev of equalized luminance histogram
        val mean = outLum.average().toFloat()
        val variance = outLum.map { (it - mean) * (it - mean) }.average().toFloat()
        val stddev = Math.sqrt(variance.toDouble()).toFloat()
        lastContrastScore = (stddev / 80f).coerceIn(0f, 1f)

        return lastContrastScore
    }
}
