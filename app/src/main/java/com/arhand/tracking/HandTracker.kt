package com.arhand.tracking

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/**
 * Wraps MediaPipe HandLandmarker Tasks SDK.
 * Loads the model from raw/hand_landmarker.task (offline, embedded in APK).
 * Uses GPU delegate via NNAPI for ~60fps on Snapdragon 8 Gen 2.
 */
class HandTracker(
    private val context: Context,
    private val onResult: (HandLandmarkerResult, Long) -> Unit
) {
    private var landmarker: HandLandmarker? = null

    fun init() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .setDelegate(Delegate.GPU)
            .build()

        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener { result, image ->
                onResult(result, System.currentTimeMillis())
            }
            .build()

        landmarker = HandLandmarker.createFromOptions(context, options)
    }

    /**
     * Submit a frame for async detection.
     * Results arrive via onResult callback.
     */
    fun detect(bitmap: Bitmap, timestampMs: Long) {
        val mpImage = BitmapImageBuilder(bitmap).build()
        landmarker?.detectAsync(mpImage, timestampMs)
    }

    fun close() {
        landmarker?.close()
        landmarker = null
    }

    /**
     * Convert MediaPipe result to our Landmark type.
     * Returns list of up to 2 hands, each with 21 landmarks.
     */
    fun parseResult(result: HandLandmarkerResult): List<HandLandmarks> {
        return result.landmarks().map { hand ->
            hand.map { lm ->
                Landmark(x = lm.x(), y = lm.y(), z = lm.z())
            }
        }
    }
}
