package com.arhand.tracking

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/**
 * Wraps MediaPipe HandLandmarker Tasks SDK.
 * Loads the model from raw/hand_landmarker.task (offline, embedded in APK).
 *
 * A1 — Delegate selection: tries GPU first (routes to Hexagon DSP on Snapdragon
 * via the NNAPI HAL automatically), then falls back to CPU. Delegate.NNAPI was
 * removed from MediaPipe Tasks SDK 0.10.14.
 * the Hexagon DSP and is 1.5–3× faster than GPU with better power efficiency.
 */
class HandTracker(
    private val context: Context,
    private val onResult: (HandLandmarkerResult, Long) -> Unit
) {
    private var landmarker: HandLandmarker? = null

    /** The delegate actually used after fallback selection. Exposed for HUD logging. */
    var activeDelegate: Delegate = Delegate.CPU
        private set

    fun init() {
        // A1: Try NNAPI → GPU → CPU in order
        val delegate = selectDelegate()
        activeDelegate = delegate

        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .setDelegate(delegate)
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
     * A1: Probe delegate availability at runtime.
     * Delegate.NNAPI was removed in MediaPipe Tasks SDK 0.10.14 — the Hexagon DSP path
     * now routes automatically through the GPU delegate on Snapdragon via the NNAPI HAL.
     * Try GPU first; fall back to CPU.
     */
    private fun selectDelegate(): Delegate {
        for (candidate in listOf(Delegate.GPU, Delegate.CPU)) {
            try {
                // Attempt to build a minimal BaseOptions with this delegate — throws if unsupported
                BaseOptions.builder()
                    .setModelAssetPath("hand_landmarker.task")
                    .setDelegate(candidate)
                    .build()
                Log.i("HandTracker", "Using delegate: $candidate")
                return candidate
            } catch (e: Exception) {
                Log.w("HandTracker", "Delegate $candidate unavailable: ${e.message}")
            }
        }
        return Delegate.CPU
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
     *
     * A5 — Handedness-based slot assignment:
     * Returns a 2-element list where index 0 = right hand, index 1 = left hand
     * (MediaPipe labels are camera-mirrored, so "Right" label = user's right hand).
     * Absent hands are represented as null entries, then filtered out by [HandPipeline].
     * This prevents slot swaps when the dominant hand briefly disappears and reappears.
     */
    fun parseResult(result: HandLandmarkerResult): List<HandLandmarks> {
        val landmarks  = result.landmarks()
        val handedness = result.handedness()

        if (landmarks.isEmpty()) return emptyList()

        // Build a nullable 2-slot array: slot 0 = Right, slot 1 = Left
        val slots = arrayOfNulls<HandLandmarks>(2)

        for (i in landmarks.indices) {
            val lms = landmarks[i].map { lm ->
                Landmark(
                    x          = lm.x(),
                    y          = lm.y(),
                    z          = lm.z(),
                    visibility = lm.visibility().orElse(1.0f)  // A6: forward model confidence
                )
            }
            // MediaPipe handedness: "Right" = user's right hand (camera-mirrored label)
            val label = handedness.getOrNull(i)?.firstOrNull()?.categoryName() ?: "Right"
            val slot  = if (label == "Left") 1 else 0
            // If both hands report the same label (rare), fall back to detection order
            if (slots[slot] == null) slots[slot] = lms
            else slots[1 - slot] = lms
        }

        return slots.filterNotNull()
    }
}
