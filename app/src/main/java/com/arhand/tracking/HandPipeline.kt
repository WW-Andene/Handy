package com.arhand.tracking

import com.arhand.util.clampDelta
import com.arhand.util.FrameThrottler
import com.arhand.tracking.LM
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/** Linear interpolation helper used for A2 confidence-weighted OEF params. */
private fun lerp(a: Float, b: Float, t: Float) = a + (b - a) * t

data class ProcessedHand(
    val landmarks: HandLandmarks,
    val slotIndex: Int           // 0 or 1
)

/**
 * Orchestrates the full tracking pipeline per hand slot.
 *
 * raw → clampDelta → TDF → OcclusionEngine → OEF → accepted
 *
 * Also manages:
 *   - Grace period (8 frames) on hand loss before slot cleared
 *   - Flicker rejection (PERSIST_MIN = 2 frames) on re-appearance
 *   - Reset of OEF + TDF on confirmed hand loss
 */
class HandPipeline {

    companion object {
        const val NUM_SLOTS       = 2
        const val GRACE_FRAMES    = 8
        const val PERSIST_MIN     = 2
        const val OEF_MIN_CUTOFF_XY = 0.9f
        const val OEF_BETA_XY       = 1.5f
        const val OEF_MIN_CUTOFF_Z  = 1.2f
        const val OEF_BETA_Z        = 1.8f
    }

    private val tdf  = Array(NUM_SLOTS) { TemporalDepthFusion() }
    private val occ  = Array(NUM_SLOTS) { OcclusionEngine() }
    private val oef  = Array(NUM_SLOTS) {
        OneEuroFilter(OEF_MIN_CUTOFF_XY, OEF_BETA_XY, OEF_MIN_CUTOFF_Z, OEF_BETA_Z)
    }

    // G4 — One GestureClassifier per hand slot
    private val gestureClassifiers = Array(NUM_SLOTS) { GestureClassifier() }

    private val graceCounts  = IntArray(NUM_SLOTS) { 0 }   // frames since last detection
    private val persistCounts = IntArray(NUM_SLOTS) { 0 }  // consecutive detection frames
    private val prevLms      = arrayOfNulls<HandLandmarks>(NUM_SLOTS)
    private val activeLms    = arrayOfNulls<HandLandmarks>(NUM_SLOTS)

    private val _processed = MutableStateFlow<List<ProcessedHand>>(emptyList())
    val processed: StateFlow<List<ProcessedHand>> = _processed

    /**
     * G4 — Current recognized gesture per slot (index 0 = primary hand).
     * Null means no gesture was confidently recognized this frame.
     * Consumers (AppViewModel, HUD, shortcut handler) collect this flow.
     */
    private val _gestures = MutableStateFlow<List<Gesture?>>(listOf(null, null))
    val gestures: StateFlow<List<Gesture?>> = _gestures

    /**
     * Feed raw hand results from MediaPipe.
     * @param rawHands List of up to 2 raw HandLandmarks from detector
     * @param nowMs    Current timestamp in milliseconds
     */
    fun update(rawHands: List<HandLandmarks>, confidence: Float, nowMs: Long) {
        val output = mutableListOf<ProcessedHand>()

        for (slot in 0 until NUM_SLOTS) {
            val raw = rawHands.getOrNull(slot)

            if (raw == null) {
                // Hand not present this frame
                graceCounts[slot]++
                persistCounts[slot] = 0
                if (graceCounts[slot] >= GRACE_FRAMES) {
                    // Confirmed loss — reset pipeline
                    if (activeLms[slot] != null) {
                        tdf[slot].reset()
                        oef[slot].reset()
                        occ[slot].reset()
                        gestureClassifiers[slot].reset()
                        activeLms[slot] = null
                        prevLms[slot] = null
                    }
                } else {
                    // Still in grace period — keep last accepted
                    activeLms[slot]?.let { output.add(ProcessedHand(it, slot)) }
                }
                continue
            }

            // Hand present — check flicker rejection
            graceCounts[slot] = 0
            persistCounts[slot]++
            if (persistCounts[slot] < PERSIST_MIN) {
                activeLms[slot]?.let { output.add(ProcessedHand(it, slot)) }
                continue
            }

            // Step 1: clamp delta
            val clamped = prevLms[slot]?.let { clampDelta(it, raw) } ?: raw

            // Step 2: Temporal Depth Fusion
            val fused = tdf[slot].apply(clamped, nowMs)

            // Step 3: Occlusion inference
            val inferred = occ[slot].applyInference(fused, confidence, nowMs)

            // Step 4: One Euro Filter — A2: scale params by per-frame confidence.
            // Low confidence (0.3–0.5) → higher minCutoff, lower beta (suppress jitter).
            // High confidence (> 0.85) → base params (more responsive).
            val confNorm = confidence.coerceIn(0.3f, 0.9f) / 0.9f
            val effectiveCutoff = lerp(1.4f, OEF_MIN_CUTOFF_XY, confNorm)
            val effectiveBeta   = lerp(0.8f, OEF_BETA_XY,       confNorm)

            // A3 — Wrist-relative coordinate normalization.
            // Subtract wrist position so OEF only sees hand pose, not global translation.
            // This prevents fast panning (hand moving across frame) from being treated as
            // high velocity and under-filtered. Re-add wrist position after smoothing.
            val wrist = inferred[LM.WRIST]
            val localLms = inferred.map { lm ->
                lm.copy(x = lm.x - wrist.x, y = lm.y - wrist.y, z = lm.z - wrist.z)
            }
            val smoothedLocal = oef[slot].smooth(localLms, nowMs, effectiveCutoff, effectiveBeta)
            val smoothed = smoothedLocal.map { lm ->
                lm.copy(x = lm.x + wrist.x, y = lm.y + wrist.y, z = lm.z + wrist.z)
            }

            prevLms[slot] = smoothed
            activeLms[slot] = smoothed
            output.add(ProcessedHand(smoothed, slot))
        }

        _processed.value = output

        // G4 — Classify gesture for each slot from the accepted (smoothed) landmarks.
        // Run after all slots processed so the gesture array matches slot indices.
        val gestureList = List(NUM_SLOTS) { slot ->
            activeLms[slot]?.let { gestureClassifiers[slot].classify(it) }
        }
        _gestures.value = gestureList
    }

    fun resetAll() {
        for (slot in 0 until NUM_SLOTS) {
            tdf[slot].reset()
            oef[slot].reset()
            occ[slot].reset()
            gestureClassifiers[slot].reset()
            graceCounts[slot] = 0
            persistCounts[slot] = 0
            prevLms[slot] = null
            activeLms[slot] = null
        }
        _processed.value = emptyList()
        _gestures.value  = listOf(null, null)
    }

    fun getOcclusionProb(slot: Int): FloatArray = occ.getOrNull(slot)?.occlusionProb ?: FloatArray(21)
}
