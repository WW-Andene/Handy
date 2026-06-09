package com.arhand.tracking

import com.arhand.util.clampDelta
import com.arhand.util.FrameThrottler
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

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

    private val graceCounts  = IntArray(NUM_SLOTS) { 0 }   // frames since last detection
    private val persistCounts = IntArray(NUM_SLOTS) { 0 }  // consecutive detection frames
    private val prevLms      = arrayOfNulls<HandLandmarks>(NUM_SLOTS)
    private val activeLms    = arrayOfNulls<HandLandmarks>(NUM_SLOTS)

    private val _processed = MutableStateFlow<List<ProcessedHand>>(emptyList())
    val processed: StateFlow<List<ProcessedHand>> = _processed

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

            // Step 4: One Euro Filter
            val smoothed = oef[slot].smooth(inferred, nowMs)

            prevLms[slot] = smoothed
            activeLms[slot] = smoothed
            output.add(ProcessedHand(smoothed, slot))
        }

        _processed.value = output
    }

    fun resetAll() {
        for (slot in 0 until NUM_SLOTS) {
            tdf[slot].reset()
            oef[slot].reset()
            occ[slot].reset()
            graceCounts[slot] = 0
            persistCounts[slot] = 0
            prevLms[slot] = null
            activeLms[slot] = null
        }
        _processed.value = emptyList()
    }

    fun getOcclusionProb(slot: Int): FloatArray = occ.getOrNull(slot)?.occlusionProb ?: FloatArray(21)
}
