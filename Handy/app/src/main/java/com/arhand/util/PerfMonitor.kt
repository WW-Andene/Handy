package com.arhand.util

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

data class PerfState(
    val fps: Float = 0f,
    val inferenceMs: Float = 0f,
    val skipRatio: Float = 0f
)

/**
 * Tracks FPS and inference timing.
 * Exposes StateFlow<PerfState> consumed by HUD overlay.
 */
class PerfMonitor {
    private val _state = MutableStateFlow(PerfState())
    val state: StateFlow<PerfState> = _state

    private var frameCount = 0
    private var windowStart = System.nanoTime()
    private var inferenceMsEma = 0f
    private var inferredFrames = 0
    private var totalFrames = 0

    fun onFrame() {
        frameCount++
        totalFrames++
        val now = System.nanoTime()
        val elapsed = (now - windowStart) / 1_000_000_000f
        if (elapsed >= 0.5f) {
            val fps = frameCount / elapsed
            _state.value = _state.value.copy(
                fps = fps,
                skipRatio = if (totalFrames > 0) 1f - (inferredFrames.toFloat() / totalFrames) else 0f
            )
            frameCount = 0
            windowStart = now
        }
    }

    fun onInference(ms: Float) {
        inferredFrames++
        inferenceMsEma = inferenceMsEma * 0.8f + ms * 0.2f
        _state.value = _state.value.copy(inferenceMs = inferenceMsEma)
    }

    fun reset() {
        frameCount = 0
        totalFrames = 0
        inferredFrames = 0
        inferenceMsEma = 0f
        windowStart = System.nanoTime()
        _state.value = PerfState()
    }
}
