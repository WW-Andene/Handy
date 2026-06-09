package com.arhand.util

/**
 * Adaptive frame throttler — mirrors _inferEvery logic in the HTML prototype.
 *
 * On mobile, inference budget is capped at MIN=1 MAX=4 frames between runs.
 * The throttler adjusts based on last inference duration to keep UI smooth.
 */
class FrameThrottler(
    private val minEvery: Int = 1,
    private val maxEvery: Int = 4,
    private val targetMs: Float = 20f   // target inference time budget per frame
) {
    private var inferEvery: Int = 2
    private var frameCount: Int = 0

    /**
     * Call every render frame. Returns true if inference should run this frame.
     */
    fun shouldInfer(): Boolean {
        frameCount++
        return (frameCount % inferEvery) == 0
    }

    /**
     * Call after inference completes with the measured duration.
     * Adjusts throttle rate to stay within budget.
     */
    fun reportInferenceMs(ms: Float) {
        inferEvery = when {
            ms < targetMs * 0.5f -> maxOf(minEvery, inferEvery - 1)
            ms > targetMs * 1.5f -> minOf(maxEvery, inferEvery + 1)
            else -> inferEvery
        }
    }

    fun currentRate(): Int = inferEvery
    fun reset() { frameCount = 0; inferEvery = 2 }
}
