package com.arhand.feature.record

/**
 * A single completed recording take.
 *
 * @param label      Human-readable name set by the user before recording.
 * @param bvhPath    Absolute path to the exported BVH file, or null if export failed.
 * @param gltfPath   Absolute path to the exported animated GLB, or null.
 * @param durationMs Recording duration in milliseconds.
 * @param frameCount Number of frames captured.
 * @param takenAtMs  Epoch-ms when the take was completed.
 */
data class TakeEntry(
    val label:      String,
    val bvhPath:    String?,
    val gltfPath:   String?,
    val durationMs: Long,
    val frameCount: Int,
    val takenAtMs:  Long = System.currentTimeMillis()
)
