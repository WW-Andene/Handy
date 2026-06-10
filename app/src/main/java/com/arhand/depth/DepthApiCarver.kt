package com.arhand.depth

import com.arhand.util.Vec3
import kotlin.math.sqrt

/**
 * B6 — ARCore Depth API integration (Phase 2), steps 4-5.
 *
 * Replaces [DepthCarver]'s capsule-SDF proxy with a real metric-depth SDF built from
 * the fused world-space point cloud accumulated via [ArDepthSession].
 *
 * Approach:
 *   1. Voxel-fuse all accumulated depth points into a deduplicated cloud (avoids
 *      O(points × voxels) blowup with multi-second scans).
 *   2. For each voxel center in the carve grid, the SDF value is the signed distance
 *      to the nearest fused point, with sign determined by whether the voxel lies on
 *      the camera side (outside) or far side (inside) of the local point — approximated
 *      via a thin-shell offset (points are surface samples, so we carve a shell of
 *      thickness ~2*cellSize around them, marking voxels within the shell as "inside").
 *   3. [MarchingCubes.extract] runs unchanged on the resulting grid (B6 step 5),
 *      reusing the same [DepthCarver.GRID_SIZE] / [DepthCarver.effectiveGridN] /
 *      origin-centering conventions as the capsule-SDF path.
 *
 * If the fused cloud is too sparse (e.g. depth API unsupported mid-scan, or user moved
 * too fast), [carveAndExtract] returns an empty array so the caller can fall back to
 * [DepthCarver.carveAndExtract].
 */
object DepthApiCarver {

    /** Minimum number of fused points required to attempt a depth-based carve. */
    const val MIN_FUSED_POINTS = 500

    /** Shell half-thickness, in multiples of cellSize, around each surface point. */
    private const val SHELL_CELLS = 1.5f

    /**
     * B6 step 4 — fuse raw per-frame depth point clouds into a single deduplicated
     * world-space cloud via voxel-grid hashing (tolerance = [cellSize]).
     */
    fun fusePointClouds(frames: List<List<Vec3>>, cellSize: Float): List<Vec3> {
        val voxelMap = HashMap<Long, Vec3>()
        for (frame in frames) {
            for (p in frame) {
                val key = ArDepthSession.voxelKey(p, cellSize)
                // Keep first sample per voxel — averaging across many frames is possible
                // but first-write is sufficient for SDF shell purposes and avoids
                // per-voxel running-average bookkeeping.
                voxelMap.putIfAbsent(key, p)
            }
        }
        return voxelMap.values.toList()
    }

    /**
     * B6 steps 4-5 — build a thin-shell SDF from the fused point cloud and run
     * [MarchingCubes.extract] over [DepthCarver.effectiveGridN]³ voxels.
     *
     * Grid origin/size match [DepthCarver] conventions (B3 centroid centering,
     * B4 LOD grid resolution) so downstream consumers (smoothing, GLB export,
     * point-cloud export) need no changes.
     *
     * @param depthFrames Per-frame world-space depth point clouds from [ArDepthSession.update]
     * @return Marching cubes mesh as flat position array, or empty if the fused cloud
     *         is too sparse (caller should fall back to [DepthCarver.carveAndExtract]).
     */
    fun carveAndExtract(depthFrames: List<List<Vec3>>): FloatArray {
        if (depthFrames.isEmpty()) return FloatArray(0)

        val N = DepthCarver.effectiveGridN
        val cellSize = DepthCarver.GRID_SIZE * 2f / N

        val fused = fusePointClouds(depthFrames, cellSize)
        if (fused.size < MIN_FUSED_POINTS) return FloatArray(0)

        // B3-equivalent — center grid on the fused cloud centroid.
        var cx = 0f; var cy = 0f; var cz = 0f
        for (p in fused) { cx += p.x; cy += p.y; cz += p.z }
        cx /= fused.size; cy /= fused.size; cz /= fused.size

        val origin = floatArrayOf(cx - DepthCarver.GRID_SIZE, cy - DepthCarver.GRID_SIZE, cz - DepthCarver.GRID_SIZE)

        // Spatial hash of fused points for nearest-neighbor lookup, bucketed by the
        // same voxel grid as the carve grid for O(1) average lookups.
        val buckets = HashMap<Long, MutableList<Vec3>>()
        for (p in fused) {
            val key = ArDepthSession.voxelKey(p, cellSize)
            buckets.getOrPut(key) { mutableListOf() }.add(p)
        }

        fun nearestDist(px: Float, py: Float, pz: Float): Float {
            val qx = ((px - 0f) / cellSize).let { Math.round(it) }
            val qy = ((py - 0f) / cellSize).let { Math.round(it) }
            val qz = ((pz - 0f) / cellSize).let { Math.round(it) }
            var best = Float.MAX_VALUE
            // Search the 3x3x3 neighborhood of voxel buckets around this point.
            for (dx in -1..1) for (dy in -1..1) for (dz in -1..1) {
                val key = ArDepthSession.voxelKey(
                    Vec3((qx + dx) * cellSize, (qy + dy) * cellSize, (qz + dz) * cellSize),
                    cellSize
                )
                val bucket = buckets[key] ?: continue
                for (p in bucket) {
                    val ddx = px - p.x; val ddy = py - p.y; val ddz = pz - p.z
                    val d = sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                    if (d < best) best = d
                }
            }
            return best
        }

        val shellThickness = cellSize * SHELL_CELLS
        val sdf = FloatArray(N * N * N) { DepthCarver.GRID_SIZE * 2f }

        fun idx(x: Int, y: Int, z: Int) = x + N * y + N * N * z

        for (zi in 0 until N) {
            for (yi in 0 until N) {
                for (xi in 0 until N) {
                    val px = origin[0] + (xi + 0.5f) * cellSize
                    val py = origin[1] + (yi + 0.5f) * cellSize
                    val pz = origin[2] + (zi + 0.5f) * cellSize

                    val d = nearestDist(px, py, pz)
                    // Thin-shell SDF: negative (inside) within shellThickness of a
                    // surface sample, positive (outside) beyond it. This produces a
                    // closed-ish surface for marching cubes around the depth shell.
                    sdf[idx(xi, yi, zi)] = d - shellThickness
                }
            }
        }

        return MarchingCubes.extract(sdf, N, cellSize, origin)
    }
}
