package com.arhand.depth

import com.arhand.tracking.HandLandmarks
import com.arhand.util.Vec3
import com.arhand.util.landmarkToWorld
import kotlin.math.*

/**
 * Visual Hull SDF Carver — port of SCANNER._dcCarve() from the HTML prototype.
 *
 * Builds a signed distance field from capsule volumes along each bone segment,
 * then runs Marching Cubes to extract the mesh.
 *
 * This is a proxy reconstruction — it approximates the hand as a union of
 * capsules. Real metric depth requires ARCore (Phase 2 in the master).
 */
object DepthCarver {

    const val GRID_N     = 32       // 32³ grid
    const val GRID_SIZE  = 0.5f     // world-space extent of grid

    private val BONE_SEGMENTS = listOf(
        // Thumb
        1 to 2, 2 to 3, 3 to 4,
        // Index
        5 to 6, 6 to 7, 7 to 8,
        // Middle
        9 to 10, 10 to 11, 11 to 12,
        // Ring
        13 to 14, 14 to 15, 15 to 16,
        // Pinky
        17 to 18, 18 to 19, 19 to 20,
        // Palm
        0 to 5, 0 to 9, 0 to 13, 0 to 17,
        5 to 9, 9 to 13, 13 to 17
    )

    /**
     * Carve SDF from multiple accumulated landmark frames.
     * @param frames List of world-space landmark lists captured during scan
     * @return Marching cubes mesh as flat position array
     */
    fun carveAndExtract(frames: List<List<Vec3>>): FloatArray {
        if (frames.isEmpty()) return FloatArray(0)

        val N = GRID_N
        val cellSize = GRID_SIZE * 2f / N
        val origin = floatArrayOf(-GRID_SIZE, -GRID_SIZE, -GRID_SIZE)

        // Init SDF: large positive (outside)
        val sdf = FloatArray(N * N * N) { GRID_SIZE * 2f }

        fun idx(x: Int, y: Int, z: Int) = x + N * y + N * N * z

        // Capsule SDF: signed distance to segment (A,B) with radius r
        fun capsuleSdf(px: Float, py: Float, pz: Float,
                       ax: Float, ay: Float, az: Float,
                       bx: Float, by: Float, bz: Float,
                       r: Float): Float {
            val abx = bx - ax; val aby = by - ay; val abz = bz - az
            val apx = px - ax; val apy = py - ay; val apz = pz - az
            val t = ((apx * abx + apy * aby + apz * abz) /
                     (abx * abx + aby * aby + abz * abz + 1e-8f)).coerceIn(0f, 1f)
            val qx = ax + t * abx; val qy = ay + t * aby; val qz = az + t * abz
            val dx = px - qx; val dy = py - qy; val dz = pz - qz
            return sqrt(dx * dx + dy * dy + dz * dz) - r
        }

        // Estimate bone radius from palm size across frames
        val avgPalmSize = frames.map { lms ->
            if (lms.size < 10) 0.05f
            else (lms[0] - lms[9]).length()
        }.average().toFloat().coerceAtLeast(0.02f)
        val boneR = avgPalmSize * 0.14f

        // For each voxel, compute minimum capsule SDF across all frames and segments
        for (zi in 0 until N) {
            for (yi in 0 until N) {
                for (xi in 0 until N) {
                    val px = origin[0] + (xi + 0.5f) * cellSize
                    val py = origin[1] + (yi + 0.5f) * cellSize
                    val pz = origin[2] + (zi + 0.5f) * cellSize

                    var minD = GRID_SIZE * 2f
                    for (lms in frames) {
                        if (lms.size < 21) continue
                        for ((ai, bi) in BONE_SEGMENTS) {
                            val a = lms[ai]; val b = lms[bi]
                            val d = capsuleSdf(px, py, pz, a.x, a.y, a.z, b.x, b.y, b.z, boneR)
                            if (d < minD) minD = d
                        }
                    }
                    sdf[idx(xi, yi, zi)] = minD
                }
            }
        }

        return MarchingCubes.extract(sdf, N, cellSize, origin)
    }

    /**
     * Convert HandLandmarks to world-space Vec3 list.
     */
    fun landmarksToWorld(lms: HandLandmarks, aspect: Float): List<Vec3> =
        lms.map { lm ->
            val (x, y, z) = landmarkToWorld(lm, aspect, mirrorX = false)
            Vec3(x, y, z)
        }
}
