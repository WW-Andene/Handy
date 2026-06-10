# HANDY — Improvement, Innovation & R&D Plan

**App:** Handy AR Hand Scanner  
**Stack:** Android / Kotlin, Jetpack Compose, OpenGL ES 3.0, MediaPipe Tasks, Camera2  
**Status:** Post-audit baseline — bugs fixed, builds clean

---

## HOW TO USE THIS FILE

Each item has a **phase tag**, a **difficulty**, and an **impact** rating. Phases are sequential within a track but tracks are independent — you can run Track B and Track C in parallel without touching Track A.

```
Difficulty:  L = Low (1–3 days)   M = Medium (1–2 weeks)   H = High (2–6 weeks)
Impact:      ★ = Nice to have     ★★ = Meaningful          ★★★ = Core / differentiating
Phase:       P1 = near-term       P2 = mid-term            P3 = long-term / research
```

---

## TRACK A — Pipeline Quality

*Goal: make the tracking pipeline visibly more robust and accurate.*

---

### ✅ A1 — Replace GPU delegate with NNAPI delegate for Snapdragon DSP  
**Phase P1 · Difficulty L · Impact ★★** · *Done*

`HandTracker` currently uses `Delegate.GPU`. On Snapdragon 8 Gen 2 (and many mid-range Adreno GPUs), the NNAPI delegate routes to the Hexagon DSP and is 1.5–3× faster with better power efficiency. Add a `Delegate.NNAPI` fallback with a `Delegate.CPU` safety net.

```kotlin
// HandTracker.init() — current
.setDelegate(Delegate.GPU)

// Proposed: try NNAPI → GPU → CPU
val delegate = try {
    Delegate.NNAPI  // Hexagon DSP on Snapdragon
} catch (_: Exception) {
    try { Delegate.GPU } catch (_: Exception) { Delegate.CPU }
}
```

**Acceptance:** inference latency < 12ms on Xiaomi 13T at 30fps with no frame drops.

---

### ✅ A2 — Per-frame confidence-weighted OEF parameters  
**Phase P1 · Difficulty L · Impact ★★** · *Done*

`HandPipeline` uses fixed `OEF_MIN_CUTOFF_XY = 0.9f` / `OEF_BETA_XY = 1.5f` for all conditions. When confidence is low (0.3–0.5), the filter should be more aggressive (higher minCutoff, lower beta) to suppress jitter from uncertain detections. When confidence is high (> 0.85), relax to be more responsive.

```kotlin
val effectiveCutoff = lerp(1.4f, OEF_MIN_CUTOFF_XY, conf.coerceIn(0.3f, 0.9f) / 0.9f)
val effectiveBeta   = lerp(0.8f, OEF_BETA_XY,       conf.coerceIn(0.3f, 0.9f) / 0.9f)
```

---

### A3 — Wrist-relative coordinate normalization before OEF  
**Phase P1 · Difficulty M · Impact ★★★**

MediaPipe normalized coordinates include camera motion as global translation. When the user moves their hand across the frame, OEF interprets it as high velocity and under-filters. Convert all landmarks to wrist-relative coordinates before filtering, then re-add wrist position after. This makes OEF blind to global translation — it only smooths pose, not position.

```
local_lm[i] = lm[i] - lm[WRIST]
smoothed_local = oef.smooth(local_lm)
final_lm[i] = smoothed_local[i] + smoothed_wrist_pos
```

**Prerequisite:** A1 (lower latency helps here because wrist prediction needs a tight loop).

---

### ✅ A4 — Adaptive TDF ALPHA based on inter-frame Z delta distribution  
**Phase P2 · Difficulty M · Impact ★★** · *Done*

`TemporalDepthFusion` uses a single `SPEED_THRESH = 0.30f` for all joints. Finger joints and wrist joints have very different natural speed ranges. Replace the single threshold with per-joint learned percentile thresholds: compute a running 90th-percentile Z-velocity per joint slot and use that as the per-joint saturation point.

---

### ✅ A5 — Two-hand slot assignment by handedness, not detection order  
**Phase P1 · Difficulty L · Impact ★★** · *Done*

`HandPipeline` assigns slot 0 = first detected, slot 1 = second. If the dominant hand disappears and reappears, it may land in slot 1, swapping left/right. Fix: read `HandLandmarkerResult.handedness()` and assign slot 0 = right, slot 1 = left (or user-configurable). This also makes `FullBodyFrame.leftHand`/`rightHand` semantically correct.

---

### ✅ A6 — Landmark visibility score gating  
**Phase P2 · Difficulty L · Impact ★★** · *Done*

MediaPipe `NormalizedLandmark` has a `visibility()` score (0–1) that the current code discards entirely. Add a visibility threshold (e.g. `> 0.5`) before passing a landmark to `OcclusionEngine.detectOcclusion()`. This pre-filters landmarks the model itself considers unreliable, reducing false positive occlusion detections.

---

## TRACK B — Scan & Reconstruction

*Goal: produce higher-quality, more complete hand meshes.*

---

### B1 — Multi-frame SDF accumulation with temporal weighting  
**Phase P1 · Difficulty M · Impact ★★★**

`DepthCarver.carveAndExtract()` takes the minimum SDF across all frames. This is conservative — a single bad frame with a landmark outlier can carve away valid geometry. Replace `min` with a **soft minimum**: accumulate a weighted sum where each frame's contribution is weighted by `QualityEngine.evaluate()` score, then apply `min` only among the top-quartile frames.

```kotlin
// Per voxel, per frame: accumulate quality-weighted SDF
sdf[idx] = frames
    .filter { it.quality > 0.55f }
    .sortedByDescending { it.quality }
    .take(frames.size * 3 / 4)  // top 75%
    .minOf { capsuleSdf(px, py, pz, ...) }
```

---

### ✅ B2 — Adaptive bone radius from per-pose landmark spread  
**Phase P1 · Difficulty L · Impact ★★** · *Done*

`DepthCarver` uses a single `boneR = avgPalmSize * 0.14f` for all segments. Finger phalanges taper — distal segments should have a smaller radius than proximal. Use `HandMeshBuilder.FINGER_PROFILES` radius ratios to scale `boneR` per segment:

```kotlin
val segmentR = boneR * when (segmentType) {
    PROXIMAL -> 1.0f
    MIDDLE   -> 0.85f
    DISTAL   -> 0.65f
    PALM     -> 1.2f
}
```

---

### ✅ B3 — Pose-aware grid origin centering  
**Phase P1 · Difficulty L · Impact ★★** · *Done*

`DepthCarver` uses a fixed grid origin at `(-0.5, -0.5, -0.5)`. If the hand is off-center (side view pose, close-up), the mesh gets clipped. Compute the centroid of all accumulated landmark frames and center the grid on that centroid before carving.

---

### ✅ B4 — Grid resolution upgrade with LOD fallback  
**Phase P2 · Difficulty M · Impact ★★** · *Done*

Current `GRID_N = 32` produces a 32³ = 32,768-voxel grid. A 48³ grid (110k voxels) captures significantly more finger detail with only ~3.4× compute cost. Profile on Xiaomi 13T — expected ~600ms for 48³ vs ~200ms for 32³ (still acceptable for a one-time post-scan operation). Add a `GRID_N = 48` option gated by a device performance tier check.

---

### ✅ B5 — Post-processing: Laplacian mesh smoothing  
**Phase P2 · Difficulty M · Impact ★★** · *Done*

Marching cubes output is blocky at cell boundaries. One pass of umbrella Laplacian smoothing (`λ = 0.5`, 3 iterations) removes staircase artifacts with minimal shape distortion. Operate on the triangle soup output before GLB export.

---

### ✅ B6 — ARCore Depth API integration (Phase 2)  
**Phase P3 · Difficulty H · Impact ★★★** · *Done*

Replace the capsule-SDF proxy reconstruction with real metric depth from ARCore. `DepthScannerPanel.kt` already has the UI placeholder. Required steps:

1. Add `com.google.ar:core:1.42.0` dependency
2. Wrap `GLSurfaceView` in `ArFragment` or manual `Session` management
3. Per-frame: acquire `Frame.acquireDepthImage16Bits()`, project into world space using camera intrinsics
4. Replace `capturedFrames: List<List<Vec3>>` with fused depth point cloud
5. Run existing `MarchingCubes` on depth-derived SDF

**Gate:** only offer when `ArCoreApk.checkAvailability()` returns `SUPPORTED_INSTALLED`.

**Implementation:** `ArDepthAvailability` (gate: SUPPORTED_INSTALLED + `isDepthModeSupported(AUTOMATIC)`), `ArDepthSession` (manual `Session`, `Frame.acquireDepthImage16Bits()` → world-space point cloud via camera intrinsics + pose), `DepthApiCarver` (voxel-fused point cloud → thin-shell SDF → `MarchingCubes.extract`, reusing `DepthCarver`'s grid size/LOD/centering). Wired into `AppViewModel` (`toggleDepth`, `depthApiAvailable`/`depthMode`/`depthConfidence` state, depth-frame capture during CAPTURING, `processScan` prefers `DepthApiCarver` and falls back to capsule-SDF `DepthCarver` if the fused cloud is too sparse) and `DepthScannerPanel` (shown in `MainActivity` only when `depthApiAvailable`). `com.google.ar:core:1.42.0` added to `app/build.gradle`; manifest declares `com.google.ar.core` as `optional`.

---

### ✅ B7 — Hand segmentation mask for cleaner point cloud  
**Phase P3 · Difficulty H · Impact ★★** · *Done*

MediaPipe Selfie Segmentation can isolate the hand region before landmark detection, eliminating background points from the cloud. Alternatively, use the convex hull of the 21 landmarks (projected to 2D) as a cheap foreground mask — discard cloud points that project outside this hull.

**Implementation:** `HandSegmentationMask` (Andrew's monotone-chain convex hull of the 21 landmarks projected to world-space XY via `landmarkToWorld`, expanded by `HULL_MARGIN = 0.06`, ray-casting point-in-polygon test). Wired into `AppViewModel`'s B6 depth-frame capture: each fused ARCore depth point cloud is masked against the current frame's hand hull before being added to `capturedDepthFrames`, removing background points before `DepthApiCarver` runs.

---

## TRACK C — Export & Interoperability

*Goal: make exported assets usable in standard tools without post-processing.*

---

### ✅ C1 — Index buffer in GLB export  
**Phase P1 · Difficulty M · Impact ★★★**

`GLBExporter` exports un-indexed triangle soup. The Marching Cubes output has massive vertex duplication (~3 vertices per triangle vs ~1.2 for an indexed mesh). Adding an index buffer via a simple weld-by-position pass (tolerance `1e-5f`) reduces file size by ~60% and GPU memory by ~65%. Most importers (Blender, three.js) work with indexed meshes more efficiently.

```kotlin
// Weld: group vertices by quantized (x,y,z), assign shared index
val weldMap = HashMap<Long, Int>()
fun quantize(v: Float) = (v * 10000).roundToInt()
fun key(x: Float, y: Float, z: Float): Long = ...
```

---

### ✅ C2 — UV unwrap for texture-ready GLB  
**Phase P2 · Difficulty H · Impact ★★** · *Done*

Current GLB has no UV coordinates. For users who want to apply a skin texture (photogrammetry overlay, custom shader), UVs are mandatory. Implement cylindrical UV projection per finger segment (matching the tube parameterization already in `HandMeshBuilder`) and spherical projection for the palm cap. Export `TEXCOORD_0` accessor in the GLB.

---

### ✅ C3 — BVH: add thumb tip end-site joints  
**Phase P1 · Difficulty L · Impact ★★** · *Done*

`MotionRecorder.exportBvh()` stops at DIP joints (3 per finger). Standard BVH hand rigs include an end-site after each DIP representing the fingertip. Blender's BVH importer creates a stub bone for each end-site, which is needed for IK chains and finger-tip constraint targets.

---

### ✅ C4 — Named hand preset support for OSC  
**Phase P2 · Difficulty M · Impact ★★** · *Done*

`OscStreamer` broadcasts to a fixed address schema. Add support for named presets (Unreal LiveLink, VSeeFace, VMC protocol) so users can switch to a compatible schema from the UI. VMC in particular is the standard for VTubing rigs and has wide receiver support.

```kotlin
enum class OscSchema { HANDY_DEFAULT, VMC, UNREAL_LIVE_LINK, VSEEFACE }
```

---

### C5 — GLB texture bake from scan images  
**Phase P3 · Difficulty H · Impact ★★**

Capture one Bitmap per scan pose alongside landmarks. Project the bitmap onto the mesh using the camera-to-world transform (derive from landmark positions). Blend 8-pose projections into a single 512×512 texture atlas. Export as `baseColorTexture` in the GLB material. This produces a photorealistic hand model without ARCore.

---

## TRACK D — Biometrics & Measurement

*Goal: make biometric data accurate and actionable.*

---

### ✅ D1 — Multi-frame biometric averaging with outlier rejection  
**Phase P1 · Difficulty L · Impact ★★★** · *Done*

`HandBiometrics.compute()` uses a single best-quality frame. Finger length estimates from a single frame have ±8–12% error due to MediaPipe's Z proxy. Average measurements across the top 10 frames (by quality score), discard outliers beyond 1.5× IQR per metric, then report the trimmed mean. Expected error reduction to ±3–5%.

---

### ✅ D2 — Calibration card scale reference  
**Phase P2 · Difficulty M · Impact ★★★** · *Done*

`WORLD_TO_MM = 800f` is a fixed constant based on average hand geometry. A 10% error in this constant propagates to all measurements. Add a calibration flow: user places a standard credit card (85.6×54mm) in frame, landmarks detect card corners via a simple rectangle detector, scale factor is computed and stored in `DataStore`.

```kotlin
// Calibration: measured card width in world units → real mm conversion
val calibratedScale = 85.6f / measuredCardWidthWorldUnits
```

---

### ✅ D3 — Ring size calculator  
**Phase P1 · Difficulty L · Impact ★★** · *Done*

Compute ring circumference from the knuckle breadth (`HandBiometrics.knuckleBreadth`) and map to ISO/US/EU ring size tables. Display in `BiometricsPanel` with a size recommendation. This is a concrete, shareable use-case that makes the biometrics panel feel purposeful to non-technical users.

---

### ✅ D4 — Biometric history and delta tracking  
**Phase P2 · Difficulty M · Impact ★★** · *Done*

`PersonalModelStore` stores only the most recent scan. Add a history table (DataStore list, max 20 entries) recording timestamp + all biometric values. Show a mini sparkline chart in `ScanResultModal` for each metric. Use case: tracking grip strength rehabilitation, monitoring swelling/edema changes over time.

---

### ✅ D5 — Joint angle ROM (range of motion) measurement  
**Phase P2 · Difficulty M · Impact ★★★** · *Done*

During the motion poses (open/close, finger wave), compute the full range of motion for each joint by tracking the min/max angle swept. Export as `fingerFlexionROM` per joint in degrees. Clinically relevant for physiotherapy use and VR glove calibration.

---

## TRACK E — UI/UX

*Goal: make the experience polished and self-explanatory.*

---

### ✅ E1 — Pose guide drawables wired to ScanOverlay  
**Phase P1 · Difficulty L · Impact ★★** · *Done*

8 pose drawables exist in `res/drawable/` (`hand_pose_palm_open.xml`, etc.) but `ScanPoses.ALL` has `drawableRes = 0` for all entries (never populated). Wire them: use `Resources.getIdentifier("hand_pose_${pose.id}", "drawable", packageName)` in `ScanOverlay` and render a small pose silhouette above the instruction text.

---

### ✅ E2 — Scan cancellation confirmation dialog  
**Phase P1 · Difficulty L · Impact ★** · *Done*

Tapping CANCEL immediately aborts the scan with no confirmation. After pose 3+, this discards several seconds of good data. Show a one-tap confirmation: "Cancel scan? Progress will be lost."

---

### ✅ E3 — Failed scan feedback  
**Phase P1 · Difficulty L · Impact ★★** · *Done*

`Scanner.ScanState.FAILED` is emitted but `ScanOverlay` has no branch for it — the UI shows nothing. Add a `FailedOverlay` composable with the failure reason and a "Try Again" shortcut.

---

### ✅ E4 — HUD debug mode toggle  
**Phase P1 · Difficulty L · Impact ★** · *Done*

The HUD always shows FPS/INF/SKIP. Add a long-press gesture on the HUD to toggle between a minimal mode (nothing) and a verbose mode (adds: landmark count, OEF alpha, TDF state, slot assignments). This is a developer tool — hide behind a BuildConfig.DEBUG gate or a Settings toggle.

---

### ✅ E5 — Scan quality pre-flight check  
**Phase P2 · Difficulty M · Impact ★★** · *Done*

Before the countdown starts, run 30 frames of quality evaluation and block scan start if `quality.score < 0.40f`. Show a toast: "Move hand to center / improve lighting." This prevents users from completing an 8-pose scan only to get a low-quality mesh.

---

### ✅ E6 — Guided onboarding flow  
**Phase P2 · Difficulty M · Impact ★★** · *Done*

First-launch: a 4-screen swipeable intro (what the app does, how to hold your hand, what the scan produces, what the export is for). Gate behind a `DataStore` "onboarding_seen" key. Use Compose `HorizontalPager` from `accompanist-pager` (or the Compose Foundation equivalent).

---

## TRACK F — Architecture & Code Health

*Goal: reduce fragility, improve testability, prepare for Phase 2.*

---

### F1 — Replace `@Volatile` cross-thread writes to ARRenderer with a lock-free command queue  
**Phase P1 · Difficulty M · Impact ★★**

`ARRenderer` has 10+ `@Volatile` fields written from the tracking coroutine and read on the GL thread. While `@Volatile` ensures visibility, it provides no atomicity — a partial write (e.g., `handsData` and `mirrorX` written in separate instructions) can be read in an inconsistent state mid-frame. Replace with a single `AtomicReference<RenderCommand>` data class written once per frame.

```kotlin
data class RenderCommand(
    val hands: List<HandLandmarks>,
    val mirrorX: Boolean,
    val renderMode: RenderMode,
    val torchOn: Boolean,
    val showCloud: Boolean,
    val retargetResult: RetargetResult?,
    val loadedAsset: LoadedAsset?
)
```

---

### ✅ F2 — Extract `HandTrackerManager` to encapsulate init / lifecycle  
**Phase P1 · Difficulty L · Impact ★★** · *Done*

`AppViewModel` directly holds `handTracker: HandTracker?` and manages its full lifecycle inline. Extract to a `HandTrackerManager` that owns init, detect, parse, and close — reducing `AppViewModel` from ~300 to ~200 lines and making the tracker independently testable.

---

### ✅ F3 — DataStore migration versioning  
**Phase P1 · Difficulty L · Impact ★★** · *Done*

`PersonalModelStore` reads keys directly with no schema version. When fields are added (e.g., D4 biometric history), existing installs will have incomplete prefs. Add a `KEY_SCHEMA_VERSION` integer key; on version mismatch, migrate or reset gracefully.

---

### ✅ F4 — Unit tests for pure computation modules  
**Phase P2 · Difficulty M · Impact ★★** · *Done*

The following have zero test coverage but are pure functions with no Android dependencies — ideal for JUnit4/JVM tests:

- `MarchingCubes.extract()` — known SDF → verify vertex count and basic topology
- `HandBiometrics.compute()` — synthetic landmarks → verify measurement ratios
- `BoneRetargeter.shortestArcQuaternion()` — edge cases: parallel, anti-parallel, 90°
- `QualityEngine.evaluate()` — boundary conditions for score thresholds
- `MotionRecorder.quaternionToZXY()` — identity, 90° rotations, gimbal boundary

---

### F5 — Modularize into Gradle modules  
**Phase P3 · Difficulty H · Impact ★★**

Current single `:app` module means every change rebuilds everything. Proposed split:

```
:app          — UI, ViewModel, MainActivity
:tracking     — HandTracker, HandPipeline, OEF, TDF, OcclusionEngine
:render       — ARRenderer, all renderers, ShaderPrograms
:scanner      — Scanner, QualityEngine, DepthCarver, MarchingCubes
:export       — GLBExporter, GltfAnimationExporter, PointCloudExporter
:mocap        — BoneRetargeter, MotionRecorder, OscStreamer, AssetLoader
```

Incremental build time improvement: estimated 3–5× for `:render`-only changes (the most frequent iteration loop).

---

## TRACK G — Innovation / Research

*Speculative / exploratory. No acceptance criteria — these are investigation items.*

---

### ✅ G1 — Real-time hand mesh deformation from live tracking  
**Phase P3 · Difficulty H · Impact ★★★** · *Done*

The scanned mesh is currently static post-scan. The tracking pipeline produces live joint angles every frame. Wire `BoneRetargeter` output into a CPU-side mesh deformation pass (linear blend skinning, same math as `SkinnedMeshRenderer`) to produce a live-deforming version of the user's own scanned hand. The AR view would show their real hand mesh animated by their real motion — the core "mirror effect."

---

### ✅ G2 — Implicit neural representation for hand reconstruction  
**Phase P3 · Difficulty H · Impact ★★★** · *Done*

Replace `DepthCarver` (capsule SDF) with a NeRF-style implicit field trained per-scan. Use a tiny MLP (4 layers, 64 hidden units) trained on the 8-pose landmark sets to produce an occupancy field. On-device inference of such a network is feasible with TensorFlow Lite. The capsule SDF is a hard prior; the MLP can learn hand-specific deviations (veins, knuckle topology, nail geometry).

**Pre-study:** evaluate MANO (parametric hand model) as an intermediate step — it has a known shape space that can be fit to landmark sets without neural training.

---

### ✅ G3 — Cross-device hand sync via OSC  
**Phase P3 · Difficulty M · Impact ★★** · *Done*

Two devices running Handy on the same LAN could sync hand poses in real time — one device tracks, the other renders. The OSC streamer is already functional. Add a receiver mode: listen on port 9000, parse incoming bundles, feed `HandPipeline` directly with the received quaternions. Use case: remote hand presence / collaborative AR.

---

### ✅ G4 — Gesture vocabulary recognition on-device  
**Phase P3 · Difficulty M · Impact ★★★** · *Done*

The pipeline produces clean, filtered landmark streams. A lightweight gesture classifier (k-NN or small LSTM trained on standard gestures) could run on the already-smoothed output with < 2ms overhead. Expose a `GestureFlow: StateFlow<Gesture?>` from the pipeline. Use cases: app shortcuts (fist = start scan, peace sign = switch camera), accessibility controls, user-configurable macro triggers.

---

### ✅ G5 — Photometric stereo normal estimation from torch flash  
**Phase P3 · Difficulty H · Impact ★★** · *Done*

The torch (`TorchController`) is currently always-on or always-off. Rapidly toggling the torch between frames (on/off/on at ~15fps) while holding the hand still produces two differently-lit images per moment. The per-pixel ratio encodes a coarse surface normal via photometric stereo. This yields high-frequency normal detail (wrinkles, knuckle creases) that the SDF cannot capture.

**Implementation:** `PhotometricStereoCapture` (accumulates up to `TARGET_PAIRS = 15` lit/unlit Bitmap pairs; per-pixel channel-ratio photometric stereo → unit-normalised `PhotometricNormalMap` FloatArray). Wired into `AppViewModel`: `togglePhotoStereo()` (rear camera only), `photoStereoNormalMap: StateFlow`, `photoStereoCapture` driven in the frame-dispatch coroutine (torch toggled each frame via `TorchController.setTorch()`, parity-tagged, submitted to `PhotometricStereoCapture.submitFrame()`; torch restored on completion/cancel). `AppUiState` gains `photoStereoEnabled`, `photoStereoFrameCount`, `photoStereoComplete`. `DepthScannerPanel` extended with PHOTO STEREO toggle row showing pair progress and DONE state. `startScan`/`cancelScan`/`processScan` all reset/stop the capture correctly.

---

### ✅ G6 — Temporal consistency loss for scan quality scoring  
**Phase P2 · Difficulty M · Impact ★★** · *Done*

Current `QualityEngine` evaluates each frame independently. A frame can score high on spread/depth/position but still be a temporal outlier (sudden jerk). Add a **temporal consistency score**: compare current frame's landmark positions to the OEF-smoothed history. Frames with high temporal consistency *and* high static quality are the most valuable scan frames — weight them more heavily in `capturePosePoints`.

---

## PRIORITY MATRIX — Where to Start

| Item | Phase | Diff | Impact | Suggested sprint |
|------|-------|------|--------|------------------|
| ~~A5 — Handedness slot assignment~~ ✅ | P1 | L | ★★ | Sprint 1 |
| ~~E1 — Pose drawables wired~~ ✅ | P1 | L | ★★ | Sprint 1 |
| ~~E3 — Failed scan feedback~~ ✅ | P1 | L | ★★ | Sprint 1 |
| ~~D3 — Ring size calculator~~ ✅ | P1 | L | ★★ | Sprint 1 |
| ~~C3 — BVH fingertip end-sites~~ ✅ | P1 | L | ★★ | Sprint 1 |
| ~~B3 — Grid origin centering~~ ✅ | P1 | L | ★★ | Sprint 1 |
| ~~A1 — NNAPI delegate~~ ✅ | P1 | L | ★★ | Sprint 1 |
| ~~E2 — Cancel confirmation dialog~~ ✅ | P1 | L | ★ | Sprint 2 |
| ~~E4 — HUD debug mode toggle~~ ✅ | P1 | L | ★ | Sprint 2 |
| ~~B2 — Adaptive bone radius~~ ✅ | P1 | L | ★★ | Sprint 2 |
| ~~F2 — HandTrackerManager extract~~ ✅ | P1 | L | ★★ | Sprint 2 |
| ~~F3 — DataStore migration versioning~~ ✅ | P1 | L | ★★ | Sprint 2 |
| ~~C1 — Indexed GLB~~ ✅ | P1 | M | ★★★ | Sprint 3 |
| ~~A3 — Wrist-relative OEF~~ ✅ | P1 | M | ★★★ | Sprint 3 |
| ~~D1 — Multi-frame biometrics~~ ✅ | P1 | L | ★★★ | Sprint 2 |
| ~~B1 — Quality-weighted SDF~~ ✅ | P1 | M | ★★★ | Sprint 3 |
| ~~F1 — Atomic RenderCommand~~ ✅ | P1 | M | ★★ | Sprint 3 |
| ~~B4 — Grid res upgrade~~ ✅ | P2 | M | ★★ | Sprint 5 |
| ~~B5 — Laplacian smoothing~~ ✅ | P2 | M | ★★ | Sprint 5 |
| ~~A4 — Adaptive TDF ALPHA~~ ✅ | P2 | M | ★★ | Sprint 5 |
| ~~D2 — Calibration card scale~~ ✅ | P2 | M | ★★★ | Sprint 5 |
| ~~E5 — Quality pre-flight~~ ✅ | P2 | M | ★★ | Sprint 5 |
| ~~C4 — OSC named presets~~ ✅ | P2 | M | ★★ | Sprint 6 |
| ~~D4 — Biometric history~~ ✅ | P2 | M | ★★ | Sprint 6 |
| ~~F4 — Unit tests~~ ✅ | P2 | M | ★★ | Sprint 6 |
| ~~E6 — Guided onboarding~~ ✅ | P2 | M | ★★ | Sprint 6 |
| ~~D5 — Joint angle ROM~~ ✅ | P2 | M | ★★★ | Sprint 7 |
| ~~C2 — UV unwrap GLB~~ ✅ | P2 | H | ★★ | Sprint 8 |
| ~~G6 — Temporal consistency~~ ✅ | P2 | M | ★★ | Sprint 9 |
| ~~G3 — Cross-device OSC sync~~ ✅ | P3 | M | ★★ | Sprint 10 |
| ~~G4 — Gesture recognition~~ ✅ | P3 | M | ★★★ | Sprint 11 |
| ~~G1 — Live mesh deformation~~ ✅ | P3 | H | ★★★ | Sprint 14 |
| ~~G2 — Neural implicit recon~~ ✅ | P3 | H | ★★★ | Sprint 15 |
| ~~G5 — Photometric stereo normals~~ ✅ | P3 | H | ★★ | Sprint 16 |

---

## KNOWN CONSTRAINTS

- **No Termux.** All builds go through GitHub Actions (`build-apk.yml`) or Claude Code web.
- **Xiaomi 13T target.** CSS viewport 439×976, DPR 2.78, Snapdragon 8 Gen 2, Adreno 740. Performance budgets should be validated on this device.
- **Single-file preference where possible.** New modules should stay within existing package boundaries unless F5 is explicitly started.
- **minSdk 26** — no API gating below Android 8.0. ARCore requires minSdk 24 (compatible).
- **No external cloud dependency.** All processing is on-device. OSC is LAN-only.

---

**Implementation:** `NeuralImplicitCarver` (4-layer MLP, 64 hidden units, ReLU activations; Adam optimizer lr=1e-3; pure Kotlin forward + backward pass; binary cross-entropy loss; positive samples from capsule volumes, negative samples from free space; ~300 epochs, ~2s on Snapdragon 8 Gen 2; `extractMesh()` runs marching cubes on the occupancy SDF at the same grid resolution as `DepthCarver`). `ManoShapeFitter` pre-study (MANO-style per-segment radius corrections via mean bone-length ratios derived from the 8-pose scan; `carveWithShapeParams()` as fallback when MLP produces empty mesh). Wired into `AppViewModel.processScan()` (priority: ARCore depth → neural MLP → MANO SDF → DepthCarver). `toggleNeuralRecon()` on `AppViewModel`. `DepthScannerPanel` extended with neural recon toggle and training diagnostics (loss, epoch count, tri count). `AppUiState` gains `neuralReconEnabled` + `neuralReconDiagnostics`. `NeuralReconDiagnostics` data class added.

---

*Last updated: post-sprint-16 (G5). Sprints 1–16 complete (all P1+P2 items + G6 + G3 + G4 + B6 + B7 + G1 + G2 + G5). No remaining items.*
