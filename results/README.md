# Mini-COLMAP experiment results (20 images, no bundle adjustment)

All runs use images `00001`–`00020`, SIFT 3500, the same matching settings, and no BA.
Runs are deterministic: repeating a command reproduces `report.txt` and `points_all.ply` byte-for-byte.

Each run folder contains:

| File | Content |
|---|---|
| `report.txt` | Configuration, diagnostics summary, track histogram, per-camera RMS, camera centres |
| `checkpoints.csv` | Diagnostics after 2 / 5 / 10 / 20 registered cameras |
| `registration.csv` | Every PnP attempt: correspondences, inliers, ratio, inlier RMS, new landmarks |
| `init_candidates.csv` | Phase 2 runs only: every initial-pair candidate with E/H inliers, median angle, gate result |
| `points_*.ply` | Coloured point clouds: all, 3+ views, 4+ views, reliable, unreliable |
| `cameras.ply`, `scene_with_cameras.ply` | Camera frustums + trajectory (edges), optionally with the points |
| `checkpoint_NNN_cameras.ply` | Coloured cloud at each checkpoint |
| `preview.png` | Orthographic top / front / side renders (reliable points drawn larger) |

PLY / PNG / log files are local only (git-ignored); the reports and CSVs are committed.

## Metric definitions

* **Reliable** landmark: ≥3 observations, every observation ≤2 px reprojection error, max pairwise ray angle ≥2°.
* **Object region**: landmarks within 0.35 × the median camera distance of the least-squares intersection of all optical axes. All cameras look at the statue, so this is a rough proxy for statue + pedestal points, not a segmentation.
* **Angle**: per landmark, the largest angle between two observing rays.
* **Camera step CV**: std/mean of distances between consecutive camera centres. It's scale-invariant; lower means a smoother trajectory.
* Every metric is scale-invariant, so runs with different gauges (initial pairs) can be compared.
* Reprojection error alone is **not** a quality measure. Two-view points fit their two observations almost exactly by construction.

## Comparison

| Run | Code | f [px] | Init | Pair (median angle) | Landmarks | Reliable | Object region (reliable) | Mean track | 2-view % | Angle median [deg] | Reproj median / P95 [px] | PnP inlier ratio mean / min | Camera step CV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `B0_baseline_f1344` | Phase 1 | 1344 | legacy | 1-2 (1.54) | 6860 | 1354 | 991 (146) | 2.71 | 63.0 | 2.88 | 0.546 / 2.655 | 0.697 / 0.548 | 0.463 |
| `A_focal_1200` | Phase 1 | 1200 | legacy | 4-8 (1.69) | 5660 | 1198 | 822 (208) | 2.87 | 59.1 | 2.84 | 0.597 / 2.600 | 0.753 / 0.439 | 0.568 |
| `A_focal_1133` | Phase 1 | 1133 | legacy | 4-8 (2.32) | 4870 | 875 | 718 (193) | 2.80 | 61.9 | 2.31 | 0.632 / 2.547 | 0.678 / 0.402 | 0.736 |
| `A_focal_1069` | Phase 1 | 1069 | legacy | 4-8 (2.48) | 3715 | 749 | 1 (0) | 2.95 | 59.5 | 2.53 | 0.625 / 2.632 | 0.602 / 0.381 | 1.052 |
| `AB_f1133_pair1-2` | Phase 1 | 1133 | fixed 1-2 | 1-2 (2.79) | 3680 | 660 | 7 (0) | 2.84 | 58.4 | 3.42 | 1.076 / 2.780 | 0.655 / 0.428 | 0.662 |
| `AB_f1200_pair1-2` | Phase 1 | 1200 | fixed 1-2 | 1-2 (2.93) | 5487 | 1051 | 941 (155) | 2.83 | 58.2 | 2.98 | 0.739 / 2.746 | 0.749 / 0.585 | 0.595 |
| `AB_f1344_pair4-8` | Phase 1 | 1344 | fixed 4-8 | 4-8 (1.86) | 5544 | 1124 | 887 (231) | 2.91 | 56.1 | 2.73 | 0.612 / 2.592 | 0.733 / 0.568 | 0.552 |
| `L_legacy_new_pose_f1344` | Phase 2 | 1344 | legacy | 1-2 (1.54) | 6860 | 1354 | 991 (146) | 2.71 | 63.0 | 2.88 | 0.546 / 2.655 | 0.697 / 0.548 | 0.463 |
| `P_parallax_f1344` | Phase 2 | 1344 | parallax | 2-6 (5.68) | 6189 | 1217 | 1101 (128) | 2.74 | 61.1 | 2.94 | 0.562 / 2.568 | 0.732 / 0.568 | 0.514 |
| `P_parallax_f1250` | Phase 2 | 1250 | parallax | 2-6 (4.86) | 5935 | 1554 | 1070 (172) | 2.87 | 58.1 | 2.88 | 0.397 / 2.270 | 0.851 / 0.684 | 0.509 |
| `P_parallax_f1200` | Phase 2 | 1200 | parallax | 2-6 (4.43) | 5536 | 1480 | 956 (195) | 2.87 | 56.8 | 2.59 | 0.437 / 2.160 | 0.835 / 0.612 | 0.500 |
| `P_parallax_f1150` | Phase 2 | 1150 | parallax | 2-6 (4.00) | 4898 | 928 | 873 (197) | 2.79 | 61.9 | 2.46 | 0.523 / 2.300 | 0.750 / 0.533 | 0.557 |
| `P_parallax_f1133` | Phase 2 | 1133 | parallax | 2-6 (3.85) | 3835 | 826 | 571 (167) | 2.88 | 60.6 | 2.48 | 0.563 / 2.358 | 0.737 / 0.518 | 0.555 |
| `P_parallax_f1069` | Phase 2 | 1069 | parallax | 2-6 (3.34) | 3056 | 603 | 558 (138) | 2.95 | 58.9 | 3.25 | 0.665 / 2.571 | 0.705 / 0.390 | 0.637 |

* "Pair (median angle)" = selected initial pair and the median triangulation angle of its initial landmarks.
* `AB_f1200_pair4-8` and `AB_f1133_pair4-8` are identical to `A_focal_1200` and `A_focal_1133` (the legacy selector chose 4-8 there too), so they're omitted.
* `_original_baseline_run/` holds the console log and PLYs from the original, unmodified binary (tag `baseline-20img`).

## Focal-length / distortion sweep (`calib_sweep_20img/`)

* **What it measures:** the mean fraction of F-verified matches explained by an Essential matrix (RANSAC, 1 px) over the 39 pairs with frame gap ≥3. Each k1 first undistorts the keypoints with a one-parameter radial model.
* **Result:**
  * Peak at **f = 1200, k1 = −0.02** (0.945).
  * With no distortion (k1 = 0), the peak is 1150–1200 (0.92), against 0.892 at the original 1344.
* **Caveat:** the F-RANSAC filter (in distorted coordinates) runs first, which weakens the sensitivity to k1.

## EXIF

* Sony A7S II, `FocalLength` = 21 mm, `FocalLengthIn35mmFilm` = 21 mm, 1920×1080.
* Full-frame sensor width 35.6 mm → f ≈ 21 / 35.6 × 1920 ≈ **1133 px**.
* The 35 mm diagonal rule would give ≈ 1069 px.
* The data prefers ~1200 px (+6%). Possible causes (not verified): nominal focal rounding on a zoom lens, focus breathing, or a sensor crop from stabilisation or video mode.

## Reproduce

Run from `cmake-build-*/`:

```
Mini-COLMAP.exe --ba none --focal 1344 --init legacy --out ../results/L_legacy_new_pose_f1344
Mini-COLMAP.exe --ba none --focal 1200 --out ../results/P_parallax_f1200
Mini-COLMAP.exe --ba none --focal 1133 --pair 1 2 --out ../results/AB_f1133_pair1-2   # Phase 1 code: git checkout f982ca4
Mini-COLMAP.exe --focal-sweep --out ../results/calib_sweep_20img
```

Baseline B0 and the `A_*` / `AB_*` runs were made with the Phase 1 commit (diagnostics only, original initializer).

---

# Phase 3: Bundle adjustment (20 images)

**Shared settings for every Phase 3 run:**
- parallax initializer;
- Huber loss 2 px;
- SPARSE_SCHUR, Levenberg–Marquardt, 1 Ceres thread (bit-reproducible);
- post-BA filter: drop observations above 4 px or behind the camera, then landmarks with fewer than 2 views or under 1° maximum angle (the same angle as at creation).

| Run | BA | Intrinsics |
|---|---|---|
| `BA1_final` | one global BA after all registrations | fixed f = 1200 |
| `BA2_incremental_fXXXX` | local BA (new image + 6 best-connected) after every registration, global BA after init and every +25% cameras, final global BA | fixed f = XXXX |
| `BA3_incr_focal_fXXXX` | as BA2 | shared f refined in global BA (≥ 8 cameras), starting at XXXX |
| `BA4_incr_focal_k1_fXXXX` | as BA2 | shared f and k1 refined, starting at XXXX (**current default**, start 1200) |

| run | reg | landmarks | reliable | object | object_rel | track | two_view_pct | angle_med | rp_med | rp_p95 | pnp_mean | pnp_min | step_cv | focal | k1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0_baseline_f1344 | 20 | 6860 | 1354 | 991 | 146 | 2.71 | 63.0 | 2.88 | 0.546 | 2.655 | 0.697 | 0.548 | 0.463 | 1344 | 0 |
| P_parallax_f1133 | 20 | 3835 | 826 | 571 | 167 | 2.88 | 60.6 | 2.48 | 0.563 | 2.358 | 0.737 | 0.518 | 0.555 | 1133 | 0 |
| P_parallax_f1200 | 20 | 5536 | 1480 | 956 | 195 | 2.87 | 56.8 | 2.59 | 0.437 | 2.160 | 0.835 | 0.612 | 0.500 | 1200 | 0 |
| P_parallax_f1344 | 20 | 6189 | 1217 | 1101 | 128 | 2.74 | 61.1 | 2.94 | 0.562 | 2.568 | 0.732 | 0.568 | 0.514 | 1344 | 0 |
| BA1_final | 20 | 4007 | 1296 | 969 | 347 | 3.13 | 45.3 | 2.38 | 0.334 | 1.052 | 0.835 | 0.612 | 0.497 | 1200 | 0 |
| BA2_incremental_f1133 | 20 | 2606 | 957 | 795 | 346 | 3.04 | 55.4 | 3.61 | 0.297 | 1.186 | 0.899 | 0.778 | 0.488 | 1133 | 0 |
| BA2_incremental_f1200 | 20 | 3818 | 1315 | 897 | 377 | 2.97 | 56.5 | 2.55 | 0.308 | 1.041 | 0.896 | 0.743 | 0.499 | 1200 | 0 |
| BA2_incremental_f1344 | 20 | 5425 | 2046 | 1048 | 413 | 2.85 | 59.0 | 3.05 | 0.432 | 1.397 | 0.822 | 0.638 | 0.512 | 1344 | 0 |
| BA3_incr_focal_f1133 | 20 | 3869 | 1490 | 915 | 401 | 2.91 | 57.6 | 2.73 | 0.291 | 1.015 | 0.920 | 0.760 | 0.500 | 1213.73 | 0 |
| BA3_incr_focal_f1200 | 20 | 3953 | 1419 | 926 | 390 | 2.95 | 57.3 | 2.63 | 0.308 | 1.043 | 0.901 | 0.721 | 0.499 | 1213.36 | 0 |
| BA3_incr_focal_f1344 | 20 | 4005 | 1423 | 949 | 390 | 3.04 | 55.3 | 2.65 | 0.331 | 1.064 | 0.895 | 0.638 | 0.499 | 1220.77 | 0 |
| BA4_incr_focal_k1_f1133 | 20 | 3648 | 1283 | 853 | 357 | 2.92 | 58.1 | 2.63 | 0.263 | 0.944 | 0.906 | 0.765 | 0.497 | 1168.97 | -0.0263606 |
| BA4_incr_focal_k1_f1200 | 20 | 3643 | 1264 | 846 | 356 | 3.00 | 56.2 | 2.62 | 0.276 | 0.994 | 0.926 | 0.795 | 0.496 | 1167.12 | -0.0279331 |
| BA4_incr_focal_k1_f1344 | 20 | 3691 | 1245 | 863 | 343 | 3.04 | 56.3 | 2.65 | 0.294 | 0.982 | 0.904 | 0.638 | 0.495 | 1167.67 | -0.0293948 |

`focal` / `k1` are the final values after refinement. The `ba_log.csv` in each run lists every BA call: cameras (variable + constant), points, residuals, RMS before/after, iterations, termination, time, intrinsics, filter counts.

## Findings

* **Gauge and convergence:**
  * Every one of the 28 BA calls per incremental run ends in `CONVERGENCE` and is applied.
  * The old implementation hit `NO_CONVERGENCE` and had hardcoded anchors 6, 7.
  * Global problems have exactly one constant camera (anchor 1), plus scale fixed through one translation component of anchor 2.
* **Incremental vs. final-only BA:**
  * Final-only BA (BA1) cleans the cloud: P95 drops from 2.16 to 1.05 px and reliable statue points rise from 195 to 347.
  * It can't improve registration, which happened beforehand: the PnP ratio stays 0.835 / 0.612.
  * Incremental BA raises the PnP inlier ratio to about 0.90 mean / 0.74–0.80 min, so new images register against a refined map.
* **Self-calibration:**
  * Without intrinsics refinement (BA2), the result still depends strongly on the starting focal length: 2,606–5,425 landmarks and 957–2,046 reliable for starts 1133–1344.
  * Refining f (BA3) converges to 1213–1221 px from all three starts.
  * Refining f + k1 (BA4) converges to **f = 1167–1169 px, k1 = −0.026 … −0.029** from all three starts. Landmark counts then agree within 1.3%, and reliable-statue counts are 343–357.
* **Why f + k1 is the default:**
  * f and k1 are correlated. Ignoring barrel distortion inflates the apparent focal length.
  * With k1 modelled, f lands within 3% of the EXIF value (1133 px).
  * k1 matches the independent Essential-matrix sweep (−0.02).
  * Its reprojection error is lowest (P95 0.94–0.99 px).
  * BA3 keeps somewhat more reliable points (1419–1490 vs 1245–1283), so the choice rests on physical consistency, not on counts.
* **Outlier filtering:**
  * The 4 px observation filter never fired: every observation was already within 4 px after PnP (4 px) and triangulation (3 px).
  * All removals come from the 1° angle filter. Weakly constrained two-view points slide in depth once cameras are refined.
  * Fewer landmarks with a higher reliable fraction is the intended result: 27% reliable without BA, 35% with BA4.
* **Determinism:**
  * With multiple Ceres threads, repeated runs differ slightly (landmarks ±0.1%, but the worst-image PnP ratio 0.69 ↔ 0.76).
  * `--ba-threads 1` (default) is bit-reproducible.
* **Cost:** about 12 s of BA in total for 20 images, single-threaded.

## Reproduce (Phase 3)

```
Mini-COLMAP.exe --out ../results/BA4_incr_focal_k1_f1200                       # = current default
Mini-COLMAP.exe --focal 1133 --out ../results/BA4_incr_focal_k1_f1133
Mini-COLMAP.exe --no-refine-k1 --out ../results/BA3_incr_focal_f1200
Mini-COLMAP.exe --no-refine-focal --no-refine-k1 --out ../results/BA2_incremental_f1200
Mini-COLMAP.exe --ba final --no-refine-focal --no-refine-k1 --out ../results/BA1_final
python ../tools/compare_runs.py BA1_final BA4_incr_focal_k1_f1200              # from results/
```

---

# Output size and dataset expansion

## Output levels (`--output`, default `standard`)

| Level | Files |
|---|---|
| `minimal` | `report.txt`, `checkpoints.csv`, `registration.csv`, `init_candidates.csv`, `ba_log.csv`, `preview.png` |
| `standard` | + `points_all.ply`, `points_reliable.ply`, `cameras.ply` |
| `full` | + `points_3plus`, `points_4plus`, `points_unreliable`, `scene_with_cameras`, `checkpoint_NNN_cameras.ply` |

* **PLY encoding:** binary little-endian by default; `--ascii-ply` restores text files. The binary files decode to the same points, colours and edges as ASCII (max relative difference 5e-7, i.e. float precision).
* **Size:** a 20-image run at `standard` is about 260 KB, against about 1 MB before.
* **Cleanup:** PLY/PNG/log files were removed from superseded runs. They were kept for `B0_baseline_f1344`, `P_parallax_f1200`, `BA1_final`, `BA4_incr_focal_k1_f1200` and `_original_baseline_run`. Every report and CSV remains.
* `results/latest/` is rewritten by each default run and is not versioned.

## Expansion baseline: current default at 20 / 35 / 50 images (before Phase 4)

| run | reg | landmarks | reliable | object | object_rel | track | two_view_pct | angle_med | rp_med | rp_p95 | pnp_mean | pnp_min | step_cv | focal | k1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BA4_incr_focal_k1_f1200 | 20 | 3643 | 1264 | 846 | 356 | 3.00 | 56.2 | 2.62 | 0.276 | 0.994 | 0.926 | 0.795 | 0.496 | 1167.12 | -0.0279331 |
| E35_default | 35 | 6423 | 2297 | 1667 | 634 | 2.97 | 56.1 | 3.19 | 0.289 | 1.052 | 0.908 | 0.792 | 0.374 | 1164.44 | -0.0300325 |
| E50_default | 50 | 8321 | 3006 | 1985 | 673 | 2.96 | 56.7 | 3.33 | 0.295 | 1.075 | 0.902 | 0.771 | 0.323 | 1165.98 | -0.029172 |

**Milestone gates:**

| Gate | 35 images | 50 images |
|---|---|---|
| All images registered | pass | pass |
| Worst PnP inlier ratio ≥ 0.6 | 0.79 | 0.77 |
| Reprojection P95 ≤ 1.5 px | 1.05 | 1.08 |
| All BA calls converged and applied | 45 / 45 | 62 / 62 |
| Focal length stable vs previous milestone | 1167 → 1164 | 1166 |
| Smooth trajectory in `preview.png` | pass | pass |

**Observations:**
* k1 drifts slightly (−0.028 → −0.030 → −0.029). That's small in absolute terms, but more than the 2% rule of thumb.
* At 50 images the cameras cover about 270° of the orbit. The trajectory is a smooth arc, apart from a tight cluster at frames 0–8 where the camera paused.
* Quality holds steady as images are added:
  * reliable fraction 35% / 36% / 36%;
  * mean track length 3.0;
  * two-view share 56–57%;
  * camera step CV falls 0.50 → 0.37 → 0.32.
* The 20-camera checkpoint inside the 35/50-image runs (3,644 landmarks) reproduces the standalone 20-image result (3,643): adding images doesn't disturb the early reconstruction.
* **Time:** 87 s for 35 images and 111 s for 50, both dominated by feature extraction and matching. BA takes 4–6 s.
