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
Mini-COLMAP.exe --focal 1344 --init legacy --out ../results/L_legacy_new_pose_f1344
Mini-COLMAP.exe --focal 1200 --out ../results/P_parallax_f1200      # = current default
Mini-COLMAP.exe --focal 1133 --pair 1 2 --out ../results/AB_f1133_pair1-2
Mini-COLMAP.exe --focal-sweep --out ../results/calib_sweep_20img
```

Baseline B0 and the `A_*` / `AB_*` runs were made with the Phase 1 commit (diagnostics only, original initializer).
