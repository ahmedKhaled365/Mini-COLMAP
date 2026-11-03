
#include "Geometry.h"
#include <stdexcept>

using namespace std;
using namespace cv;

PoseResult Geometry::estimateInitialPose(
    const ImageData& img1,
    const ImageData& img2,
    const vector<DMatch>& matches,
    const Mat& K
) const {

    if (matches.size() < 8) {
        throw runtime_error("Not enough matches to estimate pose");
    }

    vector<Point2f> points1, points2;

    /// converting the matches into pixle coordinates (u, v).
    for (auto match : matches) {
        points1.push_back(img1.keypoints[match.queryIdx].pt);
        points2.push_back(img2.keypoints[match.trainIdx].pt);
    }

    Mat essentialMask;

    Mat essential = findEssentialMat(
        points1,
        points2,
        K,
        RANSAC,
        0.999,
        1.0,
        1000,
        essentialMask
    );

    if (essential.empty() || essentialMask.empty()) {
        throw runtime_error("Failed to calculate Essential Matrix");
    }

    // The 5-point solver may return several stacked solutions; use the best one.
    if (essential.rows > 3)
        essential = essential.rowRange(0, 3).clone();

    PoseResult result;
    result.essentialInliers = countNonZero(essentialMask);

    // Pose from the Essential Matrix. The standard overload is used: the
    // former variant with distanceThresh = 1e6 counted points at infinity
    // as cheirality inliers, which made near-zero-baseline pairs look strong.
    // Cheirality, reprojection and parallax are enforced afterwards by
    // Triangulation::triangulate on every E-RANSAC inlier.
    Mat poseMask = essentialMask.clone();

    int inliers = recoverPose(
        essential,
        points1,
        points2,
        K,
        result.R,
        result.t,
        poseMask
    );

    if (inliers < 15) {
        throw runtime_error("Not enough inliers for initial pose");
    }

    for (int i = 0; i < static_cast<int>(matches.size()); i++) {
        if (essentialMask.at<uchar>(i)) {
            result.inlierMatches.push_back(matches[i]);
        }
    }

    // Homography test: a pair explained almost as well by a homography is
    // rotation-dominated or planar, so its relative translation is unreliable.
    Mat homographyMask;
    Mat H = findHomography(points1, points2, RANSAC, 2.0, homographyMask, 2000, 0.999);

    if (!H.empty() && !homographyMask.empty())
        result.homographyInliers = countNonZero(homographyMask);

    return result;
}



