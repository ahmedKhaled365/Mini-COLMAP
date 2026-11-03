
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

    PoseResult result;

    Mat poseMask = essentialMask.clone();

    Mat triangulatedPoints;

    int inliers = recoverPose(
        essential,
        points1,
        points2,
        K,
        result.R,
        result.t,
        1e6,
        poseMask,
        triangulatedPoints
    );

    cout << "\n[Initial Pose Debug]\n";
    cout << "Pair: " << img1.id
         << " <-> " << img2.id << '\n';

    cout << "Input Matches: "
         << matches.size() << '\n';

    cout << "Essential RANSAC Inliers: "
         << countNonZero(essentialMask) << '\n';

    cout << "RecoverPose Inliers: "
         << inliers << '\n';

    if (inliers < 15) {
        throw runtime_error("Not enough inliers for initial pose");
    }

    for (int i = 0; i < matches.size(); i++) {
        if (poseMask.at<uchar>(i)) {
            result.inlierMatches.push_back(matches[i]);
        }
    }

    return result;
}



