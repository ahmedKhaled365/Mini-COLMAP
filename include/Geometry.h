
#pragma once

#include "ImageData.h"
#include <opencv2/calib3d.hpp>

struct PoseResult {
    Mat R;
    Mat t;
    vector<DMatch> inlierMatches;

    // Two-view model statistics used for initial pair selection
    int essentialInliers = 0;
    int homographyInliers = 0;
};

class Geometry {
public:
    PoseResult estimateInitialPose(
        const ImageData& img1,
        const ImageData& img2,
        const vector<DMatch>& matches,
        const Mat& K
    ) const;
};
