#pragma once

#include "ImageData.h"
#include "Geometry.h"
#include <opencv2/calib3d.hpp>

struct Observation {
    int imageId;
    int keypointId;
};

// Statistics of one two-view triangulation, measured before the
// triangulation-angle filter so that weak-parallax pairs are visible.
struct TwoViewStats {
    int candidates = 0;   // input correspondences
    int triangulated = 0; // positive depth + good reprojection
    double medianAngle = 0; // over triangulated points [deg]
};

struct Landmark {
    Point3d position;
    vector<Observation> observations;
};

class Triangulation {
public:
    vector<Landmark> triangulate(
        ImageData &img1,
        ImageData &img2,
        PoseResult &pose,
        Mat &K,
        TwoViewStats *stats = nullptr
    );

    static bool hasPositiveDepth(
        const Point3d &X,
        const Mat &R,
        const Mat &t
    );

    static bool hasGoodReprojection(
        const cv::Point3d &X,
        const cv::Point2f &pt1,
        const cv::Point2f &pt2,
        const cv::Mat &P1,
        const cv::Mat &P2,
        double maxError = 3.0
    );

    static bool hasGoodAngle(
        const cv::Point3d &X,
        const cv::Mat &R1,
        const cv::Mat &t1,
        const cv::Mat &R2,
        const cv::Mat &t2,
        double minAngle = 1.0
    );
};
