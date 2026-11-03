#pragma once

#include <opencv2/opencv.hpp>

#include "ImageData.h"
#include "Triangulation.h"

struct BAOptions {
    string label = "global";

    int maxIterations = 100;
    double huberScale = 2.0; // px

    // Ceres threads. 1 makes results bit-reproducible; more threads change
    // the floating-point summation order between runs.
    int numThreads = 1;

    // Shared intrinsics refinement (global BA only)
    bool refineFocal = false;
    bool refineK1 = false;

    // Reject the update if the focal length would change more than this
    double maxFocalChange = 0.2;
};

struct BAResult {
    string label;
    int registered = 0;
    int variableCameras = 0;
    int constantCameras = 0;
    int points = 0;
    int residuals = 0;

    double initialCost = 0;
    double finalCost = 0;
    double rmsBefore = 0; // unrobustified reprojection RMS [px]
    double rmsAfter = 0;
    int iterations = 0;
    string termination;
    double seconds = 0;

    double focalBefore = 0, focalAfter = 0;
    double k1Before = 0, k1After = 0;

    bool usable = false;
    bool applied = false;
    string reason;

    // Filled by the caller after outlier filtering
    int filteredObservations = 0;
    int removedLandmarks = 0;
};

class BundleAdjuster {
public:
    // Optimizes the cameras in variableCameras and every landmark in
    // variablePoints. All other registered cameras observing those
    // landmarks enter the problem as constants.
    //
    // Gauge: anchor1 is always held constant and one translation component
    // of anchor2 is fixed (scale), which removes exactly 7 degrees of freedom.
    //
    // Observations are the raw (distorted) keypoints, projected with the
    // shared intrinsics (f, cx, cy, k1). Results are written back only if
    // the solution is usable and the reprojection RMS did not increase.
    static BAResult optimize(
        vector<ImageData> &images,
        vector<Landmark> &landmarks,
        Mat &K,
        double &k1,
        const set<int> &variableCameras,
        const set<int> &variablePoints,
        int anchor1,
        int anchor2,
        const BAOptions &options
    );

    static void writeLog(
        const vector<BAResult> &log,
        const string &path
    );
};
