#pragma once

#include "ImageData.h"
#include "Triangulation.h"
#include "Diagnostics.h"
#include "BundleAdjuster.h"


class Reconstruction {
private:
    vector<ImageData> &images;
    vector<Landmark> landmarks;
    vector<vector<int> > observationToLandmark;

    // One entry per PnP registration attempt
    vector<RegistrationRecord> registrationLog;

    // Initial pair, used as the gauge anchors of bundle adjustment
    int anchor1 = -1;
    int anchor2 = -1;

    // Shared radial distortion coefficient (keypoints[].pt is undistorted)
    double k1 = 0.0;

    vector<BAResult> baLog;

    // Remove deleted landmarks and rebuild observationToLandmark
    void compactLandmarks(const vector<bool> &keep);

    struct CorrespondenceInfo {
        int landmarkId;
        int keypointId;
    };

public:
    explicit Reconstruction(vector<ImageData> &images);

    void printTrackStatistics() const;

    bool runBundleAdjustment(Mat &K);

    // Bundle adjustment of the new image and its best-connected neighbors.
    BAResult localBundleAdjustment(
        int imageId,
        Mat &K,
        const BAOptions &options,
        int numNeighbors
    );

    // Bundle adjustment of all registered cameras and landmarks.
    BAResult globalBundleAdjustment(Mat &K, const BAOptions &options);

    // Removes observations with reprojection error above maxError (or
    // behind the camera) and landmarks left with fewer than two
    // observations or a maximum triangulation angle below minAngle.
    // Only landmarks in candidates are checked (all if empty).
    // Returns {removed observations, removed landmarks}.
    pair<int, int> filterLandmarks(
        const set<int> &candidates,
        const Mat &K,
        double maxError,
        double minAngle
    );

    // Recompute keypoints[].pt from rawPoints with the current K and k1.
    void updateUndistortedKeypoints(const Mat &K);

    double getK1() const {
        return k1;
    }

    vector<BAResult> &getBALog() {
        return baLog;
    }

    int countCorrespondences(
        int imageId,
        vector<PairMatch> &verifiedPairs
    );

    size_t getLandmarkCount() const {
        return landmarks.size();
    }

    const vector<Landmark> &getLandmarks() const {
        return landmarks;
    }

    const vector<ImageData> &getImages() const {
        return images;
    }

    vector<RegistrationRecord> &getRegistrationLog() {
        return registrationLog;
    }

    bool exportFilteredPLY(
        const string &filename,
        int minObservations
    ) const;

    int triangulateNewPoints(
        int imageId,
        const vector<PairMatch> &verifiedPairs,
        const Mat &K
    );

    void findCorrespondences(
        int imageId,
        vector<PairMatch> &verifiedPairs,
        vector<Point3f> &points3D,
        vector<Point2f> &points2D,
        vector<CorrespondenceInfo> &correspondenceInfo
    );

    void initialize(
        int imageId1,
        int imageId2,
        PoseResult &pose,
        vector<Landmark> &initialLandmarks
    );

    bool registerImage(
        int imageId,
        vector<PairMatch> &verifiedPairs,
        Mat &K
    );

    bool exportPLY(const string &filename) const;
};
