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

    // Correspondence graph from the verified pairs:
    // graph[image][keypoint] -> list of (other image, other keypoint)
    vector<vector<vector<pair<int, int> > > > correspondenceGraph;

    // Reprojection error of landmark position X in image/keypoint, or
    // infinity if X is behind the camera.
    double reprojectionError(const Point3d &X, int imageId, int keypointId, const Mat &K) const;

public:
    // Cumulative track statistics for reporting
    struct TrackStats {
        int completedObservations = 0;
        int mergedLandmarks = 0;
        int retriangulatedLandmarks = 0;
    };

private:
    TrackStats trackStats;

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

    // Build correspondenceGraph from the verified pairs (call once).
    void buildCorrespondenceGraph(const vector<PairMatch> &verifiedPairs);

    // Extend tracks through the correspondence graph: an unassigned
    // keypoint matched to an observation of a landmark is added to it if the
    // landmark has no observation in that image and reprojects within
    // maxError. Repeats until no observation is added. Returns the number
    // of added observations.
    int completeTracks(const Mat &K, double maxError);

    // Merge landmarks connected by a correspondence when they share no image
    // and the merged point reprojects within maxError in every observation.
    // Returns the number of merges.
    int mergeTracks(const Mat &K, double maxError);

    // Re-run two-view triangulation for every pair of registered images
    // (used after global BA, when poses are more accurate).
    int retriangulate(const vector<PairMatch> &verifiedPairs, const Mat &K);

    // Counts violations of the track invariants: an observation owned by
    // more than one landmark, two observations of a landmark in one image,
    // or observationToLandmark disagreeing with the landmarks.
    int countInvariantViolations() const;

    const TrackStats &getTrackStats() const {
        return trackStats;
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
