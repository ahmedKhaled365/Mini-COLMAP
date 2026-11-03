#pragma once

#include "ImageData.h"
#include "Triangulation.h"
#include "Diagnostics.h"


class Reconstruction {
private:
    vector<ImageData> &images;
    vector<Landmark> landmarks;
    vector<vector<int> > observationToLandmark;

    // One entry per PnP registration attempt
    vector<RegistrationRecord> registrationLog;

    struct CorrespondenceInfo {
        int landmarkId;
        int keypointId;
    };

public:
    explicit Reconstruction(vector<ImageData> &images);

    void printTrackStatistics() const;

    bool runBundleAdjustment(const Mat &K);

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
