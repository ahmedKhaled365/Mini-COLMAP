#pragma once

#include "ImageData.h"
#include "Triangulation.h"

// --------------------------------------------------
// Diagnostics: measurements and exports that describe the
// geometric quality of a reconstruction. Nothing in here
// modifies cameras or landmarks.
// --------------------------------------------------

struct Stats {
    size_t count = 0;
    double mean = 0, median = 0, p95 = 0, max = 0, rms = 0;
};

Stats computeStats(vector<double> values);

// One PnP registration attempt (successful or not).
struct RegistrationRecord {
    int imageId = -1;
    int attempt = 0; // global attempt counter
    int correspondences = 0;
    int inliers = 0;
    double inlierRatio = 0;
    double inlierRms = 0; // reprojection RMS of PnP inliers [px]
    int addedObservations = 0;
    int newLandmarks = 0;
    bool registered = false;
    string reason;
};

// One evaluated initial-pair candidate.
struct InitCandidateRecord {
    int idx1 = -1, idx2 = -1;
    int fInliers = 0;
    int eInliers = 0;
    int hInliers = 0;
    double hOverE = 0;
    int triangulated = 0; // positive depth + good reprojection
    int landmarks = 0;    // also passes the angle filter
    double medianAngle = 0;
    double score = 0;
    bool accepted = false;
    string reason;
};

struct DiagnosticsConfig {
    // Quality classes for the reliable/unreliable PLY split
    int reliableMinTrack = 3;
    double reliableMaxError = 2.0; // px, worst observation
    double reliableMinAngle = 2.0; // deg, max pairwise ray angle

    // Object region radius relative to median camera-to-lookAt distance
    double objectRegionFactor = 0.35;

    // PLY encoding: binary little-endian (default) or ASCII
    bool binaryPly = true;
};

struct CameraStats {
    int imageId = -1;
    int observations = 0;
    double rms = 0;
    Point3d center;
};

struct ReconstructionReport {
    int registeredImages = 0;
    int totalImages = 0;
    size_t landmarks = 0;
    size_t observations = 0;
    int negativeDepthObservations = 0;

    Stats reprojection;   // all observations [px]
    Stats triAngle;       // per landmark, max pairwise angle [deg]
    Stats trackLength;
    map<int, int> trackHistogram;
    array<int, 6> angleHistogram{}; // <1, 1-2, 2-5, 5-10, 10-20, >=20 deg

    vector<CameraStats> cameras;

    size_t reliableLandmarks = 0;

    // Scene geometry proxies (no ground truth available)
    Point3d lookAt;                  // least-squares intersection of optical axes
    double medianCameraDistance = 0; // camera centers to lookAt
    size_t objectRegionLandmarks = 0;
    size_t objectRegionReliable = 0;
    Stats objectRegionReprojection;

    // Trajectory consistency along the image sequence
    Stats cameraStep;             // distance between consecutive registered ids
    double cameraStepCv = 0;      // std / mean of steps
};

class Diagnostics {
public:
    explicit Diagnostics(DiagnosticsConfig config = {});

    ReconstructionReport analyze(
        const vector<ImageData> &images,
        const vector<Landmark> &landmarks,
        const Mat &K
    ) const;

    // Per-landmark quality helpers
    static double observationError(
        const ImageData &img,
        const Point3d &X,
        int keypointId,
        const Mat &K,
        double *depth = nullptr
    );

    static double maxTriangulationAngle(
        const vector<ImageData> &images,
        const Landmark &landmark
    );

    static Point3d cameraCenter(const ImageData &img);

    bool isReliable(
        const vector<ImageData> &images,
        const Landmark &landmark,
        const Mat &K
    ) const;

    // Text outputs
    static void printSummary(
        const ReconstructionReport &report,
        ostream &out
    );

    static void writeReport(
        const ReconstructionReport &report,
        const string &path,
        const string &header
    );

    static void appendCheckpointCsv(
        const ReconstructionReport &report,
        const string &path
    );

    static void writeRegistrationCsv(
        const vector<RegistrationRecord> &records,
        const string &path
    );

    static void writeInitCandidatesCsv(
        const vector<InitCandidateRecord> &records,
        const string &path
    );

    // PLY outputs (vertex colors sampled from the source photographs)
    // minTrack filters by number of observations; quality: 0 = all,
    // 1 = reliable only, 2 = unreliable only.
    bool exportPoints(
        const string &path,
        const vector<ImageData> &images,
        const vector<Landmark> &landmarks,
        const Mat &K,
        int minTrack = 2,
        int quality = 0
    ) const;

    // Camera centers, frustums and trajectory as a PLY with edges.
    // When withPoints is true, the colored landmarks are included too.
    bool exportCameras(
        const string &path,
        const vector<ImageData> &images,
        const vector<Landmark> &landmarks,
        const Mat &K,
        bool withPoints
    ) const;

    // Orthographic top / front / side renderings of the point cloud.
    bool renderPreview(
        const string &path,
        const vector<ImageData> &images,
        const vector<Landmark> &landmarks,
        const Mat &K,
        const string &title
    ) const;

    // Mean Essential-matrix inlier fraction over verified pairs for each
    // candidate focal length. A correct focal length should explain
    // more of the F-verified correspondences with a calibrated model.
    // Each k1 value undistorts the keypoints with a one-parameter radial
    // model before fitting, to test for unmodeled lens distortion.
    static void focalSweep(
        const vector<ImageData> &images,
        const vector<PairMatch> &verifiedPairs,
        const vector<double> &focals,
        const vector<double> &k1s,
        int minGap,
        const string &csvPath
    );

private:
    DiagnosticsConfig config;

    static Vec3b landmarkColor(
        const vector<ImageData> &images,
        const Landmark &landmark
    );

    static Point3d estimateLookAt(const vector<ImageData> &images);
};
