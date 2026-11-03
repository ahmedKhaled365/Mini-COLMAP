#include <fstream>
#include <cmath>
#include "Reconstruction.h"
#include "BundleAdjuster.h"

Reconstruction::Reconstruction(vector<ImageData> &images) : images(images) {
}

bool Reconstruction::runBundleAdjustment(Mat &K) {
    BAOptions options;
    return globalBundleAdjustment(K, options).applied;
}

BAResult Reconstruction::localBundleAdjustment(
    int imageId,
    Mat &K,
    const BAOptions &options,
    int numNeighbors
) {
    // Rank registered images by the number of landmarks shared with imageId
    map<int, int> shared;

    for (int kp = 0; kp < static_cast<int>(observationToLandmark[imageId].size()); kp++) {
        int landmarkId = observationToLandmark[imageId][kp];
        if (landmarkId < 0)
            continue;

        for (const auto &obs: landmarks[landmarkId].observations) {
            if (obs.imageId != imageId && images[obs.imageId].registered)
                shared[obs.imageId]++;
        }
    }

    vector<pair<int, int> > ranked;
    for (const auto &[id, count]: shared)
        ranked.push_back({count, id});
    sort(ranked.begin(), ranked.end(), greater<>());

    set<int> variableCameras = {imageId};
    for (int i = 0; i < min(numNeighbors, static_cast<int>(ranked.size())); i++)
        variableCameras.insert(ranked[i].second);

    set<int> variablePoints;
    for (int l = 0; l < static_cast<int>(landmarks.size()); l++) {
        for (const auto &obs: landmarks[l].observations) {
            if (variableCameras.count(obs.imageId)) {
                variablePoints.insert(l);
                break;
            }
        }
    }

    BAResult result = BundleAdjuster::optimize(
        images, landmarks, K, k1,
        variableCameras, variablePoints,
        anchor1, anchor2, options
    );

    baLog.push_back(result);
    return result;
}

BAResult Reconstruction::globalBundleAdjustment(Mat &K, const BAOptions &options) {
    set<int> variableCameras, variablePoints;

    for (const auto &img: images) {
        if (img.registered)
            variableCameras.insert(img.id);
    }

    for (int l = 0; l < static_cast<int>(landmarks.size()); l++)
        variablePoints.insert(l);

    const double focal = K.at<double>(0, 0);
    const double distortion = k1;

    BAResult result = BundleAdjuster::optimize(
        images, landmarks, K, k1,
        variableCameras, variablePoints,
        anchor1, anchor2, options
    );

    if (result.applied &&
        (K.at<double>(0, 0) != focal || k1 != distortion))
        updateUndistortedKeypoints(K);

    baLog.push_back(result);
    return result;
}

void Reconstruction::updateUndistortedKeypoints(const Mat &K) {
    Mat dist = (Mat_<double>(1, 4) << k1, 0, 0, 0);

    for (auto &img: images) {
        if (img.rawPoints.empty())
            continue;

        vector<Point2f> undistorted = img.rawPoints;

        if (k1 != 0.0)
            undistortPoints(img.rawPoints, undistorted, K, dist, noArray(), K);

        for (size_t i = 0; i < undistorted.size(); i++)
            img.keypoints[i].pt = undistorted[i];
    }
}

pair<int, int> Reconstruction::filterLandmarks(
    const set<int> &candidates,
    const Mat &K,
    double maxError,
    double minAngle
) {
    int removedObservations = 0;
    vector<bool> keep(landmarks.size(), true);

    auto check = [&](int l) {
        auto &observations = landmarks[l].observations;
        vector<Observation> kept;

        for (const auto &obs: observations) {
            double depth = 0;
            double error = Diagnostics::observationError(
                images[obs.imageId], landmarks[l].position, obs.keypointId, K, &depth
            );

            if (depth > 0 && error <= maxError)
                kept.push_back(obs);
            else
                removedObservations++;
        }

        observations = kept;

        if (observations.size() < 2 ||
            Diagnostics::maxTriangulationAngle(images, landmarks[l]) < minAngle)
            keep[l] = false;
    };

    if (candidates.empty()) {
        for (int l = 0; l < static_cast<int>(landmarks.size()); l++)
            check(l);
    } else {
        for (int l: candidates)
            check(l);
    }

    int removedLandmarks = static_cast<int>(count(keep.begin(), keep.end(), false));
    compactLandmarks(keep);

    return {removedObservations, removedLandmarks};
}

void Reconstruction::compactLandmarks(const vector<bool> &keep) {
    vector<Landmark> compacted;
    compacted.reserve(landmarks.size());

    for (size_t l = 0; l < landmarks.size(); l++) {
        if (keep[l])
            compacted.push_back(move(landmarks[l]));
    }

    landmarks = move(compacted);

    for (auto &index: observationToLandmark)
        fill(index.begin(), index.end(), -1);

    for (int l = 0; l < static_cast<int>(landmarks.size()); l++) {
        for (const auto &obs: landmarks[l].observations)
            observationToLandmark[obs.imageId][obs.keypointId] = l;
    }
}

void Reconstruction::printTrackStatistics() const {
    int twoViews = 0;
    int threeViews = 0;
    int fourOrMoreViews = 0;

    for (const auto& landmark : landmarks) {
        int n = static_cast<int>(
            landmark.observations.size()
        );

        if (n == 2)
            twoViews++;
        else if (n == 3)
            threeViews++;
        else if (n >= 4)
            fourOrMoreViews++;
    }

    cout << "\n========== TRACK STATISTICS ==========\n";

    cout << "2-view landmarks: "
         << twoViews << '\n';

    cout << "3-view landmarks: "
         << threeViews << '\n';

    cout << "4+ view landmarks: "
         << fourOrMoreViews << '\n';
}


bool Reconstruction::exportFilteredPLY(
    const string& filename,
    int minObservations
) const {
    ofstream file(filename);

    if (!file.is_open())
        return false;

    vector<Point3d> validPoints;

    for (const auto& landmark : landmarks) {
        const Point3d& X = landmark.position;

        if (static_cast<int>(landmark.observations.size()) <
            minObservations)
            continue;

        if (!isfinite(X.x) ||
            !isfinite(X.y) ||
            !isfinite(X.z))
            continue;

        validPoints.push_back(X);
    }

    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << validPoints.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "end_header\n";

    for (const auto& X : validPoints) {
        file << X.x << " "
             << X.y << " "
             << X.z << "\n";
    }

    file.close();

    return file.good();
}

int Reconstruction::countCorrespondences(
    int imageId,
    vector<PairMatch> &verifiedPairs
) {
    if (images[imageId].registered)
        return 0;

    vector<Point3f> points3D;
    vector<Point2f> points2D;
    vector<CorrespondenceInfo> info;

    findCorrespondences(
        imageId,
        verifiedPairs,
        points3D,
        points2D,
        info
    );

    return static_cast<int>(points3D.size());
}

int Reconstruction::triangulateNewPoints(
    int imageId,
    const vector<PairMatch> &verifiedPairs,
    const Mat &K
) {
    if (!images[imageId].registered)
        return 0;

    int addedPoints = 0;

    for (const auto &pair: verifiedPairs) {
        if (pair.idx1 != imageId && pair.idx2 != imageId)
            continue;

        int oldImageId;

        if (pair.idx1 == imageId)
            oldImageId = pair.idx2;
        else
            oldImageId = pair.idx1;

        if (!images[oldImageId].registered)
            continue;

        auto calcProjectoinMatrix = [&]() -> array<Mat, 2> {
            Mat Rt1, Rt2;

            hconcat(
                images[pair.idx1].R,
                images[pair.idx1].t,
                Rt1
            );

            hconcat(
                images[pair.idx2].R,
                images[pair.idx2].t,
                Rt2
            );

            Mat P1 = K * Rt1;
            Mat P2 = K * Rt2;
            return {P1, P2};
        };

        for (auto match: pair.matches) {
            auto l1 = observationToLandmark[pair.idx1][match.queryIdx];
            auto l2 = observationToLandmark[pair.idx2][match.trainIdx];

            if (l1 == -1 and l2 == -1) {
                Point2f pt1 = images[pair.idx1].keypoints[match.queryIdx].pt;
                Point2f pt2 = images[pair.idx2].keypoints[match.trainIdx].pt;
                auto [P1, P2] = calcProjectoinMatrix();
                vector<Point2f> v1 = {pt1};
                vector<Point2f> v2 = {pt2};
                Mat points4D;
                triangulatePoints(P1, P2, v1, v2, points4D);
                float w = points4D.at<float>(3, 0);

                if (!isfinite(w) || abs(w) < 1e-8f) //// if it's eq to zero ignor dividing.
                    continue;

                Point3d X(
                    points4D.at<float>(0, 0) / w,
                    points4D.at<float>(1, 0) / w,
                    points4D.at<float>(2, 0) / w
                );
                const auto &img1 = images[pair.idx1];
                const auto &img2 = images[pair.idx2];

                // Test 1: Positive Depth
                if (!Triangulation::hasPositiveDepth(X, img1.R, img1.t) ||
                    !Triangulation::hasPositiveDepth(X, img2.R, img2.t))
                    continue;

                // Test 2: Reprojection Error
                if (!Triangulation::hasGoodReprojection(
                    X, pt1, pt2, P1, P2, 3.0))
                    continue;

                // Test 3: Triangulation Angle
                if (!Triangulation::hasGoodAngle(
                    X,
                    img1.R, img1.t,
                    img2.R, img2.t,
                    1.0))
                    continue;
                int newLandmarkId = static_cast<int>(landmarks.size());

                Landmark newLandmark;
                newLandmark.position = X;

                newLandmark.observations.push_back({
                    pair.idx1, match.queryIdx
                });

                newLandmark.observations.push_back({
                    pair.idx2, match.trainIdx
                });

                landmarks.push_back(newLandmark);

                observationToLandmark[pair.idx1][match.queryIdx] = newLandmarkId;
                observationToLandmark[pair.idx2][match.trainIdx] = newLandmarkId;

                addedPoints++;
            }
        }
    }

    return addedPoints;
}


void Reconstruction::findCorrespondences(
    int imageId,
    vector<PairMatch> &verifiedPairs,
    vector<Point3f> &points3D,
    vector<Point2f> &points2D,
    vector<CorrespondenceInfo> &correspondenceInfo
) {
    points3D.clear();
    points2D.clear();
    correspondenceInfo.clear();

    if (imageId < 0 || imageId >= static_cast<int>(images.size()))
        return;

    // keypointId -> landmarkId
    map<int, int> candidates;

    // Keypoints with conflicting landmark associations
    set<int> ambiguousKeypoints;

    for (const auto &pair: verifiedPairs) {
        if (pair.idx1 != imageId && pair.idx2 != imageId)
            continue;

        int oldImageId =
                (pair.idx1 == imageId) ? pair.idx2 : pair.idx1;

        if (oldImageId < 0 ||
            oldImageId >= static_cast<int>(images.size()) ||
            !images[oldImageId].registered)
            continue;

        for (const auto &match: pair.matches) {
            int oldKeypoint, newKeypoint;

            if (pair.idx1 == oldImageId) {
                oldKeypoint = match.queryIdx;
                newKeypoint = match.trainIdx;
            } else {
                oldKeypoint = match.trainIdx;
                newKeypoint = match.queryIdx;
            }

            if (oldKeypoint < 0 ||
                oldKeypoint >= static_cast<int>(
                    observationToLandmark[oldImageId].size()) ||
                newKeypoint < 0 ||
                newKeypoint >= static_cast<int>(
                    images[imageId].keypoints.size()))
                continue;

            int landmarkId =
                    observationToLandmark[oldImageId][oldKeypoint];

            if (landmarkId < 0 ||
                landmarkId >= static_cast<int>(landmarks.size()))
                continue;

            if (ambiguousKeypoints.count(newKeypoint))
                continue;

            auto [it, inserted] =
                    candidates.emplace(newKeypoint, landmarkId);

            if (!inserted && it->second != landmarkId) {
                ambiguousKeypoints.insert(newKeypoint);
                candidates.erase(it);
            }
        }
    }

    // Ensure that every landmark is used only once
    set<int> usedLandmarks;

    for (const auto &[keypointId, landmarkId]: candidates) {
        if (!usedLandmarks.insert(landmarkId).second)
            continue;

        const Point3d &X = landmarks[landmarkId].position;

        if (!isfinite(X.x) ||
            !isfinite(X.y) ||
            !isfinite(X.z))
            continue;

        Point2f pt = images[imageId].keypoints[keypointId].pt;

        points3D.emplace_back(
            static_cast<float>(X.x),
            static_cast<float>(X.y),
            static_cast<float>(X.z)
        );

        points2D.push_back(pt);

        correspondenceInfo.push_back({
            landmarkId,
            keypointId
        });
    }
}


bool Reconstruction::registerImage(
    int imageId,
    vector<PairMatch> &verifiedPairs,
    Mat &K
) {
    if (images[imageId].registered)
        return false;

    vector<Point3f> points3D;
    vector<Point2f> points2D;
    vector<CorrespondenceInfo> correspondenceInfo;

    findCorrespondences(
        imageId,
        verifiedPairs,
        points3D,
        points2D,
        correspondenceInfo
    );

    RegistrationRecord record;
    record.imageId = imageId;
    record.attempt = static_cast<int>(registrationLog.size());
    record.correspondences = static_cast<int>(points3D.size());

    if (points3D.size() < 10 or points2D.size() != points3D.size()) {
        record.reason = "too_few_correspondences";
        registrationLog.push_back(record);
        return false;
    }

    /// calculating the camera pose to the new image.
    Mat rvec, tvec, inliers;

    bool res = solvePnPRansac(
        points3D,
        points2D,
        K,
        noArray(),
        rvec, tvec,
        false,
        1000,
        4,
        0.99,
        inliers
    );

    cout << "[PnP] Image " << imageId
            << " | Correspondences: " << points3D.size()
            << " | Success: " << res
            << " | Inliers: " << inliers.total()
            << " | Ratio: "
            << (points3D.empty() ? 0.0 : 100.0 * inliers.total() / points3D.size())
            << "%" << endl;

    record.inliers = static_cast<int>(inliers.total());
    record.inlierRatio = static_cast<double>(inliers.total()) / points3D.size();

    if (!res) {
        record.reason = "pnp_failed";
        registrationLog.push_back(record);
        return false;
    }

    // Reprojection RMS of the PnP inliers (diagnostics only)
    if (!inliers.empty()) {
        vector<Point3f> inlier3D;
        vector<Point2f> inlier2D, projected;

        for (int i = 0; i < inliers.rows; i++) {
            inlier3D.push_back(points3D[inliers.at<int>(i, 0)]);
            inlier2D.push_back(points2D[inliers.at<int>(i, 0)]);
        }

        projectPoints(inlier3D, rvec, tvec, K, noArray(), projected);

        double sumSq = 0;
        for (size_t i = 0; i < projected.size(); i++) {
            Point2f d = projected[i] - inlier2D[i];
            sumSq += d.dot(d);
        }
        record.inlierRms = sqrt(sumSq / projected.size());
    }

    double inlierRatio =
            static_cast<double>(inliers.total()) / points3D.size();

    if (inliers.total() < 20 || inlierRatio < 0.20) {
        record.reason = "inlier_threshold";
        registrationLog.push_back(record);
        return false;
    }

    Mat R;
    Rodrigues(rvec, R);

    images[imageId].R = R;
    images[imageId].t = tvec;
    images[imageId].registered = true;

    map<int, int> updatedLandmarks;

    int countnumberofaddedobservations = 0;
    for (int i = 0; i < inliers.rows; i++) {
        int correspondenceIndex = inliers.at<int>(i, 0);

        const auto &info = correspondenceInfo[correspondenceIndex];

        int landmarkId = info.landmarkId;
        int keypointId = info.keypointId;

        if (observationToLandmark[imageId][keypointId] != -1)
            continue;

        if (updatedLandmarks.count(landmarkId))
            continue;

        updatedLandmarks[landmarkId] = keypointId;

        landmarks[landmarkId].observations.push_back({
            imageId,
            keypointId
        });

        observationToLandmark[imageId][keypointId] = landmarkId;
        countnumberofaddedobservations++;
    }

    cout << "number of added observations: " << countnumberofaddedobservations << '\n';

    record.addedObservations = countnumberofaddedobservations;
    record.registered = true;
    record.reason = "ok";
    registrationLog.push_back(record);

    return true;
}

void Reconstruction::initialize(int id1, int id2, PoseResult &pose, vector<Landmark> &initialLandmarks) {
    images[id1].R = Mat::eye(3, 3, CV_64F);
    images[id1].t = Mat::zeros(3, 1, CV_64F);
    images[id1].registered = true;

    images[id2].R = pose.R.clone();
    images[id2].t = pose.t.clone();
    images[id2].registered = true;

    anchor1 = id1;
    anchor2 = id2;

    landmarks = initialLandmarks;

    observationToLandmark.resize(images.size());
    for (int i = 0; i < images.size(); i++) {
        observationToLandmark[i].assign(images[i].keypoints.size(), -1);
    }

    for (int i = 0; i < landmarks.size(); i++) {
        for (auto &obs: landmarks[i].observations) {
            observationToLandmark[obs.imageId][obs.keypointId] = i;
        }
    }
}

bool Reconstruction::exportPLY(const string &filename) const {
    ofstream file(filename);

    if (!file.is_open()) return false;

    int validPoints = 0;

    for (const auto &landmark: landmarks) {
        const Point3d &p = landmark.position;

        if (isfinite(p.x) && isfinite(p.y) && isfinite(p.z))
            validPoints++;
    }

    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << validPoints << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "end_header\n";

    for (const auto &landmark: landmarks) {
        const Point3d &p = landmark.position;

        if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z))
            continue;

        file << p.x << " " << p.y << " " << p.z << "\n";
    }

    file.close();

    return file.good();
}
