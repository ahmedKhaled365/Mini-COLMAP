#include "BundleAdjuster.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;
using namespace cv;

// Reprojection error of a raw (distorted) keypoint.
// Camera: [angle-axis (3), translation (3)], world-to-camera.
// Intrinsics: [f, k1] shared by all images; cx, cy fixed.
struct ReprojectionError {
    ReprojectionError(double observedX, double observedY, double cx, double cy)
        : observedX(observedX), observedY(observedY), cx(cx), cy(cy) {
    }

    template<typename T>
    bool operator()(
        const T *const camera,
        const T *const point,
        const T *const intrinsics,
        T *residuals
    ) const {
        T p[3];

        ceres::AngleAxisRotatePoint(camera, point, p);

        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        const T x = p[0] / p[2];
        const T y = p[1] / p[2];
        const T r2 = x * x + y * y;
        const T distortion = T(1.0) + intrinsics[1] * r2;

        const T predictedX = intrinsics[0] * distortion * x + T(cx);
        const T predictedY = intrinsics[0] * distortion * y + T(cy);

        residuals[0] = predictedX - T(observedX);
        residuals[1] = predictedY - T(observedY);

        return true;
    }

    static ceres::CostFunction *Create(
        double observedX,
        double observedY,
        double cx,
        double cy
    ) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3, 2>(
            new ReprojectionError(observedX, observedY, cx, cy)
        );
    }

    double observedX, observedY;
    double cx, cy;
};

namespace {
    struct ResidualEntry {
        int imageId;
        int pointIndex; // index into the local point array
        double u, v;
    };

    double reprojectionRms(
        const vector<ResidualEntry> &entries,
        const vector<array<double, 6> > &cameras,
        const vector<array<double, 3> > &points,
        const array<double, 2> &intrinsics,
        double cx,
        double cy
    ) {
        if (entries.empty())
            return 0;

        double sumSq = 0;

        for (const auto &e: entries) {
            ReprojectionError f(e.u, e.v, cx, cy);
            double r[2];
            f(cameras[e.imageId].data(), points[e.pointIndex].data(),
              intrinsics.data(), r);
            sumSq += r[0] * r[0] + r[1] * r[1];
        }

        return sqrt(sumSq / entries.size());
    }
}

BAResult BundleAdjuster::optimize(
    vector<ImageData> &images,
    vector<Landmark> &landmarks,
    Mat &K,
    double &k1,
    const set<int> &variableCameras,
    const set<int> &variablePoints,
    int anchor1,
    int anchor2,
    const BAOptions &options
) {
    const auto start = chrono::steady_clock::now();

    BAResult result;
    result.label = options.label;
    result.focalBefore = result.focalAfter = K.at<double>(0, 0);
    result.k1Before = result.k1After = k1;

    for (const auto &img: images) {
        if (img.registered)
            result.registered++;
    }

    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    // --- Parameter blocks -----------------------------------------------
    vector<array<double, 6> > cameras(images.size());
    vector<bool> cameraUsed(images.size(), false);

    for (int i = 0; i < static_cast<int>(images.size()); i++) {
        if (!images[i].registered)
            continue;

        Mat rvec;
        Rodrigues(images[i].R, rvec);

        for (int j = 0; j < 3; j++) {
            cameras[i][j] = rvec.at<double>(j);
            cameras[i][j + 3] = images[i].t.at<double>(j);
        }
    }

    vector<int> pointIds(variablePoints.begin(), variablePoints.end());
    vector<array<double, 3> > points(pointIds.size());

    for (size_t k = 0; k < pointIds.size(); k++) {
        const Point3d &X = landmarks[pointIds[k]].position;
        points[k] = {X.x, X.y, X.z};
    }

    array<double, 2> intrinsics = {K.at<double>(0, 0), k1};

    // --- Residuals --------------------------------------------------------
    ceres::Problem problem;
    vector<ResidualEntry> entries;

    for (size_t k = 0; k < pointIds.size(); k++) {
        for (const auto &obs: landmarks[pointIds[k]].observations) {
            const auto &img = images[obs.imageId];

            if (!img.registered)
                continue;

            // Skip observations that start behind the camera; they are
            // removed by the outlier filter after optimization.
            double p[3];
            ceres::AngleAxisRotatePoint(cameras[obs.imageId].data(), points[k].data(), p);
            if (p[2] + cameras[obs.imageId][5] <= 1e-9)
                continue;

            const Point2f raw = img.rawPoints.empty()
                                    ? img.keypoints[obs.keypointId].pt
                                    : img.rawPoints[obs.keypointId];

            problem.AddResidualBlock(
                ReprojectionError::Create(raw.x, raw.y, cx, cy),
                new ceres::HuberLoss(options.huberScale),
                cameras[obs.imageId].data(),
                points[k].data(),
                intrinsics.data()
            );

            entries.push_back({obs.imageId, static_cast<int>(k), raw.x, raw.y});
            cameraUsed[obs.imageId] = true;
        }
    }

    result.points = static_cast<int>(pointIds.size());
    result.residuals = static_cast<int>(entries.size());

    if (entries.size() < 20) {
        result.reason = "too_few_residuals";
        return result;
    }

    // --- Constant blocks and gauge ----------------------------------------
    vector<int> constants, variables;

    for (int i = 0; i < static_cast<int>(images.size()); i++) {
        if (!cameraUsed[i])
            continue;

        if (variableCameras.count(i) && i != anchor1)
            variables.push_back(i);
        else
            constants.push_back(i);
    }

    // Gauge (7 DoF). If anchor1 is the only constant camera, the scale is
    // fixed through one translation component of anchor2. Otherwise at least
    // two constant cameras are required; the best-connected variable cameras
    // are fixed until that holds.
    const bool anchorGauge =
            constants.size() == 1 && constants[0] == anchor1 &&
            find(variables.begin(), variables.end(), anchor2) != variables.end();

    if (!anchorGauge) {
        while (constants.size() < 2 && variables.size() > 1) {
            map<int, int> counts;
            for (const auto &e: entries)
                counts[e.imageId]++;

            auto best = max_element(variables.begin(), variables.end(),
                                    [&](int a, int b) { return counts[a] < counts[b]; });
            constants.push_back(*best);
            variables.erase(best);
            result.reason = "gauge_fallback";
        }
    }

    for (int i: constants)
        problem.SetParameterBlockConstant(cameras[i].data());

    if (anchorGauge) {
        // Fix the largest translation component of the second anchor (scale)
        const auto &c = cameras[anchor2];
        int axis = 3;
        for (int j = 4; j < 6; j++) {
            if (abs(c[j]) > abs(c[axis]))
                axis = j;
        }
        problem.SetManifold(cameras[anchor2].data(),
                            new ceres::SubsetManifold(6, {axis}));
    }

    if (!options.refineFocal && !options.refineK1)
        problem.SetParameterBlockConstant(intrinsics.data());
    else if (!options.refineFocal)
        problem.SetManifold(intrinsics.data(), new ceres::SubsetManifold(2, {0}));
    else if (!options.refineK1)
        problem.SetManifold(intrinsics.data(), new ceres::SubsetManifold(2, {1}));

    result.variableCameras = static_cast<int>(variables.size());
    result.constantCameras = static_cast<int>(constants.size());

    // --- Solve ------------------------------------------------------------
    result.rmsBefore = reprojectionRms(entries, cameras, points, intrinsics, cx, cy);

    ceres::Solver::Options solverOptions;
    solverOptions.linear_solver_type = ceres::SPARSE_SCHUR;
    solverOptions.minimizer_type = ceres::TRUST_REGION;
    solverOptions.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solverOptions.max_num_iterations = options.maxIterations;
    solverOptions.num_threads = options.numThreads > 0
                                    ? options.numThreads
                                    : max(1, static_cast<int>(thread::hardware_concurrency()));
    solverOptions.function_tolerance = 1e-6;
    solverOptions.gradient_tolerance = 1e-10;
    solverOptions.parameter_tolerance = 1e-8;
    solverOptions.minimizer_progress_to_stdout = false;
    solverOptions.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(solverOptions, &problem, &summary);

    result.initialCost = summary.initial_cost;
    result.finalCost = summary.final_cost;
    result.iterations = static_cast<int>(summary.iterations.size());
    result.termination = ceres::TerminationTypeToString(summary.termination_type);
    result.rmsAfter = reprojectionRms(entries, cameras, points, intrinsics, cx, cy);
    result.focalAfter = intrinsics[0];
    result.k1After = intrinsics[1];
    result.seconds = chrono::duration<double>(chrono::steady_clock::now() - start).count();

    // --- Validate before writing back --------------------------------------
    result.usable = summary.IsSolutionUsable() &&
                    isfinite(result.finalCost) && isfinite(result.rmsAfter);

    if (!result.usable) {
        result.reason = "solver_unusable";
        return result;
    }

    if (result.finalCost > result.initialCost) {
        result.reason = "cost_increased";
        return result;
    }

    if (abs(intrinsics[0] / result.focalBefore - 1.0) > options.maxFocalChange ||
        abs(intrinsics[1]) > 0.5) {
        result.reason = "intrinsics_out_of_range";
        return result;
    }

    for (int i: variables) {
        Mat rvec = (Mat_<double>(3, 1) << cameras[i][0], cameras[i][1], cameras[i][2]);
        Mat t = (Mat_<double>(3, 1) << cameras[i][3], cameras[i][4], cameras[i][5]);
        Mat R;
        Rodrigues(rvec, R);
        images[i].R = R;
        images[i].t = t;
    }

    if (anchorGauge) {
        const auto &c = cameras[anchor2];
        Mat rvec = (Mat_<double>(3, 1) << c[0], c[1], c[2]);
        Mat R;
        Rodrigues(rvec, R);
        images[anchor2].R = R;
        images[anchor2].t = (Mat_<double>(3, 1) << c[3], c[4], c[5]);
    }

    for (size_t k = 0; k < pointIds.size(); k++)
        landmarks[pointIds[k]].position = Point3d(points[k][0], points[k][1], points[k][2]);

    K.at<double>(0, 0) = intrinsics[0];
    K.at<double>(1, 1) = intrinsics[0];
    k1 = intrinsics[1];

    result.applied = true;
    if (result.reason.empty())
        result.reason = "ok";

    return result;
}

void BundleAdjuster::writeLog(const vector<BAResult> &log, const string &path) {
    ofstream file(path);

    if (!file.is_open())
        return;

    file << "label,registered,variable_cameras,constant_cameras,points,residuals,"
            "initial_cost,final_cost,rms_before,rms_after,iterations,termination,"
            "seconds,focal_before,focal_after,k1_before,k1_after,applied,reason,"
            "filtered_observations,removed_landmarks\n";

    file << fixed << setprecision(5);

    for (const auto &r: log) {
        file << r.label << ',' << r.registered << ',' << r.variableCameras << ','
                << r.constantCameras << ',' << r.points << ',' << r.residuals << ','
                << r.initialCost << ',' << r.finalCost << ','
                << r.rmsBefore << ',' << r.rmsAfter << ',' << r.iterations << ','
                << r.termination << ',' << r.seconds << ','
                << r.focalBefore << ',' << r.focalAfter << ','
                << r.k1Before << ',' << r.k1After << ','
                << r.applied << ',' << r.reason << ','
                << r.filteredObservations << ',' << r.removedLandmarks << '\n';
    }
}
