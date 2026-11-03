#include "BundleAdjuster.h"

#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;
using namespace cv;

struct ReprojectionError {
    ReprojectionError(
        double observedX,
        double observedY,
        double fx,
        double fy,
        double cx,
        double cy
    )
        : observedX(observedX),
          observedY(observedY),
          fx(fx),
          fy(fy),
          cx(cx),
          cy(cy) {
    }

    template<typename T>
    bool operator()(
        const T *const camera,
        const T *const point,
        T *residuals
    ) const {
        // Camera: [angle-axis (3), translation (3)]
        T p[3];

        ceres::AngleAxisRotatePoint(camera, point, p);

        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T predictedX = T(fx) * p[0] / p[2] + T(cx);
        T predictedY = T(fy) * p[1] / p[2] + T(cy);

        residuals[0] = predictedX - T(observedX);
        residuals[1] = predictedY - T(observedY);

        return true;
    }

    static ceres::CostFunction *Create(
        double observedX,
        double observedY,
        double fx,
        double fy,
        double cx,
        double cy
    ) {
        return new ceres::AutoDiffCostFunction<
            ReprojectionError,
            2,
            6,
            3
        >(
            new ReprojectionError(
                observedX,
                observedY,
                fx,
                fy,
                cx,
                cy
            )
        );
    }

    double observedX, observedY;
    double fx, fy, cx, cy;
};


bool BundleAdjuster::optimize(
    vector<ImageData>& images,
    vector<Landmark>& landmarks,
    const Mat& K
) {
    cout << "\n========== BUNDLE ADJUSTMENT ==========\n";

    if (images.size() < 2 || landmarks.empty() ||
        K.rows != 3 || K.cols != 3) {
        cerr << "Invalid Bundle Adjustment input\n";
        return false;
    }

    Mat K64;
    K.convertTo(K64, CV_64F);

    const double fx = K64.at<double>(0, 0);
    const double fy = K64.at<double>(1, 1);
    const double cx = K64.at<double>(0, 2);
    const double cy = K64.at<double>(1, 2);

    vector<array<double, 6>> cameras(images.size());
    vector<array<double, 3>> points(landmarks.size());

    vector<bool> validCamera(images.size(), false);
    vector<bool> validPoint(landmarks.size(), false);

    int registeredCount = 0;

    for (int i = 0; i < static_cast<int>(images.size()); ++i) {
        if (!images[i].registered)
            continue;

        if (images[i].R.empty() || images[i].t.empty())
            continue;

        Mat R64, t64, rvec;
        images[i].R.convertTo(R64, CV_64F);
        images[i].t.convertTo(t64, CV_64F);

        if (R64.rows != 3 || R64.cols != 3 ||
            t64.total() != 3)
            continue;

        Rodrigues(R64, rvec);
        t64 = t64.reshape(1, 3);

        if (!checkRange(rvec) || !checkRange(t64))
            continue;

        for (int j = 0; j < 3; ++j) {
            cameras[i][j] = rvec.at<double>(j, 0);
            cameras[i][j + 3] = t64.at<double>(j, 0);
        }

        validCamera[i] = true;
        ++registeredCount;
    }

    if (registeredCount < 2) {
        cerr << "Not enough valid cameras\n";
        return false;
    }

    for (int j = 0; j < static_cast<int>(landmarks.size()); ++j) {
        const Point3d& X = landmarks[j].position;

        if (!std::isfinite(X.x) ||
            !std::isfinite(X.y) ||
            !std::isfinite(X.z))
            continue;

        points[j] = {X.x, X.y, X.z};
        validPoint[j] = true;
    }

    // Use the initial reconstruction pair as gauge anchors.
    const int anchor1 = 6;
    const int anchor2 = 7;

    if (anchor1 >= static_cast<int>(images.size()) ||
        anchor2 >= static_cast<int>(images.size()) ||
        !validCamera[anchor1] ||
        !validCamera[anchor2]) {
        cerr << "Initial pair cameras are not available\n";
        return false;
    }

    ceres::Problem problem;

    int residualCount = 0;
    set<pair<int, int>> usedObservations;

    for (int j = 0; j < static_cast<int>(landmarks.size()); ++j) {
        if (!validPoint[j])
            continue;

        for (const auto& obs : landmarks[j].observations) {
            int imageId = obs.imageId;
            int keypointId = obs.keypointId;

            if (imageId < 0 ||
                imageId >= static_cast<int>(images.size()) ||
                !validCamera[imageId])
                continue;

            if (keypointId < 0 ||
                keypointId >= static_cast<int>(
                    images[imageId].keypoints.size()))
                continue;

            if (!usedObservations.insert(
                    {imageId, keypointId}).second)
                continue;

            const Point2f observed =
                images[imageId].keypoints[keypointId].pt;

            if (!std::isfinite(observed.x) ||
                !std::isfinite(observed.y))
                continue;

            // Check the initial camera-space depth.
            const auto& cam = cameras[imageId];
            const auto& X = points[j];

            double transformed[3];

            ceres::AngleAxisRotatePoint(
                cam.data(),
                X.data(),
                transformed
            );

            transformed[0] += cam[3];
            transformed[1] += cam[4];
            transformed[2] += cam[5];

            if (!std::isfinite(transformed[2]) ||
                transformed[2] <= 1e-6)
                continue;

            const double u =
                fx * transformed[0] / transformed[2] + cx;
            const double v =
                fy * transformed[1] / transformed[2] + cy;

            if (!std::isfinite(u) || !std::isfinite(v))
                continue;

            auto* cost = ReprojectionError::Create(
                observed.x,
                observed.y,
                fx, fy, cx, cy
            );

            problem.AddResidualBlock(
                cost,
                new ceres::HuberLoss(2.0),
                cameras[imageId].data(),
                points[j].data()
            );

            ++residualCount;
        }
    }

    cout << "Registered Cameras: " << registeredCount << '\n';
    cout << "Total Landmarks: " << landmarks.size() << '\n';
    cout << "Residual Blocks: " << residualCount << '\n';

    if (residualCount < 100) {
        cerr << "Not enough residuals for optimization\n";
        return false;
    }

    if (!problem.HasParameterBlock(cameras[anchor1].data()) ||
        !problem.HasParameterBlock(cameras[anchor2].data())) {
        cerr << "Gauge anchor cameras have no residuals\n";
        return false;
    }

    problem.SetParameterBlockConstant(cameras[anchor1].data());
    problem.SetParameterBlockConstant(cameras[anchor2].data());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 30;
    options.num_threads = 6;

    options.initial_trust_region_radius = 1e3;
    options.max_trust_region_radius = 1e6;

    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-8;

    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.FullReport() << '\n';
    cout << summary.BriefReport() << '\n';

    if (!summary.IsSolutionUsable()) {
        cerr << "Bundle Adjustment failed\n";
        return false;
    }

    for (int i = 0; i < static_cast<int>(images.size()); ++i) {
        if (!validCamera[i])
            continue;

        if (!problem.HasParameterBlock(cameras[i].data()))
            continue;

        const auto& cam = cameras[i];

        Mat rvec = (Mat_<double>(3, 1));
        Mat t = (Mat_<double>(3, 1));

        for (int j = 0; j < 3; ++j) {
            rvec.at<double>(j, 0) = cam[j];
            t.at<double>(j, 0) = cam[j + 3];
        }

        Mat R;
        Rodrigues(rvec, R);

        images[i].R = R.clone();
        images[i].t = t.clone();
    }

    for (int j = 0; j < static_cast<int>(landmarks.size()); ++j) {
        if (!validPoint[j])
            continue;

        if (!problem.HasParameterBlock(points[j].data()))
            continue;

        const auto& X = points[j];

        landmarks[j].position =
            Point3d(X[0], X[1], X[2]);
    }

    cout << "Bundle Adjustment completed\n";

    return true;
}

