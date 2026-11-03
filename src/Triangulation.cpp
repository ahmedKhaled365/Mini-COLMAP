#include "Triangulation.h"

using namespace std;
using namespace cv;

bool Triangulation::hasPositiveDepth(
    const Point3d &X,
    const Mat &R,
    const Mat &t
) {
    Mat Xworld = (Mat_<double>(3, 1) << X.x, X.y, X.z);

    Mat Xcam = R * Xworld + t;

    return Xcam.at<double>(2, 0) > 0;
}

bool Triangulation::hasGoodReprojection(
    const Point3d &X,
    const Point2f &pt1,
    const Point2f &pt2,
    const Mat &P1,
    const Mat &P2,
    double maxError
) {
    if (!isfinite(X.x) || !isfinite(X.y) || !isfinite(X.z))
        return false;

    Mat Xh = (Mat_<double>(4, 1) << X.x, X.y, X.z, 1.0);

    Mat projected1 = P1 * Xh;
    Mat projected2 = P2 * Xh;

    double z1 = projected1.at<double>(2, 0);
    double z2 = projected2.at<double>(2, 0);

    if (!isfinite(z1) || !isfinite(z2) ||
        abs(z1) < 1e-9 || abs(z2) < 1e-9)
        return false;

    Point2d reproj1(
        projected1.at<double>(0, 0) / z1,
        projected1.at<double>(1, 0) / z1
    );

    Point2d reproj2(
        projected2.at<double>(0, 0) / z2,
        projected2.at<double>(1, 0) / z2
    );

    double error1 = norm(reproj1 - Point2d(pt1));
    double error2 = norm(reproj2 - Point2d(pt2));

    return isfinite(error1) && isfinite(error2) &&
           error1 <= maxError && error2 <= maxError;
}

bool Triangulation::hasGoodAngle(
    const Point3d &X,
    const Mat &R1,
    const Mat &t1,
    const Mat &R2,
    const Mat &t2,
    double minAngle
) {
    if (!isfinite(X.x) || !isfinite(X.y) || !isfinite(X.z))
        return false;

    Mat Xworld = (Mat_<double>(3, 1) << X.x, X.y, X.z);

    Mat C1 = -R1.t() * t1;
    Mat C2 = -R2.t() * t2;

    Mat ray1 = Xworld - C1;
    Mat ray2 = Xworld - C2;

    double n1 = norm(ray1);
    double n2 = norm(ray2);

    if (!isfinite(n1) || !isfinite(n2) ||
        n1 < 1e-9 || n2 < 1e-9)
        return false;

    double cosAngle = ray1.dot(ray2) / (n1 * n2);

    cosAngle = clamp(cosAngle, -1.0, 1.0);

    double angle = acos(cosAngle) * 180.0 / CV_PI;

    return isfinite(angle) && angle >= minAngle;
}

vector<Landmark> Triangulation::triangulate(
    ImageData &img1,
    ImageData &img2,
    PoseResult &pose,
    Mat &K,
    TwoViewStats *stats
) {
    vector<Landmark> landmarks;
    vector<double> angles;

    if (pose.inlierMatches.size() < 2)
        return landmarks;

    // 1. Build projection matrices
    Mat P1, P2, Rt;

    hconcat(Mat::eye(3, 3, CV_64F),
            Mat::zeros(3, 1, CV_64F), P1);

    hconcat(pose.R, pose.t, Rt);

    P1 = K * P1;
    P2 = K * Rt;

    // 2. Collect corresponding 2D points
    vector<Point2f> points1, points2;

    for (const auto &match: pose.inlierMatches) {
        points1.push_back(img1.keypoints[match.queryIdx].pt);
        points2.push_back(img2.keypoints[match.trainIdx].pt);
    }

    // 3. Triangulation
    Mat points4D;
    triangulatePoints(P1, P2, points1, points2, points4D);

    // 4. Convert homogeneous coordinates
    for (int i = 0; i < points4D.cols; i++) {
        float w = points4D.at<float>(3, i);

        if (!isfinite(w) || abs(w) < 1e-8f)
            continue;

        Point3d X(
            points4D.at<float>(0, i) / w,
            points4D.at<float>(1, i) / w,
            points4D.at<float>(2, i) / w
        );

        if (!isfinite(X.x) || !isfinite(X.y) || !isfinite(X.z))
            continue;

        // 5. Positive depth in both cameras
        Mat Xmat = (Mat_<double>(3, 1) << X.x, X.y, X.z);
        Mat X2 = pose.R * Xmat + pose.t;

        if (X.z <= 0 || X2.at<double>(2, 0) <= 0)
            continue;

        if (!hasGoodReprojection(
            X,
            points1[i],
            points2[i],
            P1,
            P2,
            3.0
        )) {
            continue;
        }

        // Camera 1 center in world coordinates
        Vec3d C1(0.0, 0.0, 0.0);

        // Camera 2 center in world coordinates
        Mat C2mat = -pose.R.t() * pose.t;

        Vec3d C2(
            C2mat.at<double>(0),
            C2mat.at<double>(1),
            C2mat.at<double>(2)
        );

        Vec3d point(X.x, X.y, X.z);

        Vec3d ray1 = point - C1;
        Vec3d ray2 = point - C2;

        double cosine = ray1.dot(ray2) /
                        (norm(ray1) * norm(ray2));

        cosine = clamp(cosine, -1.0, 1.0);

        double angle = acos(cosine) * 180.0 / M_PI;

        angles.push_back(angle);

        if (angle < 1.0)
            continue;

        const auto &match = pose.inlierMatches[i];

        landmarks.push_back({
            X,
            {
                {img1.id, match.queryIdx},
                {img2.id, match.trainIdx}
            }
        });
    }

    if (stats) {
        stats->candidates = static_cast<int>(pose.inlierMatches.size());
        stats->triangulated = static_cast<int>(angles.size());
        if (!angles.empty()) {
            nth_element(angles.begin(), angles.begin() + angles.size() / 2, angles.end());
            stats->medianAngle = angles[angles.size() / 2];
        }
    }

    return landmarks;
}
