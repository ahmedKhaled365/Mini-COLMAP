#include "Diagnostics.h"

#include <filesystem>
#include <fstream>
#include <iomanip>

using namespace std;
using namespace cv;

// --------------------------------------------------
// Small helpers
// --------------------------------------------------

Stats computeStats(vector<double> values) {
    Stats s;
    s.count = values.size();

    if (values.empty())
        return s;

    sort(values.begin(), values.end());

    double sum = 0, sumSq = 0;

    for (double v: values) {
        sum += v;
        sumSq += v * v;
    }

    const size_t n = values.size();
    s.mean = sum / n;
    s.rms = sqrt(sumSq / n);
    s.median = values[n / 2];
    s.p95 = values[min(n - 1, static_cast<size_t>(0.95 * (n - 1) + 0.5))];
    s.max = values.back();

    return s;
}

static Vec3d toVec(const Point3d &p) {
    return {p.x, p.y, p.z};
}

static Vec3d matToVec(const Mat &m) {
    Mat m64;
    m.convertTo(m64, CV_64F);
    return {m64.at<double>(0), m64.at<double>(1), m64.at<double>(2)};
}

static Matx33d toMatx(const Mat &R) {
    Mat R64;
    R.convertTo(R64, CV_64F);
    return Matx33d(R64);
}

// Optical axis (camera +z) expressed in world coordinates.
static Vec3d viewingDirection(const ImageData &img) {
    return toMatx(img.R).t() * Vec3d(0, 0, 1);
}

static bool isFinitePoint(const Point3d &X) {
    return isfinite(X.x) && isfinite(X.y) && isfinite(X.z);
}

Diagnostics::Diagnostics(DiagnosticsConfig config) : config(config) {
}

Point3d Diagnostics::cameraCenter(const ImageData &img) {
    Vec3d C = -(toMatx(img.R).t() * matToVec(img.t));
    return {C[0], C[1], C[2]};
}

double Diagnostics::observationError(
    const ImageData &img,
    const Point3d &X,
    int keypointId,
    const Mat &K,
    double *depth
) {
    Vec3d Xc = toMatx(img.R) * toVec(X) + matToVec(img.t);

    if (depth)
        *depth = Xc[2];

    if (Xc[2] <= 1e-12)
        return numeric_limits<double>::infinity();

    const Matx33d K64 = toMatx(K);
    const double u = K64(0, 0) * Xc[0] / Xc[2] + K64(0, 2);
    const double v = K64(1, 1) * Xc[1] / Xc[2] + K64(1, 2);

    const Point2f &pt = img.keypoints[keypointId].pt;

    return hypot(u - pt.x, v - pt.y);
}

double Diagnostics::maxTriangulationAngle(
    const vector<ImageData> &images,
    const Landmark &landmark
) {
    vector<Vec3d> rays;
    const Vec3d X = toVec(landmark.position);

    for (const auto &obs: landmark.observations) {
        const auto &img = images[obs.imageId];

        if (!img.registered)
            continue;

        Vec3d ray = X - toVec(cameraCenter(img));
        double n = norm(ray);

        if (n > 1e-12)
            rays.push_back(ray / n);
    }

    double best = 0;

    for (size_t i = 0; i < rays.size(); i++) {
        for (size_t j = i + 1; j < rays.size(); j++) {
            double c = clamp(rays[i].dot(rays[j]), -1.0, 1.0);
            best = max(best, acos(c) * 180.0 / CV_PI);
        }
    }

    return best;
}

bool Diagnostics::isReliable(
    const vector<ImageData> &images,
    const Landmark &landmark,
    const Mat &K
) const {
    if (static_cast<int>(landmark.observations.size()) < config.reliableMinTrack)
        return false;

    if (!isFinitePoint(landmark.position))
        return false;

    for (const auto &obs: landmark.observations) {
        double e = observationError(
            images[obs.imageId], landmark.position, obs.keypointId, K
        );

        if (!(e <= config.reliableMaxError))
            return false;
    }

    return maxTriangulationAngle(images, landmark) >= config.reliableMinAngle;
}

Vec3b Diagnostics::landmarkColor(
    const vector<ImageData> &images,
    const Landmark &landmark
) {
    Vec3d sum(0, 0, 0);
    int n = 0;

    for (const auto &obs: landmark.observations) {
        const auto &img = images[obs.imageId];

        if (img.image.empty())
            continue;

        // Sample the photograph at the detected (distorted) position
        const Point2f &pt = img.rawPoints.empty()
                                ? img.keypoints[obs.keypointId].pt
                                : img.rawPoints[obs.keypointId];
        int x = clamp(cvRound(pt.x), 0, img.image.cols - 1);
        int y = clamp(cvRound(pt.y), 0, img.image.rows - 1);

        Vec3b c = img.image.at<Vec3b>(y, x);
        sum += Vec3d(c[0], c[1], c[2]);
        n++;
    }

    if (n == 0)
        return {200, 200, 200};

    sum /= n;

    return Vec3b(
        saturate_cast<uchar>(sum[0]),
        saturate_cast<uchar>(sum[1]),
        saturate_cast<uchar>(sum[2])
    );
}

Point3d Diagnostics::estimateLookAt(const vector<ImageData> &images) {
    // Least-squares point closest to all optical axes:
    // sum_i (I - d_i d_i^T) (X - C_i) = 0
    Matx33d A = Matx33d::zeros();
    Vec3d b(0, 0, 0);
    int n = 0;

    for (const auto &img: images) {
        if (!img.registered)
            continue;

        Vec3d d = viewingDirection(img);
        Vec3d C = toVec(cameraCenter(img));
        Matx33d M = Matx33d::eye() - d * d.t();

        A += M;
        b += M * C;
        n++;
    }

    if (n < 2)
        return {0, 0, 0};

    Vec3d X;
    solve(Mat(A), Mat(b), X, DECOMP_SVD);

    return {X[0], X[1], X[2]};
}

// --------------------------------------------------
// Analysis
// --------------------------------------------------

ReconstructionReport Diagnostics::analyze(
    const vector<ImageData> &images,
    const vector<Landmark> &landmarks,
    const Mat &K
) const {
    ReconstructionReport r;
    r.totalImages = static_cast<int>(images.size());
    r.landmarks = landmarks.size();

    map<int, vector<double> > perCamera;
    vector<double> allErrors, angles, tracks;

    for (const auto &img: images) {
        if (img.registered)
            r.registeredImages++;
    }

    r.lookAt = estimateLookAt(images);

    vector<double> camDistances;

    for (const auto &img: images) {
        if (img.registered)
            camDistances.push_back(norm(cameraCenter(img) - r.lookAt));
    }

    r.medianCameraDistance = computeStats(camDistances).median;
    const double regionRadius = config.objectRegionFactor * r.medianCameraDistance;

    vector<double> regionErrors;

    for (const auto &landmark: landmarks) {
        if (!isFinitePoint(landmark.position))
            continue;

        const int n = static_cast<int>(landmark.observations.size());
        r.trackHistogram[n]++;
        tracks.push_back(n);

        const bool inRegion =
                norm(landmark.position - r.lookAt) <= regionRadius;

        for (const auto &obs: landmark.observations) {
            double depth = 0;
            double e = observationError(
                images[obs.imageId], landmark.position, obs.keypointId, K, &depth
            );

            r.observations++;

            if (depth <= 0) {
                r.negativeDepthObservations++;
                continue;
            }

            allErrors.push_back(e);
            perCamera[obs.imageId].push_back(e);

            if (inRegion)
                regionErrors.push_back(e);
        }

        double angle = maxTriangulationAngle(images, landmark);
        angles.push_back(angle);

        const double edges[] = {1, 2, 5, 10, 20};
        int bin = 0;
        while (bin < 5 && angle >= edges[bin])
            bin++;
        r.angleHistogram[bin]++;

        const bool reliable = isReliable(images, landmark, K);

        if (reliable)
            r.reliableLandmarks++;

        int minId = INT_MAX, maxId = INT_MIN;
        for (const auto &obs: landmark.observations) {
            minId = min(minId, obs.imageId);
            maxId = max(maxId, obs.imageId);
        }
        if (maxId - minId > config.longRangeSpan)
            r.longRangeLandmarks++;

        if (inRegion) {
            r.objectRegionLandmarks++;
            if (reliable)
                r.objectRegionReliable++;
        }
    }

    r.reprojection = computeStats(allErrors);
    r.triAngle = computeStats(angles);
    r.trackLength = computeStats(tracks);
    r.objectRegionReprojection = computeStats(regionErrors);

    vector<double> steps;
    Point3d previous;
    bool hasPrevious = false;

    for (const auto &img: images) {
        if (!img.registered)
            continue;

        CameraStats cs;
        cs.imageId = img.id;
        cs.center = cameraCenter(img);

        auto it = perCamera.find(img.id);
        if (it != perCamera.end()) {
            cs.observations = static_cast<int>(it->second.size());
            cs.rms = computeStats(it->second).rms;
        }

        r.cameras.push_back(cs);

        if (hasPrevious)
            steps.push_back(norm(cs.center - previous));

        previous = cs.center;
        hasPrevious = true;
    }

    r.cameraStep = computeStats(steps);

    if (!steps.empty() && r.cameraStep.mean > 0) {
        double var = 0;
        for (double s: steps)
            var += (s - r.cameraStep.mean) * (s - r.cameraStep.mean);
        r.cameraStepCv = sqrt(var / steps.size()) / r.cameraStep.mean;
    }

    return r;
}

// --------------------------------------------------
// Text outputs
// --------------------------------------------------

static string statsLine(const Stats &s, int precision = 3) {
    ostringstream os;
    os << fixed << setprecision(precision)
            << "mean " << s.mean
            << " | median " << s.median
            << " | P95 " << s.p95
            << " | max " << s.max
            << " | RMS " << s.rms
            << " (n=" << s.count << ")";
    return os.str();
}

void Diagnostics::printSummary(const ReconstructionReport &r, ostream &out) {
    int two = 0, three = 0, fourPlus = 0;

    for (const auto &[len, count]: r.trackHistogram) {
        if (len == 2) two += count;
        else if (len == 3) three += count;
        else if (len >= 4) fourPlus += count;
    }

    out << "\n========== DIAGNOSTICS ==========\n";
    out << "Registered: " << r.registeredImages << " / " << r.totalImages
            << " | Landmarks: " << r.landmarks
            << " | Observations: " << r.observations
            << " | Negative-depth obs: " << r.negativeDepthObservations << '\n';
    out << "Reprojection [px]: " << statsLine(r.reprojection) << '\n';
    out << "Track length: mean " << fixed << setprecision(2) << r.trackLength.mean
            << " | 2-view " << two
            << " | 3-view " << three
            << " | 4+ view " << fourPlus << '\n';
    out << "Tri. angle [deg]: " << statsLine(r.triAngle, 2) << '\n';
    out << "Angle bins <1/1-2/2-5/5-10/10-20/20+: ";
    for (int i = 0; i < 6; i++)
        out << r.angleHistogram[i] << (i < 5 ? " / " : "\n");
    out << "Reliable landmarks: " << r.reliableLandmarks << '\n';
    out << "Long-range landmarks: " << r.longRangeLandmarks << '\n';
    out << "Object region: " << r.objectRegionLandmarks
            << " landmarks (" << r.objectRegionReliable << " reliable)"
            << " | reproj median " << setprecision(3)
            << r.objectRegionReprojection.median << " px\n";
    out << "Camera step CV: " << setprecision(3) << r.cameraStepCv << '\n';
    out.unsetf(ios::floatfield);
}

void Diagnostics::writeReport(
    const ReconstructionReport &r,
    const string &path,
    const string &header
) {
    ofstream file(path);

    if (!file.is_open())
        return;

    file << header << '\n';
    printSummary(r, file);

    file << "\nTrack length histogram:\n";
    for (const auto &[len, count]: r.trackHistogram)
        file << "  " << len << " views: " << count << '\n';

    file << fixed << setprecision(4);
    file << "\nLookAt point: " << r.lookAt
            << " | median camera distance: " << r.medianCameraDistance << '\n';
    file << "Camera step: " << statsLine(r.cameraStep, 4) << '\n';

    file << "\nPer-camera reprojection:\n";
    file << "  image  obs     rms[px]   center\n";
    for (const auto &c: r.cameras) {
        file << "  " << setw(5) << c.imageId
                << "  " << setw(5) << c.observations
                << "  " << setw(8) << setprecision(3) << c.rms
                << "   " << setprecision(4) << c.center << '\n';
    }
}

void Diagnostics::appendCheckpointCsv(
    const ReconstructionReport &r,
    const string &path
) {
    const bool exists = filesystem::exists(path);
    ofstream file(path, ios::app);

    if (!file.is_open())
        return;

    if (!exists) {
        file << "registered,landmarks,observations,neg_depth,"
                "reproj_mean,reproj_median,reproj_p95,reproj_max,"
                "track_mean,two_view_frac,angle_median,angle_p95,"
                "angle_lt2_frac,reliable,object_region,object_region_reliable,"
                "camera_step_cv\n";
    }

    int two = r.trackHistogram.count(2) ? r.trackHistogram.at(2) : 0;
    int lt2 = r.angleHistogram[0] + r.angleHistogram[1];
    double n = max<size_t>(1, r.landmarks);

    file << fixed << setprecision(4)
            << r.registeredImages << ',' << r.landmarks << ','
            << r.observations << ',' << r.negativeDepthObservations << ','
            << r.reprojection.mean << ',' << r.reprojection.median << ','
            << r.reprojection.p95 << ',' << r.reprojection.max << ','
            << r.trackLength.mean << ',' << two / n << ','
            << r.triAngle.median << ',' << r.triAngle.p95 << ','
            << lt2 / n << ',' << r.reliableLandmarks << ','
            << r.objectRegionLandmarks << ',' << r.objectRegionReliable << ','
            << r.cameraStepCv << '\n';
}

void Diagnostics::writeRegistrationCsv(
    const vector<RegistrationRecord> &records,
    const string &path
) {
    ofstream file(path);

    if (!file.is_open())
        return;

    file << "attempt,image,correspondences,inliers,inlier_ratio,"
            "inlier_rms,added_observations,new_landmarks,registered,reason\n";

    file << fixed << setprecision(4);

    for (const auto &r: records) {
        file << r.attempt << ',' << r.imageId << ','
                << r.correspondences << ',' << r.inliers << ','
                << r.inlierRatio << ',' << r.inlierRms << ','
                << r.addedObservations << ',' << r.newLandmarks << ','
                << r.registered << ',' << r.reason << '\n';
    }
}

void Diagnostics::writeInitCandidatesCsv(
    const vector<InitCandidateRecord> &records,
    const string &path
) {
    ofstream file(path);

    if (!file.is_open())
        return;

    file << "idx1,idx2,f_inliers,e_inliers,h_inliers,h_over_e,"
            "triangulated,landmarks,median_angle,score,accepted,reason\n";

    file << fixed << setprecision(4);

    for (const auto &r: records) {
        file << r.idx1 << ',' << r.idx2 << ','
                << r.fInliers << ',' << r.eInliers << ',' << r.hInliers << ','
                << r.hOverE << ',' << r.triangulated << ',' << r.landmarks << ','
                << r.medianAngle << ',' << r.score << ','
                << r.accepted << ',' << r.reason << '\n';
    }
}

// --------------------------------------------------
// PLY outputs
// --------------------------------------------------

// Writes colored vertices (RGB) and optional colored edges as an ASCII or
// binary little-endian PLY. Binary files are ~2.5x smaller and load faster.
static bool writePly(
    const string &path,
    const vector<Point3d> &vertices,
    const vector<Vec3b> &vertexColors,
    const vector<array<int, 2> > &edges,
    const vector<Vec3b> &edgeColors,
    bool binary
) {
    ofstream file(path, binary ? ios::binary : ios::out);

    if (!file.is_open())
        return false;

    file << "ply\n"
            << (binary ? "format binary_little_endian 1.0\n" : "format ascii 1.0\n")
            << "element vertex " << vertices.size() << '\n'
            << "property float x\nproperty float y\nproperty float z\n"
            << "property uchar red\nproperty uchar green\nproperty uchar blue\n";

    if (!edges.empty()) {
        file << "element edge " << edges.size() << '\n'
                << "property int vertex1\nproperty int vertex2\n"
                << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    }

    file << "end_header\n";

    if (binary) {
        for (size_t i = 0; i < vertices.size(); i++) {
            const float xyz[3] = {
                static_cast<float>(vertices[i].x),
                static_cast<float>(vertices[i].y),
                static_cast<float>(vertices[i].z)
            };
            file.write(reinterpret_cast<const char *>(xyz), sizeof(xyz));
            file.write(reinterpret_cast<const char *>(vertexColors[i].val), 3);
        }

        for (size_t i = 0; i < edges.size(); i++) {
            const int32_t ids[2] = {edges[i][0], edges[i][1]};
            file.write(reinterpret_cast<const char *>(ids), sizeof(ids));
            file.write(reinterpret_cast<const char *>(edgeColors[i].val), 3);
        }
    } else {
        file << setprecision(7);

        for (size_t i = 0; i < vertices.size(); i++) {
            const Vec3b &c = vertexColors[i];
            file << vertices[i].x << ' ' << vertices[i].y << ' ' << vertices[i].z << ' '
                    << int(c[0]) << ' ' << int(c[1]) << ' ' << int(c[2]) << '\n';
        }

        for (size_t i = 0; i < edges.size(); i++) {
            const Vec3b &c = edgeColors[i];
            file << edges[i][0] << ' ' << edges[i][1] << ' '
                    << int(c[0]) << ' ' << int(c[1]) << ' ' << int(c[2]) << '\n';
        }
    }

    return file.good();
}

bool Diagnostics::exportPoints(
    const string &path,
    const vector<ImageData> &images,
    const vector<Landmark> &landmarks,
    const Mat &K,
    int minTrack,
    int quality
) const {
    vector<Point3d> points;
    vector<Vec3b> colors;

    for (const auto &landmark: landmarks) {
        if (!isFinitePoint(landmark.position))
            continue;

        if (static_cast<int>(landmark.observations.size()) < minTrack)
            continue;

        if (quality != 0) {
            bool reliable = isReliable(images, landmark, K);
            if ((quality == 1) != reliable)
                continue;
        }

        points.push_back(landmark.position);

        const Vec3b c = landmarkColor(images, landmark); // BGR
        colors.push_back(Vec3b(c[2], c[1], c[0]));
    }

    return writePly(path, points, colors, {}, {}, config.binaryPly);
}

bool Diagnostics::exportCameras(
    const string &path,
    const vector<ImageData> &images,
    const vector<Landmark> &landmarks,
    const Mat &K,
    bool withPoints
) const {
    vector<int> cams;

    for (const auto &img: images) {
        if (img.registered)
            cams.push_back(img.id);
    }

    if (cams.empty())
        return false;

    const Point3d lookAt = estimateLookAt(images);
    vector<double> distances;
    for (int id: cams)
        distances.push_back(norm(cameraCenter(images[id]) - lookAt));

    const double scale = 0.06 * max(1e-6, computeStats(distances).median);
    const Matx33d Kinv = toMatx(K).inv();
    const double W = images[cams[0]].image.cols;
    const double H = images[cams[0]].image.rows;

    vector<Point3d> vertices;
    vector<Vec3b> vertexColors; // RGB
    vector<array<int, 2> > edges;
    vector<Vec3b> edgeColors;   // RGB

    for (size_t k = 0; k < cams.size(); k++) {
        const auto &img = images[cams[k]];

        // Color gradient along the sequence: blue -> red
        double a = cams.size() > 1 ? double(k) / (cams.size() - 1) : 0.0;
        Vec3b color(
            saturate_cast<uchar>(255 * a),
            60,
            saturate_cast<uchar>(255 * (1 - a))
        );

        const Point3d C = cameraCenter(img);
        const Matx33d Rt = toMatx(img.R).t();
        const int base = static_cast<int>(vertices.size());

        vertices.push_back(C);
        vertexColors.push_back(color);

        const double corners[4][2] = {{0, 0}, {W, 0}, {W, H}, {0, H}};

        for (const auto &uv: corners) {
            Vec3d ray = Kinv * Vec3d(uv[0], uv[1], 1.0);
            Vec3d P = toVec(C) + Rt * (ray * scale);
            vertices.emplace_back(P[0], P[1], P[2]);
            vertexColors.push_back(color);
        }

        for (int c = 0; c < 4; c++) {
            edges.push_back({base, base + 1 + c});
            edges.push_back({base + 1 + c, base + 1 + (c + 1) % 4});
            edgeColors.push_back(color);
            edgeColors.push_back(color);
        }

        // Trajectory edge to the previous camera center
        if (k > 0) {
            edges.push_back({base - 5, base});
            edgeColors.push_back(Vec3b(255, 255, 0));
        }
    }

    if (withPoints) {
        for (const auto &landmark: landmarks) {
            if (!isFinitePoint(landmark.position))
                continue;

            Vec3b c = landmarkColor(images, landmark);
            vertices.push_back(landmark.position);
            vertexColors.push_back(Vec3b(c[2], c[1], c[0]));
        }
    }

    return writePly(path, vertices, vertexColors, edges, edgeColors, config.binaryPly);
}

// --------------------------------------------------
// Preview rendering
// --------------------------------------------------

bool Diagnostics::renderPreview(
    const string &path,
    const vector<ImageData> &images,
    const vector<Landmark> &landmarks,
    const Mat &K,
    const string &title
) const {
    vector<int> cams;
    Vec3d up(0, 0, 0), forward(0, 0, 0);

    for (const auto &img: images) {
        if (!img.registered)
            continue;

        cams.push_back(img.id);
        up += toMatx(img.R).t() * Vec3d(0, -1, 0);
        forward += viewingDirection(img);
    }

    if (cams.size() < 2)
        return false;

    up = normalize(up);
    forward = normalize(forward - up * forward.dot(up));
    const Vec3d right = normalize(forward.cross(up));

    const Vec3d L = toVec(estimateLookAt(images));

    vector<double> distances;
    for (int id: cams)
        distances.push_back(norm(toVec(cameraCenter(images[id])) - L));
    const double camDist = computeStats(distances).median;
    const double region = config.objectRegionFactor * camDist;

    struct View {
        string name;
        Vec3d u, v, depth;
        double extent;
        bool showCameras;
    };

    const vector<View> views = {
        {"top (wide)", right, -forward, -up, 1.6 * camDist, true},
        {"top (object)", right, -forward, -up, 1.4 * region, false},
        {"front (object)", right, -up, forward, 1.4 * region, false},
        {"side (object)", -forward, -up, right, 1.4 * region, false},
    };

    struct Sample {
        Vec3d X;
        Vec3b color;
        bool reliable;
    };

    vector<Sample> samples;
    for (const auto &landmark: landmarks) {
        if (!isFinitePoint(landmark.position))
            continue;

        samples.push_back({
            toVec(landmark.position),
            landmarkColor(images, landmark),
            isReliable(images, landmark, K)
        });
    }

    const int S = 520;
    Mat canvas(2 * S + 40, 2 * S, CV_8UC3, Scalar(35, 35, 35));

    for (size_t vi = 0; vi < views.size(); vi++) {
        const View &view = views[vi];
        Mat panel = canvas(Rect((vi % 2) * S, 40 + (vi / 2) * S, S, S));

        auto project = [&](const Vec3d &X) {
            Vec3d d = X - L;
            return Point2d(
                S / 2.0 + d.dot(view.u) / view.extent * (S / 2.0),
                S / 2.0 + d.dot(view.v) / view.extent * (S / 2.0)
            );
        };

        // Painter's algorithm: far points first
        vector<int> order(samples.size());
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b) {
            return (samples[a].X - L).dot(view.depth) >
                   (samples[b].X - L).dot(view.depth);
        });

        for (int idx: order) {
            Point2d p = project(samples[idx].X);
            if (p.x < 0 || p.y < 0 || p.x >= S || p.y >= S)
                continue;

            const Vec3b &c = samples[idx].color;
            circle(panel, p, samples[idx].reliable ? 2 : 1,
                   Scalar(c[0], c[1], c[2]), FILLED, LINE_AA);
        }

        if (view.showCameras) {
            for (size_t k = 0; k < cams.size(); k++) {
                const auto &img = images[cams[k]];
                Vec3d C = toVec(cameraCenter(img));
                Point2d p = project(C);
                Point2d q = project(C + viewingDirection(img) * (0.15 * camDist));
                double a = double(k) / max<size_t>(1, cams.size() - 1);
                Scalar color(255 * (1 - a), 80, 255 * a);
                line(panel, p, q, color, 1, LINE_AA);
                circle(panel, p, 3, color, FILLED, LINE_AA);
            }

            circle(panel, project(L), region / view.extent * (S / 2.0),
                   Scalar(0, 200, 255), 1, LINE_AA);
        }

        rectangle(panel, Rect(0, 0, S, S), Scalar(90, 90, 90), 1);
        putText(panel, view.name, Point(8, 20), FONT_HERSHEY_SIMPLEX,
                0.55, Scalar(230, 230, 230), 1, LINE_AA);
    }

    putText(canvas, title, Point(10, 27), FONT_HERSHEY_SIMPLEX,
            0.65, Scalar(240, 240, 240), 1, LINE_AA);

    return imwrite(path, canvas);
}

// --------------------------------------------------
// Focal length sweep
// --------------------------------------------------

void Diagnostics::focalSweep(
    const vector<ImageData> &images,
    const vector<PairMatch> &verifiedPairs,
    const vector<double> &focals,
    const vector<double> &k1s,
    int minGap,
    const string &csvPath
) {
    ofstream file(csvPath);
    file << "focal,k1,pairs,mean_inlier_frac,median_inlier_frac,median_sampson_px\n";

    cout << "\n========== FOCAL SWEEP ==========\n";
    cout << "Pairs with gap >= " << minGap << ", E-RANSAC threshold 1 px\n";

    const double cx = images[0].image.cols / 2.0;
    const double cy = images[0].image.rows / 2.0;

    for (double k1: k1s)
    for (double f: focals) {
        Mat K = (Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
        Mat dist = (Mat_<double>(1, 4) << k1, 0, 0, 0);
        const Matx33d Kinv = Matx33d(K).inv();

        vector<double> fractions, sampson;

        for (const auto &pair: verifiedPairs) {
            if (pair.idx2 - pair.idx1 < minGap || pair.matches.size() < 50)
                continue;

            vector<Point2f> p1, p2;
            for (const auto &m: pair.matches) {
                p1.push_back(images[pair.idx1].keypoints[m.queryIdx].pt);
                p2.push_back(images[pair.idx2].keypoints[m.trainIdx].pt);
            }

            // Remove radial distortion, keeping pixel units (P = K)
            if (k1 != 0.0) {
                undistortPoints(vector<Point2f>(p1), p1, K, dist, noArray(), K);
                undistortPoints(vector<Point2f>(p2), p2, K, dist, noArray(), K);
            }

            Mat mask;
            Mat E = findEssentialMat(p1, p2, K, RANSAC, 0.999, 1.0, 1000, mask);

            if (E.rows != 3 || E.cols != 3 || mask.empty())
                continue;

            fractions.push_back(double(countNonZero(mask)) / p1.size());

            // Sampson distance in pixels under F = K^-T E K^-1
            Matx33d F = Kinv.t() * Matx33d(E) * Kinv;
            for (size_t i = 0; i < p1.size(); i++) {
                Vec3d x1(p1[i].x, p1[i].y, 1), x2(p2[i].x, p2[i].y, 1);
                Vec3d Fx1 = F * x1, Ftx2 = F.t() * x2;
                double num = x2.dot(Fx1);
                double den = Fx1[0] * Fx1[0] + Fx1[1] * Fx1[1] +
                             Ftx2[0] * Ftx2[0] + Ftx2[1] * Ftx2[1];
                if (den > 0)
                    sampson.push_back(abs(num) / sqrt(den));
            }
        }

        Stats fs = computeStats(fractions);
        Stats ss = computeStats(sampson);

        cout << "k1 = " << setw(5) << k1
                << " | f = " << setw(6) << f
                << " | pairs " << fs.count
                << " | mean inlier frac " << fixed << setprecision(4) << fs.mean
                << " | median " << fs.median
                << " | median Sampson " << ss.median << " px\n";
        cout.unsetf(ios::floatfield);

        file << f << ',' << k1 << ',' << fs.count << ',' << fs.mean << ','
                << fs.median << ',' << ss.median << '\n';
    }
}
