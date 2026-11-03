#include <bits/stdc++.h>
#include <stdexcept>
#include <utility>
#include <opencv2/opencv.hpp>

#include "ImageData.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Geometry.h"
#include "Triangulation.h"
#include "Reconstruction.h"
#include "Diagnostics.h"

#include <filesystem>

using namespace std;
using namespace cv;

// --------------------------------------------------
// 1. Configuration
// --------------------------------------------------

vector<string> IMAGE_PATHS = {};

const bool VISUALIZE = false;
const int MIN_INITIAL_POINTS = 15;

// Run configuration. Defaults are the best measured configuration
// (results/README.md); every value can be overridden from the command
// line so experiments are reproducible (see parseArguments).
struct RunConfig {
    string dataDir = "../data/horse";
    string outDir = "../results/latest";
    int maxImages = 20;
    int maxFeatures = 3500;

    // Focal length in pixels. <= 0 means 0.7 * image width (original guess,
    // 1344 px). EXIF (Sony A7S II, 21 mm, 35.6 mm sensor width) gives
    // ~1133 px; the Essential-matrix sweep and the 20-image A/B runs both
    // favour ~1200 px (see results/README.md), so that is the default here.
    double focalPx = 1200;

    // Initial pair selection: "parallax" (default) or "legacy"
    string initMode = "parallax";
    double maxHomographyRatio = 0.8; // reject if H inliers / E inliers is above
    double minInitAngle = 4.0;       // deg, minimum median triangulation angle
    double targetInitAngle = 8.0;    // deg, angle at which parallax stops adding score
    int minInitLandmarks = 100;
    double minInitTriangulatedRatio = 0.9; // triangulated / E inliers

    bool useFixedInitialPair = false;
    int initialId1 = 6;
    int initialId2 = 7;

    // Bundle adjustment: "none", "final" (one global BA at the end) or
    // "incremental" (local BA after every registration, periodic global BA)
    string baMode = "incremental";
    int localBaImages = 6;        // neighbors optimized with the new image
    double globalBaRatio = 1.25;  // global BA when registered count grows by 25%
    double huberScale = 2.0;      // px
    int baThreads = 1;            // 1 = reproducible, 0 = all cores
    bool refineFocal = true;      // shared focal length in global BA
    bool refineK1 = true;         // shared radial distortion in global BA
    int minCamerasForIntrinsics = 8;

    // Outlier filtering after each BA
    double filterMaxError = 4.0;  // px
    double filterMinAngle = 1.0;  // deg, same as the creation threshold

    // Outputs: "minimal" (reports, CSVs, preview), "standard" (+ all and
    // reliable points, cameras) or "full" (+ every PLY variant and a PLY at
    // each checkpoint). checkpoints.csv is always written.
    string outputLevel = "standard";
    bool binaryPly = true;

    // Diagnostics-only mode: sweep focal lengths and exit.
    bool focalSweep = false;
};

RunConfig CONFIG;

void printUsage() {
    cout << "Usage: Mini-COLMAP [options]\n"
            << "  --data DIR         image folder (default ../data/horse)\n"
            << "  --out DIR          output folder (default ../results/latest)\n"
            << "  --images N         number of images to use (default 20)\n"
            << "  --features N       SIFT features per image (default 3500)\n"
            << "  --focal F          focal length in pixels (default 1200; <= 0 uses 0.7 * width)\n"
            << "  --pair I J         force the initial pair (zero-based ids)\n"
            << "  --init MODE        initial pair selection: parallax | legacy\n"
            << "  --min-init-angle D minimum median triangulation angle [deg] (default 4)\n"
            << "  --max-h-ratio R    maximum homography / essential inlier ratio (default 0.8)\n"
            << "  --ba MODE          bundle adjustment: none | final | incremental\n"
            << "  --local-ba-images N neighbors in local BA (default 6)\n"
            << "  --no-refine-focal  keep the focal length fixed\n"
            << "  --no-refine-k1     keep k1 fixed (0)\n"
            << "  --huber D          Huber loss scale [px] (default 2)\n"
            << "  --ba-threads N     Ceres threads (default 1 = reproducible, 0 = all)\n"
            << "  --output LEVEL     minimal | standard | full (default standard)\n"
            << "  --ascii-ply        write ASCII instead of binary PLY files\n"
            << "  --ba-max-error E   post-BA observation filter [px] (default 4)\n"
            << "  --ba-min-angle D   post-BA triangulation angle filter [deg] (default 1)\n"
            << "  --focal-sweep      only run the focal-length sweep\n";
}

void parseArguments(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        auto next = [&]() -> string {
            if (i + 1 >= argc)
                throw runtime_error("Missing value for " + arg);
            return argv[++i];
        };

        if (arg == "--data") CONFIG.dataDir = next();
        else if (arg == "--out") CONFIG.outDir = next();
        else if (arg == "--images") CONFIG.maxImages = stoi(next());
        else if (arg == "--features") CONFIG.maxFeatures = stoi(next());
        else if (arg == "--focal") CONFIG.focalPx = stod(next());
        else if (arg == "--pair") {
            CONFIG.useFixedInitialPair = true;
            CONFIG.initialId1 = stoi(next());
            CONFIG.initialId2 = stoi(next());
        } else if (arg == "--init") CONFIG.initMode = next();
        else if (arg == "--min-init-angle") CONFIG.minInitAngle = stod(next());
        else if (arg == "--max-h-ratio") CONFIG.maxHomographyRatio = stod(next());
        else if (arg == "--ba") CONFIG.baMode = next();
        else if (arg == "--local-ba-images") CONFIG.localBaImages = stoi(next());
        else if (arg == "--refine-focal") CONFIG.refineFocal = true;
        else if (arg == "--no-refine-focal") CONFIG.refineFocal = false;
        else if (arg == "--no-refine-k1") CONFIG.refineK1 = false;
        else if (arg == "--refine-k1") CONFIG.refineK1 = true;
        else if (arg == "--huber") CONFIG.huberScale = stod(next());
        else if (arg == "--ba-threads") CONFIG.baThreads = stoi(next());
        else if (arg == "--output") CONFIG.outputLevel = next();
        else if (arg == "--ascii-ply") CONFIG.binaryPly = false;
        else if (arg == "--ba-max-error") CONFIG.filterMaxError = stod(next());
        else if (arg == "--ba-min-angle") CONFIG.filterMinAngle = stod(next());
        else if (arg == "--focal-sweep") CONFIG.focalSweep = true;
        else if (arg == "--help" || arg == "-h") {
            printUsage();
            exit(0);
        } else {
            printUsage();
            throw runtime_error("Unknown argument: " + arg);
        }
    }
}

string configDescription(const Mat &K) {
    ostringstream os;
    os << "Mini-COLMAP run\n"
            << "  data: " << CONFIG.dataDir << '\n'
            << "  images: " << CONFIG.maxImages << '\n'
            << "  features: " << CONFIG.maxFeatures << '\n'
            << "  focal: " << K.at<double>(0, 0)
            << " px (cx " << K.at<double>(0, 2)
            << ", cy " << K.at<double>(1, 2) << ")\n"
            << "  initial pair: "
            << (CONFIG.useFixedInitialPair
                    ? to_string(CONFIG.initialId1) + "-" + to_string(CONFIG.initialId2)
                    : "automatic (" + CONFIG.initMode + ")") << '\n'
            << "  init gates: H/E <= " << CONFIG.maxHomographyRatio
            << ", median angle >= " << CONFIG.minInitAngle
            << " deg, landmarks >= " << CONFIG.minInitLandmarks
            << ", triangulated/E >= " << CONFIG.minInitTriangulatedRatio
            << ", target angle " << CONFIG.targetInitAngle << " deg\n"
            << "  bundle adjustment: " << CONFIG.baMode
            << " | local images " << CONFIG.localBaImages
            << " | global ratio " << CONFIG.globalBaRatio
            << " | huber " << CONFIG.huberScale
            << " | refine focal " << CONFIG.refineFocal
            << " | refine k1 " << CONFIG.refineK1 << '\n'
            << "  post-BA filter: max error " << CONFIG.filterMaxError
            << " px, min angle " << CONFIG.filterMinAngle << " deg\n";
    return os.str();
}

string outPath(const string &name) {
    return (filesystem::path(CONFIG.outDir) / name).string();
}


// --------------------------------------------------
// 2. Load Images and Extract Features
// --------------------------------------------------

void lodaPaths() {
    for (int i = 1; i <= CONFIG.maxImages; i++) {
        ostringstream filename;

        filename << CONFIG.dataDir << "/"
                << setw(5) << setfill('0') << i
                << ".jpg";

        IMAGE_PATHS.push_back(filename.str());
    }
}

vector<ImageData> loadImages(
    const vector<string> &paths,
    FeatureExtractor &extractor
) {
    vector<ImageData> images;
    images.reserve(paths.size());

    for (int i = 0; i < static_cast<int>(paths.size()); i++) {
        ImageData img;

        img.id = i;
        img.path = paths[i];
        img.image = imread(img.path);

        if (img.image.empty())
            throw runtime_error("Cannot load image: " + img.path);

        extractor.extract(img);

        img.rawPoints.reserve(img.keypoints.size());
        for (const auto &kp: img.keypoints)
            img.rawPoints.push_back(kp.pt);

        cout << "Image " << i
                << " | Features: " << img.keypoints.size()
                << '\n';

        images.push_back(move(img));
    }

    return images;
}

// --------------------------------------------------
// 3. Estimate Camera Intrinsics
// --------------------------------------------------

Mat createCameraMatrix(const vector<ImageData> &images) {

    const Size imageSize = images[0].image.size();

    for (const auto &img: images) {
        if (img.image.size() != imageSize)
            throw runtime_error("Different image resolutions");
    }

    double W = imageSize.width, H = imageSize.height;
    double f = CONFIG.focalPx > 0 ? CONFIG.focalPx : 0.7 * W;

    return (Mat_<double>(3, 3) <<
            f, 0, W / 2.0,
            0, f, H / 2.0,
            0, 0, 1
    );
}


// --------------------------------------------------
// 4. Feature Matching and Geometric Verification
// --------------------------------------------------


vector<PairMatch> matchImagePairs(
    const vector<ImageData> &images,
    FeatureMatcher &matcher
) {
    /// vector contains all the matches between all images.
    vector<PairMatch> verifiedPairs;

    const int MATCH_WINDOW = 8;
    const int MIN_INLIERS = 30;

    for (int i = 0; i < static_cast<int>(images.size()); i++) {
        int end = min(
            static_cast<int>(images.size()),
            i + MATCH_WINDOW + 1
        );

        for (int j = i + 1; j < end; j++) {
            auto matches = matcher.match(images[i], images[j]);

            if (matches.size() < MIN_INLIERS)
                continue;

            auto inliers = matcher.geometricVerification(
                images[i],
                images[j],
                matches
            );

            if (inliers.size() < MIN_INLIERS)
                continue;

            cout << "Pair " << i << " <-> " << j
                    << " | Inliers: " << inliers.size()
                    << '\n';

            verifiedPairs.push_back({
                i,
                j,
                move(inliers)
            });
        }
    }

    cout << "\nMatching Completed"
            << " | Total Verified Pairs: "
            << verifiedPairs.size()
            << endl;

    return verifiedPairs;
}


// --------------------------------------------------
// 5. Select Initial Pair
// --------------------------------------------------

struct InitialPairResult {
    int pairIndex = -1;
    PoseResult pose;
    vector<Landmark> landmarks;
};


InitialPairResult selectInitialPair(
    vector<ImageData> &images,
    vector<PairMatch> &verifiedPairs,
    Mat &K,
    Geometry &geometry,
    Triangulation &triangulation,
    vector<InitCandidateRecord> &records
) {
    const int MAX_CANDIDATES = 100;
    const int MIN_INITIAL_MATCHES = 100;

    struct Candidate {
        InitCandidateRecord record;
        int pairIndex;
        PoseResult pose;
        vector<Landmark> landmarks;
    };

    vector<Candidate> candidates;

    // Iterate in order of verified match count (iteration order only;
    // the match count is not used for the final decision in parallax mode)
    vector<int> pairIndices(verifiedPairs.size());
    iota(pairIndices.begin(), pairIndices.end(), 0);

    sort(pairIndices.begin(), pairIndices.end(),
         [&](int a, int b) {
             return verifiedPairs[a].matches.size() >
                    verifiedPairs[b].matches.size();
         }
    );

    int candidatesToTry = min(
        MAX_CANDIDATES,
        static_cast<int>(pairIndices.size())
    );

    for (int k = 0; k < candidatesToTry; k++) {
        int pairIndex = pairIndices[k];
        const auto &pair = verifiedPairs[pairIndex];

        if (pair.matches.size() < MIN_INITIAL_MATCHES)
            continue;

        Candidate c;
        c.pairIndex = pairIndex;
        c.record.idx1 = pair.idx1;
        c.record.idx2 = pair.idx2;
        c.record.fInliers = static_cast<int>(pair.matches.size());

        try {
            c.pose = geometry.estimateInitialPose(
                images[pair.idx1],
                images[pair.idx2],
                pair.matches,
                K
            );

            TwoViewStats stats;
            c.landmarks = triangulation.triangulate(
                images[pair.idx1],
                images[pair.idx2],
                c.pose,
                K,
                &stats
            );

            c.record.eInliers = c.pose.essentialInliers;
            c.record.hInliers = c.pose.homographyInliers;
            c.record.hOverE = c.pose.essentialInliers > 0
                                  ? double(c.pose.homographyInliers) / c.pose.essentialInliers
                                  : 1.0;
            c.record.triangulated = stats.triangulated;
            c.record.landmarks = static_cast<int>(c.landmarks.size());
            c.record.medianAngle = stats.medianAngle;
        } catch (const cv::Exception &) {
            c.record.reason = "opencv_error";
        } catch (const std::runtime_error &) {
            c.record.reason = "pose_failed";
        }

        candidates.push_back(move(c));
    }

    // Selection.
    // legacy:   most surviving landmarks (original behaviour).
    // parallax: reject rotation-dominated / planar pairs (a homography explains
    //           the matches almost as well as the Essential Matrix) and pairs
    //           with a weak median triangulation angle, then prefer
    //           well-conditioned pairs:
    //           score = landmarks * min(1, medianAngle / targetAngle).
    //           If nothing passes, the angle gate is relaxed step by step.
    int bestIndex = -1;
    const bool legacy = CONFIG.initMode == "legacy";

    for (double minAngle = CONFIG.minInitAngle;
         bestIndex == -1 && minAngle >= 0.99;
         minAngle *= 0.5) {
        double bestScore = 0;

        for (int i = 0; i < static_cast<int>(candidates.size()); i++) {
            auto &r = candidates[i].record;

            if (r.reason == "opencv_error" || r.reason == "pose_failed")
                continue;

            if (legacy) {
                r.score = r.landmarks;
                r.reason = "evaluated";
            } else if (r.landmarks < CONFIG.minInitLandmarks) {
                r.reason = "few_landmarks";
                continue;
            } else if (r.triangulated < CONFIG.minInitTriangulatedRatio * r.eInliers) {
                // A correct pose places nearly all E inliers in front of both
                // cameras with small reprojection error; a large deficit means
                // the decomposition is ill-conditioned.
                r.reason = "pose_inconsistent";
                continue;
            } else if (r.hOverE > CONFIG.maxHomographyRatio) {
                r.reason = "homography_dominant";
                continue;
            } else if (r.medianAngle < minAngle) {
                r.reason = "low_parallax";
                continue;
            } else {
                r.score = r.landmarks * min(1.0, r.medianAngle / CONFIG.targetInitAngle);
                r.reason = "passed";
            }

            if (r.score > bestScore) {
                bestScore = r.score;
                bestIndex = i;
            }
        }

        if (legacy)
            break;

        if (bestIndex == -1)
            cerr << "No initial pair passed with median angle >= " << minAngle
                    << " deg, relaxing\n";
    }

    for (const auto &c: candidates)
        records.push_back(c.record);

    if (bestIndex == -1 ||
        candidates[bestIndex].landmarks.size() < MIN_INITIAL_POINTS) {
        throw runtime_error("No suitable initial pair found");
    }

    records[bestIndex].accepted = true;

    const auto &r = candidates[bestIndex].record;
    cout << fixed << setprecision(2)
            << "[Init] " << (legacy ? "legacy" : "parallax")
            << " | evaluated " << candidates.size()
            << " | selected " << r.idx1 << "-" << r.idx2
            << " | F " << r.fInliers << " E " << r.eInliers
            << " H/E " << r.hOverE
            << " | median angle " << r.medianAngle
            << " | landmarks " << r.landmarks << '\n';
    cout.unsetf(ios::floatfield);

    InitialPairResult best;
    best.pairIndex = candidates[bestIndex].pairIndex;
    best.pose = move(candidates[bestIndex].pose);
    best.landmarks = move(candidates[bestIndex].landmarks);

    return best;
}


// --------------------------------------------------
// 6. Optional Visualization
// --------------------------------------------------

void showMatches(
    const ImageData &img1,
    const ImageData &img2,
    const vector<DMatch> &matches
) {
    if (matches.empty())
        return;

    Mat visualization;

    drawMatches(
        img1.image, img1.keypoints,
        img2.image, img2.keypoints,
        matches, visualization
    );

    resize(visualization, visualization, Size(1200, 600));

    imshow("Initial Pair - Pose Inliers", visualization);
    waitKey(0);
    destroyAllWindows();
}


// --------------------------------------------------
// 7. Incremental Reconstruction
// --------------------------------------------------


void saveCheckpoint(
    Reconstruction &reconstruction,
    const Diagnostics &diagnostics,
    const Mat &K,
    const string &name
) {
    const auto &images = reconstruction.getImages();
    const auto &landmarks = reconstruction.getLandmarks();

    auto report = diagnostics.analyze(images, landmarks, K);
    Diagnostics::appendCheckpointCsv(report, outPath("checkpoints.csv"));

    if (CONFIG.outputLevel == "full")
        diagnostics.exportPoints(outPath(name + ".ply"), images, landmarks, K);

    cout << fixed << setprecision(3)
            << "[Checkpoint] " << name
            << " | landmarks " << report.landmarks
            << " | reproj median " << report.reprojection.median
            << " P95 " << report.reprojection.p95
            << " | track mean " << report.trackLength.mean
            << " | angle median " << report.triAngle.median
            << " | reliable " << report.reliableLandmarks << '\n';
    cout.unsetf(ios::floatfield);
}

// Runs one bundle adjustment followed by outlier filtering and prints a
// one-line summary. imageId < 0 means global BA.
void runBundleAdjustment(
    Reconstruction &reconstruction,
    Mat &K,
    int imageId,
    bool allowIntrinsics
) {
    int registered = 0;
    for (const auto &img: reconstruction.getImages()) {
        if (img.registered)
            registered++;
    }

    BAOptions options;
    options.huberScale = CONFIG.huberScale;
    options.numThreads = CONFIG.baThreads;

    BAResult result;

    if (imageId < 0) {
        const bool intrinsics = allowIntrinsics &&
                                registered >= CONFIG.minCamerasForIntrinsics;
        options.label = "global";
        options.maxIterations = 100;
        options.refineFocal = intrinsics && CONFIG.refineFocal;
        options.refineK1 = intrinsics && CONFIG.refineK1;
        result = reconstruction.globalBundleAdjustment(K, options);
    } else {
        options.label = "local";
        options.maxIterations = 50;
        result = reconstruction.localBundleAdjustment(
            imageId, K, options, CONFIG.localBaImages
        );
    }

    auto &entry = reconstruction.getBALog().back();

    if (result.applied) {
        auto [observations, removed] = reconstruction.filterLandmarks(
            {}, K, CONFIG.filterMaxError, CONFIG.filterMinAngle
        );
        entry.filteredObservations = observations;
        entry.removedLandmarks = removed;
    }

    cout << fixed << setprecision(3)
            << "[BA " << entry.label << "] cams " << entry.variableCameras
            << "+" << entry.constantCameras << " const | points " << entry.points
            << " | RMS " << entry.rmsBefore << " -> " << entry.rmsAfter
            << " px | " << entry.termination << " (" << entry.iterations << " it)"
            << " | " << (entry.applied ? "applied" : "REJECTED: " + entry.reason)
            << " | filtered obs " << entry.filteredObservations
            << ", landmarks " << entry.removedLandmarks;
    if (options.refineFocal || options.refineK1)
        cout << " | f " << entry.focalAfter << " k1 " << setprecision(4) << entry.k1After;
    cout << '\n';
    cout.unsetf(ios::floatfield);
}

void registerRemainingImages(
    Reconstruction &reconstruction,
    vector<ImageData> &images,
    vector<PairMatch> &verifiedPairs,
    Mat &K,
    const Diagnostics &diagnostics
) {
    // Registered count at the last global BA (incremental mode)
    int lastGlobal = 2;
    constexpr int MIN_CORRESPONDENCES = 10;

    int totalRegistered = 0;

    while (true) {
        // Score all unregistered images
        vector<pair<int, int> > candidates;

        for (int imageId = 0;
             imageId < static_cast<int>(images.size());
             imageId++) {
            if (images[imageId].registered)
                continue;

            int score = reconstruction.countCorrespondences(
                imageId,
                verifiedPairs
            );

            if (score >= MIN_CORRESPONDENCES) {
                candidates.push_back({score, imageId});
            }
        }

        if (candidates.empty())
            break;

        // Highest correspondence count first
        sort(candidates.begin(), candidates.end(), greater<>());

        bool registeredOne = false;

        for (const auto &[score, imageId]: candidates) {
            cout << "\nTrying Image " << imageId
                    << " | Correspondences: " << score
                    << endl;

            if (!reconstruction.registerImage(
                imageId, verifiedPairs, K
            )) {
                cout << "Image " << imageId
                        << " | Registration Failed\n";
                continue;
            }

            int newPoints = reconstruction.triangulateNewPoints(
                imageId,
                verifiedPairs,
                K
            );

            auto &log = reconstruction.getRegistrationLog();
            if (!log.empty() && log.back().imageId == imageId)
                log.back().newLandmarks = newPoints;

            int registeredCount = 0;

            for (const auto &img: images) {
                if (img.registered)
                    ++registeredCount;
            }

            if (CONFIG.baMode == "incremental") {
                runBundleAdjustment(reconstruction, K, imageId, false);

                if (registeredCount >= ceil(lastGlobal * CONFIG.globalBaRatio)) {
                    runBundleAdjustment(reconstruction, K, -1, true);
                    lastGlobal = registeredCount;
                }
            }

            if (registeredCount == 5 ||
                registeredCount % 10 == 0) {
                ostringstream name;
                name << "checkpoint_" << setw(3) << setfill('0')
                        << registeredCount << "_cameras";

                saveCheckpoint(reconstruction, diagnostics, K, name.str());
            }

            cout << "Image " << imageId
                    << " | Registered"
                    << " | New Landmarks: " << newPoints
                    << '\n';

            totalRegistered++;
            registeredOne = true;

            // Recalculate scores after the map expands
            break;
        }

        // Stop if none of the candidates can register
        if (!registeredOne)
            break;
    }

    int registeredCount = 0;

    for (const auto &img: images) {
        if (img.registered)
            registeredCount++;
    }

    cout << "\n========== FINAL RESULTS ==========\n";

    cout << "Registered Images: "
            << registeredCount << " / " << images.size()
            << '\n';

    cout << "Newly Registered: "
            << totalRegistered << '\n';

    cout << "Total 3D Landmarks: "
            << reconstruction.getLandmarkCount()
            << '\n';
}

// --------------------------------------------------
// 8. Main
// --------------------------------------------------

int main(int argc, char **argv) {
    try {
        parseArguments(argc, argv);
        filesystem::create_directories(CONFIG.outDir);
        filesystem::remove(outPath("checkpoints.csv"));

        FeatureExtractor extractor(CONFIG.maxFeatures);
        FeatureMatcher matcher;
        Geometry geometry;
        Triangulation triangulation;

        // Step 1: Load and extract features
        lodaPaths();
        vector<ImageData> images = loadImages(
            IMAGE_PATHS,
            extractor
        );

        // Step 2: Camera intrinsics
        Mat K = createCameraMatrix(images);

        // Step 3: Match image pairs
        vector<PairMatch> verifiedPairs = matchImagePairs(
            images,
            matcher
        );

        if (CONFIG.focalSweep) {
            vector<double> focals;
            for (double f = 900; f <= 1500; f += 50)
                focals.push_back(f);
            focals.push_back(1069.0); // EXIF, 35 mm-equivalent diagonal rule
            focals.push_back(1132.6); // EXIF, 21 mm / 35.6 mm sensor width
            focals.push_back(1344.0); // original 0.7 * width guess
            sort(focals.begin(), focals.end());

            Diagnostics::focalSweep(
                images, verifiedPairs, focals,
                {0.0, -0.02, -0.04, -0.06, -0.08, -0.12}, 3,
                outPath("focal_sweep.csv")
            );
            return 0;
        }


        // Step 4: Find initial pair
        InitialPairResult initial;

        if (CONFIG.useFixedInitialPair) {
            for (int i = 0; i < static_cast<int>(verifiedPairs.size()); i++) {
                const auto &pair = verifiedPairs[i];

                if (pair.idx1 == CONFIG.initialId1 &&
                    pair.idx2 == CONFIG.initialId2) {
                    initial.pairIndex = i;

                    initial.pose = geometry.estimateInitialPose(
                        images[pair.idx1],
                        images[pair.idx2],
                        pair.matches,
                        K
                    );

                    initial.landmarks = triangulation.triangulate(
                        images[pair.idx1],
                        images[pair.idx2],
                        initial.pose,
                        K
                    );

                    break;
                }
            }

            if (initial.pairIndex == -1 ||
                initial.landmarks.size() < MIN_INITIAL_POINTS)
                throw runtime_error("Fixed initial pair is invalid");
        } else {
            vector<InitCandidateRecord> initRecords;
            initial = selectInitialPair(
                images,
                verifiedPairs,
                K,
                geometry,
                triangulation,
                initRecords
            );
            Diagnostics::writeInitCandidatesCsv(
                initRecords, outPath("init_candidates.csv")
            );
        }

        const auto &startingPair =
                verifiedPairs[initial.pairIndex];

        cout << "\n========== INITIAL RECONSTRUCTION ==========\n";
        cout << "Selected Pair: "
                << startingPair.idx1 << " <-> "
                << startingPair.idx2 << '\n';
        cout << "Pose Inliers: "
                << initial.pose.inlierMatches.size() << '\n';
        cout << "Initial Landmarks: "
                << initial.landmarks.size() << '\n';
        cout << "K:\n" << K << '\n';


        Reconstruction reconstruction(images);

        // Step 5: Initialize reconstruction
        reconstruction.initialize(
            startingPair.idx1,
            startingPair.idx2,
            initial.pose,
            initial.landmarks
        );

        if (VISUALIZE) {
            showMatches(
                images[startingPair.idx1],
                images[startingPair.idx2],
                initial.pose.inlierMatches
            );
        }

        const double initialFocal = K.at<double>(0, 0);

        // Refine the initial two-view reconstruction (gauge: anchor pair)
        if (CONFIG.baMode == "incremental")
            runBundleAdjustment(reconstruction, K, -1, false);

        DiagnosticsConfig diagnosticsConfig;
        diagnosticsConfig.binaryPly = CONFIG.binaryPly;
        Diagnostics diagnostics(diagnosticsConfig);
        saveCheckpoint(reconstruction, diagnostics, K, "checkpoint_002_cameras");

        // Step 6: Register remaining images
        registerRemainingImages(
            reconstruction,
            images,
            verifiedPairs,
            K,
            diagnostics
        );

        // Step 7: Final global Bundle Adjustment
        if (CONFIG.baMode == "incremental" || CONFIG.baMode == "final")
            runBundleAdjustment(reconstruction, K, -1, true);

        BundleAdjuster::writeLog(reconstruction.getBALog(), outPath("ba_log.csv"));

        reconstruction.printTrackStatistics();

        // Step 7: Diagnostics and exports (all written to CONFIG.outDir)
        const auto &landmarks = reconstruction.getLandmarks();
        auto report = diagnostics.analyze(images, landmarks, K);

        Diagnostics::printSummary(report, cout);

        ostringstream header;
        header << configDescription(K)
                << "  selected initial pair: " << startingPair.idx1
                << "-" << startingPair.idx2
                << " | initial landmarks " << initial.landmarks.size() << '\n'
                << "  intrinsics: initial f " << initialFocal
                << " px | final f " << K.at<double>(0, 0)
                << " px | final k1 " << reconstruction.getK1() << '\n';

        Diagnostics::writeReport(report, outPath("report.txt"), header.str());
        Diagnostics::writeRegistrationCsv(
            reconstruction.getRegistrationLog(), outPath("registration.csv")
        );

        if (CONFIG.outputLevel == "standard" || CONFIG.outputLevel == "full") {
            diagnostics.exportPoints(outPath("points_all.ply"), images, landmarks, K, 2);
            diagnostics.exportPoints(outPath("points_reliable.ply"), images, landmarks, K, 2, 1);
            diagnostics.exportCameras(outPath("cameras.ply"), images, landmarks, K, false);
        }

        if (CONFIG.outputLevel == "full") {
            diagnostics.exportPoints(outPath("points_3plus.ply"), images, landmarks, K, 3);
            diagnostics.exportPoints(outPath("points_4plus.ply"), images, landmarks, K, 4);
            diagnostics.exportPoints(outPath("points_unreliable.ply"), images, landmarks, K, 2, 2);
            diagnostics.exportCameras(outPath("scene_with_cameras.ply"), images, landmarks, K, true);
        }
        diagnostics.renderPreview(
            outPath("preview.png"), images, landmarks, K,
            "f=" + to_string(cvRound(K.at<double>(0, 0))) +
            " init " + to_string(startingPair.idx1) + "-" + to_string(startingPair.idx2) +
            " | " + to_string(landmarks.size()) + " landmarks, " +
            to_string(report.reliableLandmarks) + " reliable"
        );

        cout << "\nOutputs written to: " << CONFIG.outDir << '\n';

        cout << "\nTotal 3D Landmarks: "
                << reconstruction.getLandmarkCount()
                << '\n';

        cout << "\nReconstruction finished!\n";
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
