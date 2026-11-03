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

// Run configuration. The defaults reproduce the original pipeline;
// every value can be overridden from the command line so experiments
// are reproducible (see parseArguments).
struct RunConfig {
    string dataDir = "../data/horse";
    string outDir = "../results/latest";
    int maxImages = 20;
    int maxFeatures = 3500;

    // Focal length in pixels. <= 0 means 0.7 * image width (original guess).
    double focalPx = 0;

    bool useFixedInitialPair = false;
    int initialId1 = 6;
    int initialId2 = 7;

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
            << "  --focal F          focal length in pixels (default 0.7 * width)\n"
            << "  --pair I J         force the initial pair (zero-based ids)\n"
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
        } else if (arg == "--focal-sweep") CONFIG.focalSweep = true;
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
                    : string("automatic")) << '\n';
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
    Triangulation &triangulation
) {
    InitialPairResult best;

    const int MAX_CANDIDATES = 100;
    const int MIN_INITIAL_MATCHES = 100;

    // Create an index list for all verified pairs
    vector<int> pairIndices(verifiedPairs.size());
    iota(pairIndices.begin(), pairIndices.end(), 0);

    // Sort pairs by the number of geometrically verified matches
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

    // Evaluate only the strongest candidate pairs
    for (int k = 0; k < candidatesToTry; k++) {
        int pairIndex = pairIndices[k];
        const auto &pair = verifiedPairs[pairIndex];

        if (pair.matches.size() < MIN_INITIAL_MATCHES)
            continue;

        try {
            // Estimate relative camera pose
            auto pose = geometry.estimateInitialPose(
                images[pair.idx1],
                images[pair.idx2],
                pair.matches,
                K
            );

            if (pose.inlierMatches.size() < MIN_INITIAL_POINTS)
                continue;

            // Triangulate initial 3D landmarks
            auto points = triangulation.triangulate(
                images[pair.idx1],
                images[pair.idx2],
                pose,
                K
            );

            cout << "Pair " << pair.idx1
                    << " <-> " << pair.idx2
                    << " | RANSAC: " << pair.matches.size()
                    << " | Pose: " << pose.inlierMatches.size()
                    << " | Landmarks: " << points.size()
                    << '\n';

            // Keep the pair with the most valid 3D landmarks
            if (points.size() > best.landmarks.size()) {
                best.pairIndex = pairIndex;
                best.pose = move(pose);
                best.landmarks = move(points);
            }
        } catch (const cv::Exception &e) {
            cerr << "Skipping pair "
                    << pair.idx1 << " <-> " << pair.idx2
                    << " | OpenCV error: " << e.what()
                    << '\n';
        } catch (const std::runtime_error &e) {
            cerr << "Skipping pair "
                    << pair.idx1 << " <-> " << pair.idx2
                    << " | Reason: " << e.what()
                    << '\n';
        }
    }

    if (best.pairIndex == -1 ||
        best.landmarks.size() < MIN_INITIAL_POINTS) {
        throw runtime_error("No suitable initial pair found");
    }

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

void registerRemainingImages(
    Reconstruction &reconstruction,
    vector<ImageData> &images,
    vector<PairMatch> &verifiedPairs,
    Mat &K,
    const Diagnostics &diagnostics
) {
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
                images, verifiedPairs, focals, 3, outPath("focal_sweep.csv")
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
            initial = selectInitialPair(
                images,
                verifiedPairs,
                K,
                geometry,
                triangulation
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

        Diagnostics diagnostics;
        saveCheckpoint(reconstruction, diagnostics, K, "checkpoint_002_cameras");

        // Step 6: Register remaining images
        registerRemainingImages(
            reconstruction,
            images,
            verifiedPairs,
            K,
            diagnostics
        );

        reconstruction.printTrackStatistics();

        // Step 7: Diagnostics and exports (all written to CONFIG.outDir)
        const auto &landmarks = reconstruction.getLandmarks();
        auto report = diagnostics.analyze(images, landmarks, K);

        Diagnostics::printSummary(report, cout);

        ostringstream header;
        header << configDescription(K)
                << "  selected initial pair: " << startingPair.idx1
                << "-" << startingPair.idx2
                << " | initial landmarks " << initial.landmarks.size() << '\n';

        Diagnostics::writeReport(report, outPath("report.txt"), header.str());
        Diagnostics::writeRegistrationCsv(
            reconstruction.getRegistrationLog(), outPath("registration.csv")
        );

        diagnostics.exportPoints(outPath("points_all.ply"), images, landmarks, K, 2);
        diagnostics.exportPoints(outPath("points_3plus.ply"), images, landmarks, K, 3);
        diagnostics.exportPoints(outPath("points_4plus.ply"), images, landmarks, K, 4);
        diagnostics.exportPoints(outPath("points_reliable.ply"), images, landmarks, K, 2, 1);
        diagnostics.exportPoints(outPath("points_unreliable.ply"), images, landmarks, K, 2, 2);
        diagnostics.exportCameras(outPath("cameras.ply"), images, landmarks, K, false);
        diagnostics.exportCameras(outPath("scene_with_cameras.ply"), images, landmarks, K, true);
        diagnostics.renderPreview(
            outPath("preview.png"), images, landmarks, K,
            "f=" + to_string(cvRound(K.at<double>(0, 0))) +
            " init " + to_string(startingPair.idx1) + "-" + to_string(startingPair.idx2) +
            " | " + to_string(landmarks.size()) + " landmarks, " +
            to_string(report.reliableLandmarks) + " reliable"
        );

        cout << "\nOutputs written to: " << CONFIG.outDir << '\n';

        // Step 8: Run Bundle Adjustment
        cout << "\nStarting Bundle Adjustment...\n";

        // bool optimized = reconstruction.runBundleAdjustment(K);

        // cout << "Bundle Adjustment Success: "
        //      << boolalpha << optimized << '\n';

        // Step 9: Export point cloud after Bundle Adjustment
        // if (optimized) {
        //     if (!reconstruction.exportPLY("horse_after_BA.ply"))
        //         throw runtime_error("Failed to export after-BA PLY");
        // } else {
        //     cerr << "Bundle Adjustment failed\n";
        // }

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
