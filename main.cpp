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

using namespace std;
using namespace cv;

// --------------------------------------------------
// 1. Configuration
// --------------------------------------------------

vector<string> IMAGE_PATHS = {};

const bool VISUALIZE = false;
const int MAX_FEATURES = 3500;
const int MAX_IMAGES = 20;
const int MIN_INITIAL_POINTS = 15;
const bool USE_FIXED_INITIAL_PAIR = false;
const int INITIAL_ID1 = 6;
const int INITIAL_ID2 = 7;


// --------------------------------------------------
// 2. Load Images and Extract Features
// --------------------------------------------------

void lodaPaths() {
    for (int i = 1; i <= MAX_IMAGES; i++) {
        ostringstream filename;

        filename << "../data/horse/"
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
    if (images.size() < 2)
        throw runtime_error("At least 2 images required");

    const Size imageSize = images[0].image.size();

    for (const auto &img: images) {
        if (img.image.size() != imageSize)
            throw runtime_error("Different image resolutions");
    }

    double W = imageSize.width;
    double H = imageSize.height;
    double f = 0.7 * W;

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


void registerRemainingImages(
    Reconstruction &reconstruction,
    vector<ImageData> &images,
    vector<PairMatch> &verifiedPairs,
    Mat &K
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

            int registeredCount = 0;

            for (const auto &img: images) {
                if (img.registered)
                    ++registeredCount;
            }

            if (registeredCount == 5 ||
                registeredCount == 10 ||
                registeredCount == 15 ||
                registeredCount == 20 ||
                registeredCount == 30) {
                string filename =
                        "horse_" + to_string(registeredCount) + "_cameras.ply";

                reconstruction.exportPLY(filename);

                cout << "Checkpoint saved: " << filename << '\n';
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

int main() {
    try {
        FeatureExtractor extractor(MAX_FEATURES);
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


        // Step 4: Find initial pair
        InitialPairResult initial;

        if (USE_FIXED_INITIAL_PAIR) {
            for (int i = 0; i < static_cast<int>(verifiedPairs.size()); i++) {
                const auto &pair = verifiedPairs[i];

                if (pair.idx1 == INITIAL_ID1 &&
                    pair.idx2 == INITIAL_ID2) {
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

        // Step 6: Register remaining images
        registerRemainingImages(
            reconstruction,
            images,
            verifiedPairs,
            K
        );

        reconstruction.printTrackStatistics();
        reconstruction.exportPLY("horse_all.ply");

        reconstruction.exportFilteredPLY(
            "horse_3plus_views.ply",
            3
        );

        reconstruction.exportFilteredPLY(
            "horse_4plus_views.ply",
            4
        );

        // Step 7: Export point cloud before Bundle Adjustment
        if (!reconstruction.exportPLY("horse_before_BA.ply"))
            throw runtime_error("Failed to export before-BA PLY");

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

        // Step 10: Export final point cloud
        if (!reconstruction.exportPLY("horse_final.ply"))
            throw runtime_error("Failed to export final PLY");

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
