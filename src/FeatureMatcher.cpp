#include "FeatureMatcher.h"
#include <opencv2/calib3d.hpp>


vector<DMatch> FeatureMatcher::match(
    const ImageData& img1,
    const ImageData& img2
) const {
    vector<DMatch> goodMatches;

    if (img1.descriptors.empty() ||
        img2.descriptors.empty() ||
        img1.descriptors.rows < 2 ||
        img2.descriptors.rows < 2)
        return goodMatches;

    FlannBasedMatcher matcher;

    vector<vector<DMatch>> forwardMatches;
    vector<vector<DMatch>> backwardMatches;

    matcher.knnMatch(
        img1.descriptors,
        img2.descriptors,
        forwardMatches,
        2
    );

    matcher.knnMatch(
        img2.descriptors,
        img1.descriptors,
        backwardMatches,
        2
    );

    constexpr float ratio = 0.75f;

    vector<int> reverseBest(
        img2.descriptors.rows,
        -1
    );

    for (const auto& candidates : backwardMatches) {
        if (candidates.size() < 2)
            continue;

        const DMatch& best = candidates[0];
        const DMatch& second = candidates[1];

        if (best.distance < ratio * second.distance) {
            reverseBest[best.queryIdx] = best.trainIdx;
        }
    }

    for (const auto& candidates : forwardMatches) {
        if (candidates.size() < 2)
            continue;

        const DMatch& best = candidates[0];
        const DMatch& second = candidates[1];

        if (best.distance >= ratio * second.distance)
            continue;

        if (reverseBest[best.trainIdx] != best.queryIdx)
            continue;

        goodMatches.push_back(best);
    }

    return goodMatches;
}

vector<DMatch> FeatureMatcher::geometricVerification(const ImageData &img1, const ImageData &img2,
                                                              const vector<cv::DMatch> &matches) const {
    vector<DMatch> inlierMatches;

    if (matches.size() < 8)
        return inlierMatches;

    vector<Point2f> points1, points2;

    for (const auto &m: matches) {
        points1.push_back(img1.keypoints[m.queryIdx].pt);
        points2.push_back(img2.keypoints[m.trainIdx].pt);
    }

    Mat mask;

    Mat F = findFundamentalMat(
        points1,
        points2,
        FM_RANSAC,
        1.0,
        0.99,
        mask
    );

    if (F.empty() || mask.empty())
        return inlierMatches;


    for (int i = 0; i < matches.size(); ++i) {
        if (mask.at<uchar>(i)) {
            inlierMatches.push_back(matches[i]);
        }
    }

    return inlierMatches;
}
