#pragma once

#include "ImageData.h"
#include <opencv2/features2d.hpp>
#include <vector>

class FeatureMatcher {
public:
    std::vector<cv::DMatch> match(
        const ImageData &img1,
        const ImageData &img2
    ) const;

    std::vector<cv::DMatch> geometricVerification(
        const ImageData &img1,
        const ImageData &img2,
        const std::vector<cv::DMatch> &matches
    ) const;
};
