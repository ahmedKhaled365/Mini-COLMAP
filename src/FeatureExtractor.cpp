#include "FeatureExtractor.h"

#include <stdexcept>
#include <opencv2/features.hpp>

FeatureExtractor::FeatureExtractor(int maxFeatures) {
    sift = cv::SIFT::create(maxFeatures);
}

void FeatureExtractor::extract(ImageData &img) {
    if (img.image.empty()) {
        throw std::runtime_error("No image to extract from.");
    }

    sift->detectAndCompute(img.image, cv::noArray(), img.keypoints, img.descriptors);
}
