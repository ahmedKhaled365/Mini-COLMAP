#pragma once

#include "ImageData.h"

class FeatureExtractor {
private:
    Ptr<SIFT> sift;
public:
    explicit FeatureExtractor(int maxFeatures = 3500);

    void extract(ImageData& image);
};