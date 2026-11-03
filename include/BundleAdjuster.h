#pragma once

#include <opencv2/opencv.hpp>

#include "ImageData.h"
#include "Triangulation.h"

class BundleAdjuster {
public:
    static bool optimize(
        vector<ImageData>& images,
        vector<Landmark>& landmarks,
        const Mat& K
    );
};