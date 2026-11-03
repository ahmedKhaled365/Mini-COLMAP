#pragma once

#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

struct ImageData {
    int id = -1;
    string path;

    Mat image;
    vector<KeyPoint> keypoints;
    Mat descriptors;

    Mat R, t;

    bool registered = false;
};

struct PairMatch {
    int idx1, idx2;

    vector<DMatch> matches;
};
