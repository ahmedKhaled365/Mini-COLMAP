#pragma once

#include "ImageData.h"

// --------------------------------------------------
// Image retrieval with a bag of visual words (tf-idf weighted).
// Used to find non-sequential image pairs that view the same part of the
// scene, e.g. the start and the end of an orbit (loop closure).
// --------------------------------------------------

class Retrieval {
public:
    // Builds a vocabulary of numWords visual words from at most sampleSize
    // randomly sampled descriptors (fixed seed, reproducible), then one
    // normalized tf-idf vector per image.
    void build(
        const vector<ImageData> &images,
        int numWords,
        int sampleSize
    );

    // Cosine similarity of the tf-idf vectors of two images.
    double similarity(int a, int b) const;

    // For every image, the k most similar images whose index differs by more
    // than minGap. Returns unique pairs (i < j).
    vector<pair<int, int> > candidatePairs(int k, int minGap) const;

private:
    Mat vocabulary;           // numWords x 128, CV_32F
    vector<Mat> imageVectors; // 1 x numWords, CV_32F, L2-normalized
};
