#include "Retrieval.h"

using namespace std;
using namespace cv;

void Retrieval::build(
    const vector<ImageData> &images,
    int numWords,
    int sampleSize
) {
    // --- Sample descriptors (fixed seed) ---------------------------------
    int total = 0;
    for (const auto &img: images)
        total += img.descriptors.rows;

    RNG rng(0x5EED);
    const double keepProbability = min(1.0, double(sampleSize) / max(1, total));

    Mat samples;
    for (const auto &img: images) {
        for (int r = 0; r < img.descriptors.rows; r++) {
            if (rng.uniform(0.0, 1.0) < keepProbability)
                samples.push_back(img.descriptors.row(r));
        }
    }

    samples.convertTo(samples, CV_32F);
    numWords = min(numWords, samples.rows);

    // --- Vocabulary (k-means, deterministic seed) ---------------------------
    Mat labels;
    theRNG().state = 0x5EED;
    kmeans(
        samples, numWords, labels,
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-3),
        1, KMEANS_PP_CENTERS, vocabulary
    );

    // --- Word histograms -----------------------------------------------------
    BFMatcher assigner(NORM_L2);
    vector<Mat> histograms;
    vector<int> documentFrequency(numWords, 0);

    for (const auto &img: images) {
        Mat histogram = Mat::zeros(1, numWords, CV_32F);

        if (!img.descriptors.empty()) {
            Mat descriptors;
            img.descriptors.convertTo(descriptors, CV_32F);

            vector<DMatch> words;
            assigner.match(descriptors, vocabulary, words);

            for (const auto &w: words)
                histogram.at<float>(w.trainIdx) += 1.0f;
        }

        for (int w = 0; w < numWords; w++) {
            if (histogram.at<float>(w) > 0)
                documentFrequency[w]++;
        }

        histograms.push_back(histogram);
    }

    // --- tf-idf weighting and L2 normalization -------------------------------
    const double N = images.size();
    imageVectors.clear();

    for (auto &histogram: histograms) {
        const double words = max(1.0, sum(histogram)[0]);

        for (int w = 0; w < numWords; w++) {
            double idf = log(N / max(1, documentFrequency[w]));
            histogram.at<float>(w) = static_cast<float>(histogram.at<float>(w) / words * idf);
        }

        double n = norm(histogram);
        if (n > 0)
            histogram /= n;

        imageVectors.push_back(histogram);
    }
}

double Retrieval::similarity(int a, int b) const {
    return imageVectors[a].dot(imageVectors[b]);
}

vector<pair<int, int> > Retrieval::candidatePairs(int k, int minGap) const {
    set<pair<int, int> > pairs;
    const int n = static_cast<int>(imageVectors.size());

    for (int i = 0; i < n; i++) {
        vector<pair<double, int> > ranked;

        for (int j = 0; j < n; j++) {
            if (abs(i - j) > minGap)
                ranked.push_back({similarity(i, j), j});
        }

        sort(ranked.begin(), ranked.end(), greater<>());

        for (int r = 0; r < min(k, static_cast<int>(ranked.size())); r++)
            pairs.insert({min(i, ranked[r].second), max(i, ranked[r].second)});
    }

    return {pairs.begin(), pairs.end()};
}
