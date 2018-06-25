#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "algorithm/gaussian_dog.h"

// ステレオマッチングを行う
class GradientBasedStereoMatching {
public:
    // コンストラクタ
    GradientBasedStereoMatching(void);
    
    // デストラクタ
    ~GradientBasedStereoMatching();

    // ステレオマッチングの事前計算を行う
    void precompute(const cv::Mat &left_image, const cv::Mat &right_image);

    // ステレオマッチングを行う
    // left_image, right_imageにモノクロ画像を与え偏差マップを得る
    void compute(cv::Mat &disparity_map, int max_disparity) const;

    // ステレオマッチングを行う
    // left_image, right_imageにモノクロ画像を与えpointsで指定した点の偏差マップをpointsのZ値に返す
    void compute(std::vector<cv::Point3f> &points, int max_disparity) const;

private:
    // フィルタのカーネルサイズ
    static constexpr int kFilterKernelSize = 21;

    // フィルタの分散
    static constexpr double kFilterVariance = 3.0;

    // 勾配の重み
    static constexpr double kAlpha = 1.0;

    // DoGの重み
    static constexpr double kBeta = 1.0;

public:
    // フィルタ
    GaussianDoG *m_GaussianDoG[2];
};
