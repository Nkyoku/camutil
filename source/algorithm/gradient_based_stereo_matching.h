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

    // ステレオマッチングを行う
    // left_image, right_imageにモノクロ画像を与え偏差マップを得る
    void compute(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &disparity_map, int max_disparity);


private:
    // フィルタのカーネルサイズ
    static constexpr int kFilterKernelSize = 21;

    // フィルタの分散
    static constexpr double kFilterVariance = 3.0;

    // フィルタ
    GaussianDoG *m_GaussianDoG[2];

};
