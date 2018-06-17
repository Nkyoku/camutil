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

    // 偏差の最大値を設定する
    void setMaximumDisparity(int max_disparity);

    // ステレオマッチングを行う
    // left_image, right_imageに入力としてCV_8UC3型のBGR画像を与えると
    // disparity_mapにCV_32F型の偏差マップが出力される
    bool compute(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &disparity_map, cv::Mat &likelihood_map);





private:
    // フィルタのカーネルサイズ
    static constexpr int kFilterKernelSize = 21;

    // フィルタの分散
    static constexpr double kFilterVariance = 4.0;


    // 偏差の最大値
    int m_MaxDisparity = 1;

    // フィルタ
    GaussianDoG *m_GaussianDoG[2];

    // モノクロ画像
    cv::Mat m_MonochromeImage[2];


    // コストボリューム
    std::vector<cv::Mat> m_CostVolume;

    

    // ある偏差でのコスト画像を計算する
    void calculateCostImage(const cv::Mat &left_image, const cv::Mat &right_image, int disparity, cv::Mat &cost_image);

    // コストボリュームから偏差マップを生成する
    void calculateDisparity(cv::Mat &disparity_map, cv::Mat &likelihood_map);

};
