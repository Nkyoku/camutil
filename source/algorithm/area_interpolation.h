#pragma once

#include <opencv2/core.hpp>

// 領域内でのピクセルの補間を行う
class AreaInterpolation {
public:
   // ステレオマッチングを行う
    // left_image, right_imageに入力としてCV_8UC3型のBGR画像を与えると
    // disparity_mapにCV_32F型の偏差マップが出力される
    void compute(const cv::Mat &guide, const cv::Mat &src, cv::Mat &dst);





private:
    // エッジ検出した画像
    cv::Mat m_CannyImage;
};
