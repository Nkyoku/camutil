#pragma once

#include <opencv2/core.hpp>

// 白線強調フィルタ
class EnhancementFilter {
public:
    // コンストラクタ
    EnhancementFilter(double scale_factor = 1.0, int kernel_size = 3);

    // フィルタ処理を行う
    void compute(const cv::Mat &src, cv::Mat &dest);

    // 中間値画像を取得する
    const cv::Mat& medianedImage(void) const {
        return m_Medianed;
    }

private:
    // 縮小比率の逆数(1以下)
    double m_InversedScaleFactor;

    // 中間値フィルタのカーネルサイズ
    int m_KernelSize;

    // 縮小画像
    cv::Mat m_Shrinked;

    // 中間値画像
    cv::Mat m_Medianed;

    // 膨張画像
    cv::Mat m_Dilated;
};
