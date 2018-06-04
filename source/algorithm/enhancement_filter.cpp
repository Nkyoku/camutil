#include "enhancement_filter.h"
#include <opencv2/imgproc.hpp>

EnhancementFilter::EnhancementFilter(double scale_factor, int kernel_size)
    : m_InversedScaleFactor(1.0 / scale_factor), m_KernelSize(kernel_size)
{

}

void EnhancementFilter::compute(const cv::Mat &src, cv::Mat &dest) {
    int width = src.cols, height = src.rows;

    // 入力画像を縮小し中間値フィルタを通すことで白線や小さなオブジェクトを消去する
    cv::resize(src, m_Shrinked, cv::Size(), m_InversedScaleFactor, m_InversedScaleFactor);
    cv::medianBlur(m_Shrinked, m_Medianed, m_KernelSize);
    
    // 入力画像をdilateフィルタに通し白線を強調する
    cv::dilate(src, m_Dilated, cv::Mat());

    // 入力画像と中間値画像の差分を取って強調する
    // 差分を他のチャンネルに影響させることで明るいピクセルが白色に飽和するように仕向ける
    dest.create(height, width, CV_8UC3);
    m_Dilated.forEach<cv::Vec3b>([&](const cv::Vec3b &p, const int pos[2]) {
        const cv::Vec3b &base = m_Medianed.at<cv::Vec3b>(static_cast<int>(pos[0] * m_InversedScaleFactor), static_cast<int>(pos[1] * m_InversedScaleFactor));
        int b = p[0], g = p[1], r = p[2];
        int diff_b = b - base[0];
        int diff_g = g - base[1];
        int diff_r = r - base[2];
        int new_b = b + static_cast<int>(diff_b + 0.5 * diff_g + 0.5 * diff_r);
        int new_g = g + static_cast<int>(diff_g + 0.5 * diff_b + 0.5 * diff_r);
        int new_r = r + static_cast<int>(diff_r + 0.5 * diff_b + 0.5 * diff_g);
        int b_limit = std::min(std::max(new_b, 0), 255);
        int g_limit = std::min(std::max(new_g, 0), 255);
        int r_limit = std::min(std::max(new_r, 0), 255);
        dest.at<cv::Vec3b>(pos[0], pos[1]) = cv::Vec3b(b_limit, g_limit, r_limit);
    });
}
