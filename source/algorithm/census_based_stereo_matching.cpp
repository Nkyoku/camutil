#include "census_based_stereo_matching.h"
#include "3rdparty/fastguidedfilter.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>
#include <intrin.h>
#include <limits>

CensusBasedStereoMatching::CensusBasedStereoMatching(void) {

}

void CensusBasedStereoMatching::setMaximumDisparity(int max_disparity) {
    m_MaxDisparity = max_disparity;
    m_CostVolume.resize(m_MaxDisparity);
}

bool CensusBasedStereoMatching::compute(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &disparity_map, cv::Mat &likelihood_map) {
    if ((left_image.cols != right_image.cols) || (left_image.rows != right_image.rows) || (left_image.cols <= m_MaxDisparity)) {
        return false;
    }
    
    const cv::Mat(&m_RawImage)[2] = { left_image, right_image };
    int width = left_image.cols;
    int height = left_image.rows;

    // コストの計算に必要な画像を用意する
    cv::cvtColor(m_RawImage[0], m_MonochromeImage[0], cv::COLOR_BGR2GRAY);
    cv::cvtColor(m_RawImage[1], m_MonochromeImage[1], cv::COLOR_BGR2GRAY);
    
    doCensus7x5Transform(m_MonochromeImage[0], m_CensusImage[0]);
    doCensus7x5Transform(m_MonochromeImage[1], m_CensusImage[1]);

    // コストを計算する
#pragma omp parallel for
    for (int d = 0; d < m_MaxDisparity; d++) {
        calculateCostImage(m_RawImage[0], m_RawImage[1], d, m_CostVolume[d]);
    }

    // 偏差マップを計算する
    calculateDisparity(m_CostVolume, m_MaxDisparity, disparity_map, likelihood_map);

    return true;
}

void CensusBasedStereoMatching::calculateCostImage(const cv::Mat &left_image, const cv::Mat &right_image, int disparity, cv::Mat &cost_image) {
    static const int kGifRadius = 5;
    static const double kGifEps = 1000;
    static const int kTc = 7;
    static const int kTg = 4;
    static const double kAlpha = 0.2;
    
    const cv::Mat(&m_RawImage)[2] = { left_image, right_image };
    int width = left_image.cols;
    int height = left_image.rows;

    // GuidedFilterを作成する
    cv::Mat guide(m_MonochromeImage[0]);
    FastGuidedFilter filter(guide, kGifRadius, kGifEps, 1);

    // コストボリュームを生成する
    cv::Mat cost(height, width, CV_32F);
    for (int y = 0; y < height; y++) {
        float *cost_ptr = cost.ptr<float>(y);
        int x = 0;
        for (; x < disparity; x++) {
            *cost_ptr++ = std::numeric_limits<float>::infinity();
        }
        for (; x < width; x++) {
            const uint8_t *rgb_l_ptr = m_RawImage[0].ptr(y);
            const uint8_t *rgb_r_ptr = m_RawImage[1].ptr(y);
            const int32_t *census_l_ptr = m_CensusImage[0].ptr<int32_t>(y);
            const int32_t *census_r_ptr = m_CensusImage[1].ptr<int32_t>(y);
            int color_diff =
                abs(rgb_l_ptr[3 * x] - rgb_r_ptr[3 * (x - disparity)])
                + abs(rgb_l_ptr[3 * x + 1] - rgb_r_ptr[3 * (x - disparity) + 1])
                + abs(rgb_l_ptr[3 * x + 2] - rgb_r_ptr[3 * (x - disparity) + 2]);
            int grad_diff =
                __popcnt(census_l_ptr[x] ^ census_r_ptr[x - disparity]);
            *cost_ptr++ = static_cast<float>(kAlpha * std::min(kTc * 16, color_diff) + (1 - kAlpha) * std::min(kTg * 16, grad_diff));
        }
    }
    cost_image = filter.filter(cost);
}

void CensusBasedStereoMatching::calculateDisparity(const std::vector<cv::Mat> &cost_volume, int max_disparity, cv::Mat &disparity_map, cv::Mat &likelihood_map) {
    int width = cost_volume[0].cols;
    int height = cost_volume[0].rows;
    disparity_map.create(height, width, CV_32F);
    likelihood_map.create(height, width, CV_32F);
    std::vector<double> cost_list(max_disparity);
    for (int y = 0; y < height; y++) {
        float *disparity_map_ptr = disparity_map.ptr<float>(y);
        float *likelihood_map_ptr = likelihood_map.ptr<float>(y);
        for (int x = 0; x < width; x++) {
            const int min_d = 0;
            const int max_d = std::min(x + 1, max_disparity);
            int min_cost_d = -1;
            double min_cost = std::numeric_limits<double>::max();
            double mean_cost = 0.0;
            for (int d = min_d; d < max_d; d++) {
                double cost = cost_volume[d].at<float>(y, x);
                cost_list[d] = cost;
                mean_cost += cost;
                if (cost < min_cost) {
                    min_cost_d = d;
                    min_cost = cost;
                }
            }
            double subpixel_disp = min_cost_d;
            if ((min_d < min_cost_d) && (min_cost_d < (max_d - 1))) {
                double prev_cost = cost_list[min_cost_d - 1];
                double next_cost = cost_list[min_cost_d + 1];
                /*if (p_next <= p_prev) {
                if (d_min < (max_disparity - 2)) {
                double p_next2 = p_list[d_min + 2];
                d_frac += (p_prev - p_next) / (p_prev - p_min - p_next + p_next2);
                } else {
                d_frac += (p_prev - p_next) / (p_prev - p_min) * 0.5;
                }
                } else {
                if (1 < d_min) {
                double p_prev2 = p_list[d_min - 2];
                d_frac += (p_prev - p_next) / (p_prev2 - p_prev - p_min + p_next);
                } else {
                d_frac += (p_prev - p_next) / (p_next - p_min) * 0.5;
                }
                }*/
                subpixel_disp += 0.5 * (prev_cost - next_cost) / (prev_cost - 2 * min_cost + next_cost);
            }
            disparity_map_ptr[x] = static_cast<float>(std::max(std::min(subpixel_disp, 255.0), 0.0));
            mean_cost /= max_d - min_d;
            likelihood_map_ptr[x] = static_cast<float>((mean_cost - min_cost) / mean_cost);
        }
    }
}


void CensusBasedStereoMatching::doCensus7x5Transform(const cv::Mat &src, cv::Mat &dst) {
    // ウィンドウの形
    //
    //  OOOOO
    // OOOOOOO
    // OOO+OOO
    // OOOOOOO
    //  OOOOO
    //
    int width = src.cols;
    int height = src.rows;
    dst.create(height, width, CV_32SC1);
    for (int y00 = 0; y00 < height; y00++) {
        int y2n = std::max(y00 - 2, 0);
        int y1n = std::max(y00 - 1, 0);
        int y1p = std::min(y00 + 1, height - 1);
        int y2p = std::min(y00 + 2, height - 1);
        for (int x00 = 0; x00 < width; x00++) {
            int x3n = std::max(x00 - 3, 0);
            int x2n = std::max(x00 - 2, 0);
            int x1n = std::max(x00 - 1, 0);
            int x1p = std::min(x00 + 1, width - 1);
            int x2p = std::min(x00 + 2, width - 1);
            int x3p = std::min(x00 + 3, width - 1);
            int32_t bitmap = 0;
            uint8_t center = src.at<uint8_t>(y00, x00);
            bitmap |= (center < src.at<uint8_t>(y2n, x2n)) ? 0x1 : 0;
            bitmap |= (center < src.at<uint8_t>(y2n, x1n)) ? 0x2 : 0;
            bitmap |= (center < src.at<uint8_t>(y2n, x00)) ? 0x4 : 0;
            bitmap |= (center < src.at<uint8_t>(y2n, x1p)) ? 0x8 : 0;
            bitmap |= (center < src.at<uint8_t>(y2n, x2p)) ? 0x10 : 0;
            bitmap |= (center < src.at<uint8_t>(y1n, x3n)) ? 0x20 : 0;
            bitmap |= (center < src.at<uint8_t>(y1n, x2n)) ? 0x40 : 0;
            bitmap |= (center < src.at<uint8_t>(y1n, x1n)) ? 0x80 : 0;
            bitmap |= (center < src.at<uint8_t>(y1n, x00)) ? 0x100 : 0;
            bitmap |= (center < src.at<uint8_t>(y1n, x1p)) ? 0x200 : 0;
            bitmap |= (center < src.at<uint8_t>(y1n, x2p)) ? 0x400 : 0;
            bitmap |= (center < src.at<uint8_t>(y1n, x3p)) ? 0x800 : 0;
            bitmap |= (center < src.at<uint8_t>(y00, x3n)) ? 0x1000 : 0;
            bitmap |= (center < src.at<uint8_t>(y00, x2n)) ? 0x2000 : 0;
            bitmap |= (center < src.at<uint8_t>(y00, x1n)) ? 0x4000 : 0;
            bitmap |= (center < src.at<uint8_t>(y00, x1p)) ? 0x8000 : 0;
            bitmap |= (center < src.at<uint8_t>(y00, x2p)) ? 0x10000 : 0;
            bitmap |= (center < src.at<uint8_t>(y00, x3p)) ? 0x20000 : 0;
            bitmap |= (center < src.at<uint8_t>(y1p, x3n)) ? 0x40000 : 0;
            bitmap |= (center < src.at<uint8_t>(y1p, x2n)) ? 0x80000 : 0;
            bitmap |= (center < src.at<uint8_t>(y1p, x1n)) ? 0x100000 : 0;
            bitmap |= (center < src.at<uint8_t>(y1p, x00)) ? 0x200000 : 0;
            bitmap |= (center < src.at<uint8_t>(y1p, x1p)) ? 0x400000 : 0;
            bitmap |= (center < src.at<uint8_t>(y1p, x2p)) ? 0x800000 : 0;
            bitmap |= (center < src.at<uint8_t>(y1p, x3p)) ? 0x1000000 : 0;
            bitmap |= (center < src.at<uint8_t>(y2p, x2n)) ? 0x2000000 : 0;
            bitmap |= (center < src.at<uint8_t>(y2p, x1n)) ? 0x4000000 : 0;
            bitmap |= (center < src.at<uint8_t>(y2p, x00)) ? 0x8000000 : 0;
            bitmap |= (center < src.at<uint8_t>(y2p, x1p)) ? 0x10000000 : 0;
            bitmap |= (center < src.at<uint8_t>(y2p, x2p)) ? 0x20000000 : 0;
            dst.at<int32_t>(y00, x00) = bitmap;
        }
    }
}
