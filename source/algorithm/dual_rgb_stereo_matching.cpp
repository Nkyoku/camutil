#include "dual_rgb_stereo_matching.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>
#include <intrin.h>
#include <limits>

DualRgbStereoMatching::DualRgbStereoMatching(void) {

}

void DualRgbStereoMatching::setMaximumDisparity(int max_disparity) {
    m_MaxDisparity = max_disparity;
    m_CostVolume.resize(m_MaxDisparity);
}

bool DualRgbStereoMatching::compute(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &disparity_map) {
    if ((left_image.cols != right_image.cols) || (left_image.rows != right_image.rows) || (left_image.cols <= m_MaxDisparity)) {
        return false;
    }
    
    const cv::Mat(&m_RawImage)[2] = { left_image, right_image };
    int width = left_image.cols;
    int height = left_image.rows;

    // コストの計算に必要な画像を用意する
    cv::cvtColor(m_RawImage[0], m_MonochromeImage[0], cv::COLOR_BGR2GRAY);
    cv::cvtColor(m_RawImage[1], m_MonochromeImage[1], cv::COLOR_BGR2GRAY);
    
    cv::GaussianBlur(m_RawImage[0], m_FilteredImage[0], cv::Size(5, 5), 1.25, 1.25);
    cv::GaussianBlur(m_RawImage[1], m_FilteredImage[1], cv::Size(5, 5), 1.25, 1.25);
    
    generateDualRgbGradientX(m_MonochromeImage[0], m_FilteredImage[0], m_DualRgbGradientXImage[0]);
    generateDualRgbGradientX(m_MonochromeImage[1], m_FilteredImage[1], m_DualRgbGradientXImage[1]);
    generateDualRgbGradientY(m_MonochromeImage[0], m_FilteredImage[0], m_DualRgbGradientYImage[0]);
    generateDualRgbGradientY(m_MonochromeImage[1], m_FilteredImage[1], m_DualRgbGradientYImage[1]);
    
    doCensus7x5Transform(m_MonochromeImage[0], m_CensusImage[0]);
    doCensus7x5Transform(m_MonochromeImage[1], m_CensusImage[1]);

    // コストを計算する
#pragma omp parallel for
    for (int disparity = 0; disparity < m_MaxDisparity; disparity++) {
        cv::Mat cost_temp;
        calculateCostImage(m_RawImage[0], m_RawImage[1], disparity, cost_temp);
        calculateCostAggregation(m_RawImage[0], cost_temp, 1, m_CostVolume[disparity]);
        calculateCostAggregation(m_RawImage[0], m_CostVolume[disparity], 3, cost_temp);
        calculateCostAggregation(m_RawImage[0], cost_temp, 9, m_CostVolume[disparity]);
    }

    // 偏差マップを計算する
    calculateDisparity(m_CostVolume, m_MaxDisparity, disparity_map);

    return true;
}

void DualRgbStereoMatching::calculateCostImage(const cv::Mat &left_image, const cv::Mat &right_image, int disparity, cv::Mat &cost_image) {
    const cv::Mat(&m_RawImage)[2] = { left_image, right_image };
    int width = left_image.cols;
    int height = left_image.rows;

    double w1 = m_GradientWeight1;
    double w2 = m_GradientWeight2;
    double alpha = m_CostWeightOfRaw / 3.0;
    double beta = m_CostWeightOfCensus;
    double xi = m_CostWeightOfGradientX / 6.0;
    double eps = m_CostWeightOfGradientY / 6.0;

    cost_image.create(height, width, CV_32F);
    for (int y = 0; y < height; y++) {
        float *cost_image_ptr = cost_image.ptr<float>(y);
        int x = 0;
        for (; x < disparity; x++) {
            *cost_image_ptr++ = std::numeric_limits<float>::infinity();
        }
        const uint8_t *raw_l_ptr = m_RawImage[0].ptr(y, disparity);
        const uint8_t *raw_r_ptr = m_RawImage[1].ptr(y, 0);
        const int *census_l_ptr = m_CensusImage[0].ptr<int>(y, disparity);
        const int *census_r_ptr = m_CensusImage[1].ptr<int>(y, 0);
        const int8_t *grad_x_l_ptr = m_DualRgbGradientXImage[0].ptr<int8_t>(y, disparity);
        const int8_t *grad_x_r_ptr = m_DualRgbGradientXImage[1].ptr<int8_t>(y, 0);
        const int8_t *grad_y_l_ptr = m_DualRgbGradientYImage[0].ptr<int8_t>(y, disparity);
        const int8_t *grad_y_r_ptr = m_DualRgbGradientYImage[1].ptr<int8_t>(y, 0);
        for (; x < width; x++) {
            double cost_raw =
                abs(static_cast<int>(raw_l_ptr[0]) - static_cast<int>(raw_r_ptr[0])) +
                abs(static_cast<int>(raw_l_ptr[1]) - static_cast<int>(raw_r_ptr[1])) +
                abs(static_cast<int>(raw_l_ptr[2]) - static_cast<int>(raw_r_ptr[2]));
            double cost_census =
                __popcnt(census_l_ptr[0] ^ census_r_ptr[0]);
            double cost_grad_x =
                w1 * abs(static_cast<int>(grad_x_l_ptr[0]) - static_cast<int>(grad_x_r_ptr[0])) +
                w2 * (abs(static_cast<int>(grad_x_l_ptr[1]) - static_cast<int>(grad_x_r_ptr[1])) +
                abs(static_cast<int>(grad_x_l_ptr[2]) - static_cast<int>(grad_x_r_ptr[2])) +
                abs(static_cast<int>(grad_x_l_ptr[3]) - static_cast<int>(grad_x_r_ptr[3])));
            double cost_grad_y =
                w1 * abs(static_cast<int>(grad_y_l_ptr[0]) - static_cast<int>(grad_y_r_ptr[0])) +
                w2 * (abs(static_cast<int>(grad_y_l_ptr[1]) - static_cast<int>(grad_y_r_ptr[1])) +
                    abs(static_cast<int>(grad_y_l_ptr[2]) - static_cast<int>(grad_y_r_ptr[2])) +
                    abs(static_cast<int>(grad_y_l_ptr[3]) - static_cast<int>(grad_y_r_ptr[3])));
            raw_l_ptr += 3;
            raw_r_ptr += 3;
            census_l_ptr += 1;
            census_r_ptr += 1;
            grad_x_l_ptr += 4;
            grad_x_r_ptr += 4;
            grad_y_l_ptr += 4;
            grad_y_r_ptr += 4;
            *cost_image_ptr++ = alpha * std::min(cost_raw, 18.0) + beta * std::min(cost_census, 21.0) +
                xi * std::min(cost_grad_x, 8.0) + eps * std::min(cost_grad_y, 8.0);
        }
    }
}

void DualRgbStereoMatching::calculateCostAggregation(const cv::Mat &raw_image, cv::Mat &cost_image, int s, cv::Mat &aggregated_cost_image) {
    int width = cost_image.cols;
    int height = cost_image.rows;
    aggregated_cost_image.create(height, width, CV_32F);

    auto deltaColor2 = [](const cv::Vec3b &a, const cv::Vec3b &b) {
        return pow(static_cast<int>(a[0]) - static_cast<int>(b[0]), 2) + pow(static_cast<int>(a[1]) - static_cast<int>(b[1]), 2) + pow(static_cast<int>(a[2]) - static_cast<int>(b[2]), 2);
    };

    double weight_distance = exp(-s / 15.0);

    // X方向にコスト集計を行う
    for (int y = 0; y < height; y++) {
        const float *cost_image_ptr = cost_image.ptr<float>(y);
        const cv::Vec3b *raw_image_ptr = raw_image.ptr<cv::Vec3b>(y);
        float *aggregated_cost_image_ptr = aggregated_cost_image.ptr<float>(y);
        for (int x = 0; x < width; x++) {
            int x_n = std::max(x - s, 0);
            int x_p = std::min(x + s, width - 1);
            double cost_n = cost_image_ptr[x_n];
            double cost_0 = cost_image_ptr[x];
            double cost_p = cost_image_ptr[x_p];
            const cv::Vec3b &color_n = raw_image_ptr[x_n];
            const cv::Vec3b &color_0 = raw_image_ptr[x];
            const cv::Vec3b &color_p = raw_image_ptr[x_p];
            double delta_cn = sqrt(deltaColor2(color_0, color_n));
            double delta_cp = sqrt(deltaColor2(color_0, color_p));
            double weight_n = weight_distance * exp(-delta_cn / 15.0);
            double weight_p = weight_distance * exp(-delta_cp / 15.0);
            aggregated_cost_image_ptr[x] = cost_0 + (weight_n * cost_n + weight_p * cost_p) / kAggregationSize;
        }
    }

    // Y方向にコスト集計を行う
    for (int y = 0; y < height; y++) {
        int y_n = std::max(y - s, 0);
        int y_p = std::min(y + s, height - 1);
        const float *cost_image_n_ptr = cost_image.ptr<float>(y_n);
        const float *cost_image_ptr = cost_image.ptr<float>(y);
        const float *cost_image_p_ptr = cost_image.ptr<float>(y_p);
        const cv::Vec3b *raw_image_n_ptr = raw_image.ptr<cv::Vec3b>(y_n);
        const cv::Vec3b *raw_image_ptr = raw_image.ptr<cv::Vec3b>(y);
        const cv::Vec3b *raw_image_p_ptr = raw_image.ptr<cv::Vec3b>(y_p);
        float *aggregated_cost_image_ptr = aggregated_cost_image.ptr<float>(y);
        for (int x = 0; x < width; x++) {
            double cost_n = cost_image_n_ptr[x];
            double cost_0 = cost_image_ptr[x];
            double cost_p = cost_image_p_ptr[x];
            const cv::Vec3b &color_n = raw_image_n_ptr[x];
            const cv::Vec3b &color_0 = raw_image_ptr[x];
            const cv::Vec3b &color_p = raw_image_p_ptr[x];
            double delta_cn = sqrt(deltaColor2(color_0, color_n));
            double delta_cp = sqrt(deltaColor2(color_0, color_p));
            double weight_n = weight_distance * exp(-delta_cn / 15.0);
            double weight_p = weight_distance * exp(-delta_cp / 15.0);
            aggregated_cost_image_ptr[x] += /*cost_0 + */(weight_n * cost_n + weight_p * cost_p) / kAggregationSize;
        }
    }
}

void DualRgbStereoMatching::calculateDisparity(const std::vector<cv::Mat> &cost_volume, int max_disparity, cv::Mat &disparity_map) {
    int width = cost_volume[0].cols;
    int height = cost_volume[0].rows;
    disparity_map.create(height, width, CV_32F);
    for (int y = 0; y < height; y++) {
        float *disparity_map_ptr = disparity_map.ptr<float>(y);
        for (int x = 0; x < width; x++) {
            const int min_d = 0;
            const int max_d = std::min(x + 1, max_disparity);
            int min_cost_d = -1;
            double min_cost = std::numeric_limits<double>::max();
            for (int d = min_d; d < max_d; d++) {
                double cost = cost_volume[d].at<float>(y, x);
                if (cost < min_cost) {
                    min_cost_d = d;
                    min_cost = cost;
                }
            }
            disparity_map_ptr[x] = static_cast<float>(min_cost_d);
        }
    }
}

void DualRgbStereoMatching::generateDualRgbGradientX(const cv::Mat &src1, const cv::Mat &src2, cv::Mat &dst) {
    int width = src1.cols;
    int height = src1.rows;
    dst.create(height, width, CV_8SC4);
    for (int y = 0; y < height; y++) {
        const uint8_t *src1_ptr = src1.ptr(y);
        const uint8_t *src2_ptr = src2.ptr(y);
        int8_t *dst_ptr = dst.ptr<int8_t>(y);
        
        // x==0のときの特別な処理
        memset(dst_ptr, 0, 4);

        for (int x = 1; x < width; x++) {
            dst_ptr[0] = static_cast<int8_t>(std::max(std::min(static_cast<int>(src1_ptr[1]) - static_cast<int>(src1_ptr[0]), 127), -128));
            dst_ptr[1] = static_cast<int8_t>(std::max(std::min(static_cast<int>(src2_ptr[3]) - static_cast<int>(src2_ptr[0]), 127), -128));
            dst_ptr[2] = static_cast<int8_t>(std::max(std::min(static_cast<int>(src2_ptr[4]) - static_cast<int>(src2_ptr[1]), 127), -128));
            dst_ptr[3] = static_cast<int8_t>(std::max(std::min(static_cast<int>(src2_ptr[5]) - static_cast<int>(src2_ptr[2]), 127), -128));
            src1_ptr += 1;
            src2_ptr += 3;
            dst_ptr += 4;
        }
    }
}

void DualRgbStereoMatching::generateDualRgbGradientY(const cv::Mat &src1, const cv::Mat &src2, cv::Mat &dst) {
    int width = src1.cols;
    int height = src1.rows;

    dst.create(height, width, CV_8SC4);

    // y==0のときの特別な処理
    memset(dst.ptr(), 0, 4 * width);

    int step_size = 3 * width;
    for (int y = 1; y < height; y++) {
        const uint8_t *src1_ptr = src1.ptr(y);
        const uint8_t *src2_ptr = src2.ptr(y);
        int8_t *dst_ptr = dst.ptr<int8_t>(y);

        for (int x = 0; x < width; x++) {
            dst_ptr[0] = static_cast<int8_t>(std::max(std::min(static_cast<int>(src1_ptr[step_size + 0]) - static_cast<int>(src1_ptr[0]), 127), -128));
            dst_ptr[1] = static_cast<int8_t>(std::max(std::min(static_cast<int>(src2_ptr[step_size + 0]) - static_cast<int>(src2_ptr[0]), 127), -128));
            dst_ptr[2] = static_cast<int8_t>(std::max(std::min(static_cast<int>(src2_ptr[step_size + 1]) - static_cast<int>(src2_ptr[1]), 127), -128));
            dst_ptr[3] = static_cast<int8_t>(std::max(std::min(static_cast<int>(src2_ptr[step_size + 2]) - static_cast<int>(src2_ptr[2]), 127), -128));
            src1_ptr += 1;
            src2_ptr += 3;
            dst_ptr += 4;
        }
    }
}

void DualRgbStereoMatching::doCensus7x5Transform(const cv::Mat &src, cv::Mat &dst) {
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
