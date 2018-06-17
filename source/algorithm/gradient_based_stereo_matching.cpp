#include "gradient_based_stereo_matching.h"
#include "3rdparty/fastguidedfilter.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>
#include <new>
#include <limits>

GradientBasedStereoMatching::GradientBasedStereoMatching(void)
{
    m_GaussianDoG[0] = new GaussianDoG(kFilterKernelSize, kFilterVariance);
    m_GaussianDoG[1] = new GaussianDoG(kFilterKernelSize, kFilterVariance);
}

GradientBasedStereoMatching::~GradientBasedStereoMatching() {
    delete m_GaussianDoG[0];
    delete m_GaussianDoG[1];
}

void GradientBasedStereoMatching::setMaximumDisparity(int max_disparity) {
    m_MaxDisparity = max_disparity;
    m_CostVolume.resize(m_MaxDisparity);
}

bool GradientBasedStereoMatching::compute(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &disparity_map, cv::Mat &likelihood_map) {
    CV_Assert((left_image.cols == right_image.cols) && (left_image.rows == right_image.rows) && (m_MaxDisparity < left_image.cols));

    int width = left_image.cols;
    int height = left_image.rows;

    cv::cvtColor(left_image, m_MonochromeImage[0], cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_image, m_MonochromeImage[1], cv::COLOR_BGR2GRAY);
    m_GaussianDoG[0]->compute(m_MonochromeImage[0]);
    m_GaussianDoG[1]->compute(m_MonochromeImage[1]);

    // コストを計算する
#pragma omp parallel for
    for (int d = 0; d < m_MaxDisparity; d++) {
        calculateCostImage(left_image, right_image, d, m_CostVolume[d]);
    }

    // 偏差マップを計算する
    calculateDisparity(disparity_map, likelihood_map);

    return true;
}

void GradientBasedStereoMatching::calculateCostImage(const cv::Mat &left_image, const cv::Mat &right_image, int disparity, cv::Mat &cost_image) {
    static const int kGifRadius = 5;
    static const double kGifEps = 1000;
    static const int kTc = 7;
    static const int kTg = 4;
    static const double kAlpha = 0.2;
    static const double kBeta = 0.4;
    static const double kGamma = 0.4;
    
    int width = left_image.cols;
    int height = left_image.rows;

    // GuidedFilterを作成する
    //FastGuidedFilter filter(m_MonochromeImage[0], kGifRadius, kGifEps, 1);

    // コストボリュームを生成する
    cv::Mat result(height, width, CV_32F);
    for (int y = 0; y < height; y++) {
        float *result_ptr = result.ptr<float>(y);
        int x = 0;
        for (; x < disparity; x++) {
            *result_ptr++ = std::numeric_limits<float>::infinity();
        }
        for (; x < width; x++) {
            const uint8_t *rgb_l_ptr = left_image.ptr(y);
            const uint8_t *rgb_r_ptr = right_image.ptr(y);
            const uint8_t *grad_x_l_ptr = m_GaussianDoG[0]->gradient0deg().ptr(y);
            const uint8_t *grad_x_r_ptr = m_GaussianDoG[1]->gradient0deg().ptr(y);
            const uint8_t *grad_y_l_ptr = m_GaussianDoG[0]->gradient90deg().ptr(y);
            const uint8_t *grad_y_r_ptr = m_GaussianDoG[1]->gradient90deg().ptr(y);
            const uint8_t *dog_x_l_ptr = m_GaussianDoG[0]->dog0deg().ptr(y);
            const uint8_t *dog_x_r_ptr = m_GaussianDoG[1]->dog0deg().ptr(y);
            const uint8_t *dog_y_l_ptr = m_GaussianDoG[0]->dog90deg().ptr(y);
            const uint8_t *dog_y_r_ptr = m_GaussianDoG[1]->dog90deg().ptr(y);
            int color_diff
                = abs(rgb_l_ptr[3 * x] - rgb_r_ptr[3 * (x - disparity)])
                + abs(rgb_l_ptr[3 * x + 1] - rgb_r_ptr[3 * (x - disparity) + 1])
                + abs(rgb_l_ptr[3 * x + 2] - rgb_r_ptr[3 * (x - disparity) + 2]);
            int grad_diff
                = abs(grad_x_l_ptr[x] - grad_x_r_ptr[x - disparity])
                + abs(grad_y_l_ptr[x] - grad_y_r_ptr[x - disparity]);
            int dog_diff
                = abs(dog_x_l_ptr[x] - dog_x_r_ptr[x - disparity])
                + abs(dog_y_l_ptr[x] - dog_y_r_ptr[x - disparity]);
            *result_ptr++ = static_cast<float>(kAlpha * color_diff + kBeta * grad_diff + kGamma * dog_diff);
        }
    }
    cost_image = result;// filter.filter(result);
}

void GradientBasedStereoMatching::calculateDisparity(cv::Mat &disparity_map, cv::Mat &likelihood_map) {
    int width = m_CostVolume[0].cols;
    int height = m_CostVolume[0].rows;
    disparity_map.create(height, width, CV_32F);
    likelihood_map.create(height, width, CV_32F);
    std::vector<double> cost_list(m_MaxDisparity);
    for (int y = 0; y < height; y++) {
        float *disparity_map_ptr = disparity_map.ptr<float>(y);
        float *likelihood_map_ptr = likelihood_map.ptr<float>(y);
        for (int x = 0; x < width; x++) {


            int likelihood
                = abs((int)m_GaussianDoG[0]->gradient0deg().at<uint8_t>(y, x) - 128)
                + abs((int)m_GaussianDoG[0]->gradient90deg().at<uint8_t>(y, x) - 128)
                + abs((int)m_GaussianDoG[0]->dog0deg().at<uint8_t>(y, x) - 128)
                + abs((int)m_GaussianDoG[0]->dog90deg().at<uint8_t>(y, x) - 128);
            if (likelihood < 4) {
                disparity_map_ptr[x] = 0.0f;
                likelihood_map_ptr[x] = likelihood;
                continue;
            }



            const int min_d = 0;
            const int max_d = std::min(x + 1, m_MaxDisparity);
            int min_cost_d = -1;
            double min_cost = std::numeric_limits<double>::max();
            double mean_cost = 0.0;
            for (int d = min_d; d < max_d; d++) {
                double cost = m_CostVolume[d].at<float>(y, x);
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
