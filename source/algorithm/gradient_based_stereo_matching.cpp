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

void GradientBasedStereoMatching::precompute(const cv::Mat &left_image, const cv::Mat &right_image) {
    CV_Assert((left_image.cols == right_image.cols) && (left_image.rows == right_image.rows));
    m_GaussianDoG[0]->compute(left_image);
    m_GaussianDoG[1]->compute(right_image);
}

void GradientBasedStereoMatching::compute(cv::Mat &disparity_map, int max_disparity) const {
    int width = m_GaussianDoG[0]->dog0deg().cols;
    int height = m_GaussianDoG[0]->dog0deg().rows;
    
    CV_Assert(max_disparity <= width);
    
    // 偏差を計算する
    disparity_map.create(height, width, CV_32F);
#pragma omp parallel
    {
        std::vector<float> costs(max_disparity);

#pragma omp for
        for (int y = 0; y < height; y++) {
            float *disparity_map_ptr = disparity_map.ptr<float>(y);
            const uint8_t *grad_x_l_ptr = m_GaussianDoG[0]->gradient0deg().ptr(y);
            const uint8_t *grad_x_r_ptr = m_GaussianDoG[1]->gradient0deg().ptr(y);
            const uint8_t *grad_y_l_ptr = m_GaussianDoG[0]->gradient90deg().ptr(y);
            const uint8_t *grad_y_r_ptr = m_GaussianDoG[1]->gradient90deg().ptr(y);
            const uint8_t *dog_x_l_ptr = m_GaussianDoG[0]->dog0deg().ptr(y);
            const uint8_t *dog_x_r_ptr = m_GaussianDoG[1]->dog0deg().ptr(y);
            const uint8_t *dog_y_l_ptr = m_GaussianDoG[0]->dog90deg().ptr(y);
            const uint8_t *dog_y_r_ptr = m_GaussianDoG[1]->dog90deg().ptr(y);
            for (int x = 0; x < width; x++) {
                // 偏差ごとのコストを計算し、最小コストの偏差を記録する(Winner Takes All)
                const int min_d = 0;
                const int max_d = std::min(x + 1, max_disparity);
                int min_cost_d = -1;
                double min_cost = std::numeric_limits<double>::max();
                for (int d = 0; d < max_d; d++) {
                    int grad_diff = abs(grad_x_l_ptr[x] - grad_x_r_ptr[x - d]) + abs(grad_y_l_ptr[x] - grad_y_r_ptr[x - d]);
                    int dog_diff = abs(dog_x_l_ptr[x] - dog_x_r_ptr[x - d]) + abs(dog_y_l_ptr[x] - dog_y_r_ptr[x - d]);
                    double cost = kAlpha * grad_diff + kBeta * dog_diff;
                    costs[d] = static_cast<float>(cost);
                    if (cost < min_cost) {
                        min_cost_d = d;
                        min_cost = cost;
                    }
                }

                // 偏差のサブピクセル推定を行う
                double subpixel_d = min_cost_d;
                if ((min_d < min_cost_d) && (min_cost_d < (max_d - 1))) {
                    double prev_cost = costs[min_cost_d - 1];
                    double next_cost = costs[min_cost_d + 1];
                    subpixel_d += 0.5 * (prev_cost - next_cost) / (prev_cost - 2 * min_cost + next_cost);
                }
                disparity_map_ptr[x] = static_cast<float>(std::max(std::min(subpixel_d, 255.0), 0.0));
            }
        }
    }
}

void GradientBasedStereoMatching::compute(std::vector<cv::Point3f> &points, int max_disparity) const {
    int num_of_points = static_cast<int>(points.size());
    int width = m_GaussianDoG[0]->dog0deg().cols;
    int height = m_GaussianDoG[0]->dog0deg().rows;

    CV_Assert(max_disparity <= width);

#pragma omp parallel
    {
        std::vector<float> costs(max_disparity);

#pragma omp for
        for (int index = 0; index < num_of_points; index++) {
            int x = static_cast<int>(points[index].x);
            int y = static_cast<int>(points[index].y);
            const uint8_t *grad_x_l_ptr = m_GaussianDoG[0]->gradient0deg().ptr(y);
            const uint8_t *grad_x_r_ptr = m_GaussianDoG[1]->gradient0deg().ptr(y);
            const uint8_t *grad_y_l_ptr = m_GaussianDoG[0]->gradient90deg().ptr(y);
            const uint8_t *grad_y_r_ptr = m_GaussianDoG[1]->gradient90deg().ptr(y);
            const uint8_t *dog_x_l_ptr = m_GaussianDoG[0]->dog0deg().ptr(y);
            const uint8_t *dog_x_r_ptr = m_GaussianDoG[1]->dog0deg().ptr(y);
            const uint8_t *dog_y_l_ptr = m_GaussianDoG[0]->dog90deg().ptr(y);
            const uint8_t *dog_y_r_ptr = m_GaussianDoG[1]->dog90deg().ptr(y);

            // 偏差ごとのコストを計算し、最小コストの偏差を記録する(Winner Takes All)
            const int min_d = 0;
            const int max_d = std::min(x + 1, max_disparity);
            int min_cost_d = -1;
            double min_cost = std::numeric_limits<double>::max();
            for (int d = 0; d < max_d; d++) {
                int grad_diff = abs(grad_x_l_ptr[x] - grad_x_r_ptr[x - d]) + abs(grad_y_l_ptr[x] - grad_y_r_ptr[x - d]);
                int dog_diff = abs(dog_x_l_ptr[x] - dog_x_r_ptr[x - d]) + abs(dog_y_l_ptr[x] - dog_y_r_ptr[x - d]);
                double cost = kAlpha * grad_diff + kBeta * dog_diff;
                costs[d] = static_cast<float>(cost);
                if (cost < min_cost) {
                    min_cost_d = d;
                    min_cost = cost;
                }
            }

            // 偏差のサブピクセル推定を行う
            double subpixel_d = min_cost_d;
            if ((min_d < min_cost_d) && (min_cost_d < (max_d - 1))) {
                double prev_cost = costs[min_cost_d - 1];
                double next_cost = costs[min_cost_d + 1];
                subpixel_d += 0.5 * (prev_cost - next_cost) / (prev_cost - 2 * min_cost + next_cost);
            }
            points[index].z = static_cast<float>(std::max(std::min(subpixel_d, 255.0), 0.0));
        }
    }
}
