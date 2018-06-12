#include "area_interpolation.h"
#include <opencv2/imgproc.hpp>
//#include <omp.h>

void AreaInterpolation::compute(const cv::Mat &guide, const cv::Mat &src, cv::Mat &dst) {
    int width = src.cols;
    int height = src.rows;

    cv::Mat m_CannyImage;
    cv::Canny(guide, m_CannyImage, 100, 200);

    dst.create(height, width, CV_32F);
    dst.setTo(0.0f);

    // X方向に補間する
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        int start_x = -1;
        double start_value = 0.0;
        for (int x = 0; x < width; x++) {
            if (m_CannyImage.at<uint8_t>(y, x) != 0) {
                double value = src.at<float>(y, x);
                dst.at<float>(y, x) = value;
                if (start_x == -1) {
                    start_value = value;
                } else {
                    int end_x = x;
                    double end_value = value;
                    if (1 < (end_x - start_x)) {
                        double alpha = (end_value - start_value) / (end_x - start_x);
                        for (int x = start_x + 1; x < end_x; x++) {
                            dst.at<float>(y, x) = start_value + alpha * (x - start_x);
                        }
                    }
                }
                start_x = x;
                start_value = value;
            }
        }
    }

    // Y方向に補間する
#pragma omp parallel for
    for (int x = 0; x < width; x++) {
        int start_y = -1;
        double start_value = 0.0;
        for (int y = 0; y < height; y++) {
            double value = dst.at<float>(y, x);
            if (value != 0.0f) {
                if (start_y == -1) {
                    start_value = value;
                } else {
                    int end_y = y;
                    double end_value = value;
                    if (1 < (end_y - start_y)) {
                        double alpha = (end_value - start_value) / (end_y - start_y);
                        for (int y = start_y + 1; y < end_y; y++) {
                            dst.at<float>(y, x) = start_value + alpha * (y - start_y);
                        }
                    }
                }
                start_y = y;
                start_value = value;
            }
        }
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (m_CannyImage.at<uint8_t>(y, x) != 0) {
                dst.at<float>(y, x) = 255.0f;
            }
        }
    }
}
