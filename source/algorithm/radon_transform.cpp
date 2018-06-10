#include "radon_transform.h"
#define _USE_MATH_DEFINES
#include <math.h>

RadonTransform::RadonTransform(int angle_resolution)
    : m_AngleTable(angle_resolution), m_CosTable(angle_resolution), m_SinTable(angle_resolution)
{
    for (int index = 0; index < angle_resolution; index++) {
        m_AngleTable[index] = M_PI * index / angle_resolution;
        m_CosTable[index] = cos(m_AngleTable[index]);
        m_SinTable[index] = sin(m_AngleTable[index]);
    }
}

void RadonTransform::compute(const cv::Mat &src, cv::Mat &dest) {
    const int width = src.cols;
    const int height = src.rows;

    const int max_rho = 2 * std::max(width, height) - 1;
    const int angle_resolution = (int)m_CosTable.size();

    dest.create(max_rho, angle_resolution, CV_32F);

    for (int angle_index = 0; angle_index < angle_resolution; angle_index++) {
        double cos_theta = m_CosTable[angle_index];
        double sin_theta = m_SinTable[angle_index];
        if (abs(sin_theta) < (1.0 / sqrt(2))) {
            for (int rho = 0; rho < max_rho; rho++) {
                int sum = 0;
                for (int y = 0; y < height; y++) {
                    int x = static_cast<int>(round((-y * sin_theta + rho) / cos_theta));
                    if ((0 <= x) && (x < width)) {
                        sum += src.at<uint8_t>(y, x);
                    }
                }
                dest.at<float>(rho, angle_index) = sum;
            }
        } else {
            for (int rho = 0; rho < max_rho; rho++) {
                int sum = 0;
                for (int x = 0; x < width; x++) {
                    int y = static_cast<int>(round((-x * cos_theta + rho) / sin_theta));
                    if ((0 <= y) && (y < height)) {
                        sum += src.at<uint8_t>(y, x);
                    }
                }
                dest.at<float>(rho, angle_index) = sum;
            }
        }
    }
}
