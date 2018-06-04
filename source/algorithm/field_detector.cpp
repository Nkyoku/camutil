#include "field_detector.h"
#define _USE_MATH_DEFINES
#include <math.h>

const FieldDetector::LabRegion FieldDetector::kDefaultGrassRegion = {
    0, 255,
    20, 180,
    M_PI * 0.6667, -M_PI * 0.8333
};

const FieldDetector::LabRegion FieldDetector::kDefaultWhiteRegion = {
    180, 255,
    0, 20,
    -M_PI, M_PI
};

FieldDetector::FieldDetector(void){
    m_Lsd = cv::createLineSegmentDetector();
}

cv::Mat& FieldDetector::detectGrass(const cv::Mat &lab_image, std::vector<cv::Point2d> &rectangle, double margin) {
    // 芝の色を検知して2値化する
    m_BinaryGrass.create(lab_image.rows, lab_image.cols, CV_8UC1);
    lab_image.forEach<cv::Vec3b>([&](const cv::Vec3b &value, const int pos[2]) {
        m_BinaryGrass.at<uint8_t>(pos[0], pos[1]) = isInsideLab(value[0], value[1], value[2], kDefaultGrassRegion) ? 255 : 0;
    });

    // 芝の領域のモーメントを求め、領域を矩形で近似する
    cv::Moments moments = cv::moments(m_BinaryGrass, true);
    rectangle.resize(4);
    if (moments.m00 == 0) {
        rectangle[0].x = 0.0;
        rectangle[0].y = 0.0;
        rectangle[1].x = 0.0;
        rectangle[1].y = 0.0;
        rectangle[2].x = 0.0;
        rectangle[2].y = 0.0;
        rectangle[3].x = 0.0;
        rectangle[3].y = 0.0;
    } else {
        double center_x = moments.m10 / moments.m00;
        double center_y = moments.m01 / moments.m00;
        double mu20 = moments.m20 / moments.m00 - center_x * center_x;
        double mu02 = moments.m02 / moments.m00 - center_y * center_y;
        double mu11 = moments.m11 / moments.m00 - center_x * center_y;
        double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        double half_length1 = sqrt(1.5 * (mu20 + mu02 + sqrt(4 * mu11 * mu11 + pow(mu20 - mu02, 2)))) * margin;
        double half_length2 = sqrt(1.5 * (mu20 + mu02 - sqrt(4 * mu11 * mu11 + pow(mu20 - mu02, 2)))) * margin;
        rectangle[0].x = center_x - half_length1 * cos_theta - half_length2 * sin_theta;
        rectangle[0].y = center_y - half_length1 * sin_theta + half_length2 * cos_theta;
        rectangle[1].x = center_x + half_length1 * cos_theta - half_length2 * sin_theta;
        rectangle[1].y = center_y + half_length1 * sin_theta + half_length2 * cos_theta;
        rectangle[2].x = center_x + half_length1 * cos_theta + half_length2 * sin_theta;
        rectangle[2].y = center_y + half_length1 * sin_theta - half_length2 * cos_theta;
        rectangle[3].x = center_x - half_length1 * cos_theta + half_length2 * sin_theta;
        rectangle[3].y = center_y - half_length1 * sin_theta - half_length2 * cos_theta;
    }

    return m_BinaryGrass;
}

cv::Mat& FieldDetector::detectLines(const cv::Mat &lab_image, const std::vector<cv::Point2d> &rectangle, std::vector<cv::Vec4f> &line_segments) {
    // rectangleが正しく与えられたときは芝の領域に含まれる白色の成分のみを抽出する
    // そうでないときはすべての領域を対象にする
    std::vector<cv::Range> ranges(lab_image.rows);
    if (rectangle.size() == 4) {
        // 水平スキャンラインと芝の領域の交点を計算し、始点・終点をテーブルに格納する
        for (int y = 0; y < lab_image.rows; y++) {
            int start_x = lab_image.cols, end_x = 0;
            for (int edge = 0; edge < 4; edge++) {
                double x1 = rectangle[edge].x;
                double y1 = rectangle[edge].y;
                double x2 = rectangle[(edge + 1) % 4].x;
                double y2 = rectangle[(edge + 1) % 4].y;
                if (y2 < y1) {
                    std::swap(x1, x2);
                    std::swap(y1, y2);
                }
                int y1_int = static_cast<int>(ceil(y1));
                int y2_int = static_cast<int>(floor(y2));
                if (((y1_int <= y) && (y <= y2_int)) && (y1_int != y2_int)) {
                    double dx = (x2 - x1) / (y2 - y1);
                    double sx = x1 - dx * y1;
                    double x = sx + dx * y;
                    start_x = std::min(start_x, static_cast<int>(floor(x)));
                    end_x = std::max(end_x, static_cast<int>(ceil(x)) + 1);
                }
            }
            ranges[y] = cv::Range(start_x, end_x);
        }
    } else {
        for (auto &range : ranges) {
            range.start = 0;
            range.end = lab_image.cols;
        }
    }
    
    // 白線の色を検知して2値化する
    m_BinaryLines.create(lab_image.rows, lab_image.cols, CV_8UC1);
    lab_image.forEach<cv::Vec3b>([&](const cv::Vec3b &value, const int pos[2]) {
        const cv::Range &range = ranges[pos[0]];
        bool inside_grass = (range.start <= pos[1]) && (pos[1] < range.end);
        m_BinaryLines.at<uint8_t>(pos[0], pos[1]) = (inside_grass && isInsideLab(value[0], value[1], value[2], kDefaultWhiteRegion)) ? 255 : 0;
    });
    
    // 線分を抽出する
    m_Lsd->detect(m_BinaryLines, line_segments);

    return m_BinaryLines;
}

bool FieldDetector::isInsideLab(int L, int a, int b, const LabRegion &region) {
    a -= 128;
    b -= 128;
    if ((region.L_min <= L) && (L <= region.L_max)) {
        int chroma_squared = a * a + b * b;
        int chroma_min_squared = region.chroma_min * region.chroma_min;
        int chroma_max_squared = region.chroma_max * region.chroma_max;
        if ((chroma_min_squared <= chroma_squared) && (chroma_squared <= chroma_max_squared)) {
            double hue = atan2(b, a);
            if (region.hue_min < region.hue_max) {
                if ((region.hue_min <= hue) && (hue <= region.hue_max)) {
                    return true;
                }
            } else {
                if ((hue <= region.hue_max) || (region.hue_min <= hue)) {
                    return true;
                }
            }
        }
    }
    return false;
}

int FieldDetector::isInsideLab(int L, int a, int b, const std::vector<LabRegion> &region_list) {
    a -= 128;
    b -= 128;
    int result = 0;
    int bitmask = 1;
    int chroma_squared = a * a + b * b;
    double hue = atan2(b, a);
    for (const auto &region : region_list) {
        if ((region.L_min <= L) && (L <= region.L_max)) {
            int chroma_min_squared = region.chroma_min * region.chroma_min;
            int chroma_max_squared = region.chroma_max * region.chroma_max;
            if ((chroma_min_squared <= chroma_squared) && (chroma_squared <= chroma_max_squared)) {
                if (region.hue_min < region.hue_max) {
                    if ((region.hue_min <= hue) && (hue <= region.hue_max)) {
                        result |= bitmask;
                    }
                } else {
                    if ((hue <= region.hue_max) || (region.hue_min <= hue)) {
                        result |= bitmask;
                    }
                }
            }
        }
        bitmask <<= 1;
    }
    return result;
}
