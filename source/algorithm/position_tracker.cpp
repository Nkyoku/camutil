#include "position_tracker.h"
#include "geometry_2d.h"
#include <opencv2/imgproc.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <sstream>
#include <iomanip>

const std::vector<cv::Vec4f> PositionTracker::kLineTemplate = {
    // 外側の矩形
    cv::Vec4f(-4.5 + kThickness1 / 2, -3.0 + kThickness1 / 2, +4.5 - kThickness1 / 2, -3.0 + kThickness1 / 2),
    cv::Vec4f(+4.5 - kThickness1 / 2, -3.0 + kThickness1 / 2, +4.5 - kThickness1 / 2, +3.0 - kThickness1 / 2),
    cv::Vec4f(+4.5 - kThickness1 / 2, +3.0 - kThickness1 / 2, -4.5 + kThickness1 / 2, +3.0 - kThickness1 / 2),
    cv::Vec4f(-4.5 + kThickness1 / 2, +3.0 - kThickness1 / 2, -4.5 + kThickness1 / 2, -3.0 + kThickness1 / 2),

    // Aサイドゴールエリア
    cv::Vec4f(-4.5 + kThickness1 / 2, -2.5 + kThickness1 / 2, -3.5 - kThickness1 / 2, -2.5 + kThickness1 / 2),
    cv::Vec4f(-3.5 - kThickness1 / 2, -2.5 + kThickness1 / 2, -3.5 - kThickness1 / 2, +2.5 - kThickness1 / 2),
    cv::Vec4f(-3.5 - kThickness1 / 2, +2.5 - kThickness1 / 2, -4.5 + kThickness1 / 2, +2.5 - kThickness1 / 2),

    // Bサイドゴールエリア
    cv::Vec4f(+4.5 - kThickness1 / 2, -2.5 + kThickness1 / 2, +3.5 + kThickness1 / 2, -2.5 + kThickness1 / 2),
    cv::Vec4f(+3.5 + kThickness1 / 2, -2.5 + kThickness1 / 2, +3.5 + kThickness1 / 2, +2.5 - kThickness1 / 2),
    cv::Vec4f(+3.5 + kThickness1 / 2, +2.5 - kThickness1 / 2, +4.5 - kThickness1 / 2, +2.5 - kThickness1 / 2),

    // センターライン
    cv::Vec4f(0.0, -3.0 + kThickness1 / 2, 0.0, +3.0 - kThickness1 / 2),

    // センターサークル(8角形で近似)
    cv::Vec4f(0.0, -1.0823922, +0.76536686, -0.76536686) * (0.75 - kThickness1 / 2),
    cv::Vec4f(+0.76536686, -0.76536686, +1.0823922, 0.0) * (0.75 - kThickness1 / 2),
    cv::Vec4f(+1.0823922, 0.0, +0.76536686, +0.76536686) * (0.75 - kThickness1 / 2),
    cv::Vec4f(+0.76536686, +0.76536686, 0.0, +1.0823922) * (0.75 - kThickness1 / 2),
    cv::Vec4f(0.0, +1.0823922, -0.76536686, +0.76536686) * (0.75 - kThickness1 / 2),
    cv::Vec4f(-0.76536686, +0.76536686, -1.0823922, 0.0) * (0.75 - kThickness1 / 2),
    cv::Vec4f(-1.0823922, 0.0, -0.76536686, -0.76536686) * (0.75 - kThickness1 / 2),
    cv::Vec4f(-0.76536686, -0.76536686, 0.0, -1.0823922) * (0.75 - kThickness1 / 2),
};




PositionTracker::PositionTracker(void) {
    


}

void PositionTracker::estimate(const std::vector<cv::Vec4f> &left_lines, const std::vector<cv::Vec4f> &right_lines, const Undistort &undistort, cv::Mat &debug, int width, int height, int vanishing_y) {
    debug.create(height, width, CV_8UC3);
    debug.setTo(0);
    for (const cv::Vec4f &segment : left_lines) {
        cv::Point2d a(segment[0], segment[1]);
        cv::Point2d b(segment[2], segment[3]);
        cv::line(debug, a, b, cv::Scalar(255, 0, 0), 1);
    }
    for (const cv::Vec4f &segment : right_lines) {
        cv::Point2d a(segment[0], segment[1]);
        cv::Point2d b(segment[2], segment[3]);
        cv::line(debug, a, b, cv::Scalar(0, 128, 255), 1);
    }

    double diagonal = sqrt(width * width + height * height);
    double horizontal_threshold = kSameSegmentHorizontalDistance * width;
    double distance_threshold = kSameSegmentDistance * diagonal;
    double same_length_threshold = kSameLengthForImage * diagonal;

    std::vector<cv::Vec3f> disparity_points;

    for (const cv::Vec4f &left : left_lines) {
        cv::Point2d a1(left[0], left[1]);
        cv::Point2d a2(left[2], left[3]);
        if (a1.y < a2.y) {
            std::swap(a1, a2);
        }
        cv::Point2d vector_a(a2 - a1);
        double length_a = normalizeAndLength(vector_a);
        for (const cv::Vec4f &right : right_lines) {
            cv::Point2d b1(right[0], right[1]);
            cv::Point2d b2(right[2], right[3]);
            cv::Point2d vector_b(b2 - b1);
            double length_b = normalizeAndLength(vector_b);

            // 成す角cosθが閾値以上か確認する
            double cos_angle = vector_a.dot(vector_b);
            if (abs(cos_angle) < kSameSegmentAngle1) {
                continue;
            }
            if (cos_angle < 0.0) {
                cos_angle = -cos_angle;
                std::swap(b1, b2);
                vector_b *= -1.0;
            }

            // 垂直に近い線分の視差を計算する
            // 線分同士の水平距離を求める
            if (a1.y == a2.y) {
                continue;
            }
            double alpha = (a2.x - a1.x) / (a2.y - a1.y);
            double disparity_b1 = alpha * (b1.y - a1.y) + a1.x - b1.x;
            double disparity_b2 = alpha * (b2.y - a1.y) + a1.x - b2.x;
            double likelihood = 1.0 - exp(-10.0 * abs(vector_a.y));

            // 右目で見た線分が左目で見た線分の左側にあり、十分に近いか確認する
            if ((disparity_b1 < 0.0) || (disparity_b2 < 0.0) || (horizontal_threshold < ((disparity_b1 + disparity_b2) * 0.5))) {
                continue;
            }
            
            cv::line(debug, cv::Point(b1.x, b1.y), cv::Point(b1.x + disparity_b1, b1.y), cv::Scalar(255 * likelihood, 255, 0));
            cv::line(debug, cv::Point(b2.x, b2.y), cv::Point(b2.x + disparity_b2, b2.y), cv::Scalar(255 * likelihood, 255, 0));

            disparity_points.push_back(cv::Vec3f(b1.x + disparity_b1, b1.y, disparity_b1));
            disparity_points.push_back(cv::Vec3f(b2.x + disparity_b2, b2.y, disparity_b2));
        }
    }

    int size[1] = { static_cast<int>(disparity_points.size()) };
    cv::Mat disparity_points_mat(1, size, CV_32FC3);
    for (int index = 0; index < static_cast<int>(disparity_points.size()); index++) {
        disparity_points_mat.at<cv::Vec3f>(index) = disparity_points[index];
    }
    cv::Mat depth_points_mat;

    undistort.reprojectPointsTo3D(disparity_points_mat, depth_points_mat);

    for (int index = 0; index < static_cast<int>(disparity_points.size()); index++) {
        double x = disparity_points[index][0];
        double y = disparity_points[index][1];
        double z = depth_points_mat.at<cv::Vec3f>(index)[2];
        std::ostringstream ss;
        ss << std::setprecision(3) << z;

        int baseline;
        cv::Size size = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::putText(debug, ss.str(), cv::Point(x - size.width / 2, y - size.height - 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    }
}

