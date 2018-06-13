#include "geometry_2d.h"
#include <math.h>
#include <stdlib.h>
#include <limits>
#include "geometry_2d_inl.h"

double distanceBetweenSegmentAndPoint(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &p) {
    if ((p - a1).dot(a2 - a1) <= 0) {
        return distance(a1, p);
    } else if ((p - a2).dot(a1 - a2) <= 0) {
        return distance(a2, p);
    } else {
        return abs((a2 - a1).cross(p - a1)) / distance(a1, a2);
    }
}

double distanceBetweenSegments(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2) {
    if (segmentIntersectCheck(a1, a2, b1, b2) == true) {
        return 0.0;
    }
    double a_b1 = distanceBetweenSegmentAndPoint(a1, a2, b1);
    double a_b2 = distanceBetweenSegmentAndPoint(a1, a2, b2);
    double b_a1 = distanceBetweenSegmentAndPoint(b1, b2, a1);
    double b_a2 = distanceBetweenSegmentAndPoint(b1, b2, a2);
    return std::min(std::min(a_b1, a_b2), std::min(b_a1, b_a2));
}

double distanceBetweenSegmentsSimple(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2) {
    cv::Point2d c((a1 + a2) * 0.5);
    cv::Point2d d((b1 + b2) * 0.5);
    return distance(c, d) - (distance(a1, a2) + distance(b1, b2)) * 0.5;
}

double distanceBetweenLineAndPoint(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &p) {
    return abs((a2 - a1).cross(p - a1)) / distance(a1, a2);
}

double distanceBetweenLineAndSegment(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2) {
    cv::Point2d vector_a(a2 - a1);
    double rlength_a = 1.0 / distance(a1, a2);
    double distance_a_b1 = vector_a.cross(b1 - a1);
    double distance_a_b2 = vector_a.cross(b2 - a1);
    if ((distance_a_b1 * distance_a_b2) <= 0.0) {
        return 0.0;
    }
    distance_a_b1 = abs(distance_a_b1) * rlength_a;
    distance_a_b2 = abs(distance_a_b2) * rlength_a;
    return std::min(distance_a_b1, distance_a_b2);
}

bool segmentIntersectCheck(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2) {
    return (((a2 - a1).cross(b1 - a1) * (a2 - a1).cross(b2 - a1)) < 0)
        && (((b2 - b1).cross(a1 - b1) * (b2 - b1).cross(a2 - b1)) < 0);
}

cv::Point2d segmentIntersection(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2, bool *a_intersect, bool *b_intersect) {
    cv::Point2d result = lineIntersection(a1, a2, b1, b2);
    if (a_intersect != nullptr) {
        *a_intersect = isPointInRectangle(result, a1, a2);
    }
    if (b_intersect != nullptr) {
        *b_intersect = isPointInRectangle(result, b1, b2);
    }
    return result;
}

cv::Point2d lineIntersection(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2) {
    cv::Point2d vec_a = a2 - a1;
    cv::Point2d vec_b = b2 - b1;
    double cross_of_2_vectors = vec_b.cross(vec_a);
    if (cross_of_2_vectors != 0.0) {
        return a1 + vec_a * vec_b.cross(b1 - a1) / cross_of_2_vectors;
    } else {
        return cv::Point2d(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    }
}

bool isPointOnLeftSideOfLine(const cv::Point2d &p, const cv::Point2d &a1, const cv::Point2d &a2) {
    return 0.0 <= (a2 - a1).cross(p - a1);
}


void combine2Segments(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2, cv::Point2d &c1, cv::Point2d &c2) {
    double weight_a = distance(a1, a2);
    double weight_b = distance(b1, b2);
    cv::Point2d center_p((weight_a * (a1 + a2) + weight_b * (b1 + b2)) * 0.5 / (weight_a + weight_b));
    cv::Point2d vector_p(a2 - a1 + b2 - b1);
    vector_p *= 1.0 / length(vector_p);
    double t1 = vector_p.dot(a1 - center_p);
    double t2 = vector_p.dot(a2 - center_p);
    double t3 = vector_p.dot(b1 - center_p);
    double t4 = vector_p.dot(b2 - center_p);
    double min_t = std::min(std::min(t1, t2), std::min(t3, t4));
    double max_t = std::max(std::max(t1, t2), std::max(t3, t4));
    c1 = center_p + min_t * vector_p;
    c2 = center_p + max_t * vector_p;
}

void reduceSegments(const std::vector<cv::Vec4f> &input, std::vector<cv::Vec4f> &output, double pos_threshold, double dir_threshold) {
    output.clear();

    std::vector<bool> processed(input.size(), false);
    for (int i = 0; i < static_cast<int>(input.size()); i++) {
        if (processed[i] == true) {
            continue;
        }
        processed[i] = true;

        cv::Point2d a1(input[i][0], input[i][1]);
        cv::Point2d a2(input[i][2], input[i][3]);
        double length_a = distance(a1, a2);

        // 傾き・座標の近い線分を探す
        for (int j = i + 1; j < static_cast<int>(input.size()); j++) {
            if (processed[j] == true) {
                continue;
            }
            cv::Point2d b1(input[j][0], input[j][1]);
            cv::Point2d b2(input[j][2], input[j][3]);
            double length_b = distance(b1, b2);

            // 傾きが近いかチェックする
            double cos_angle = (a2 - a1).dot(b2 - b1) / (length_a * length_b);
            if (dir_threshold < abs(cos_angle)) {
                // 距離が近いかチェックする
                if (distanceBetweenSegmentsSimple(a1, a2, b1, b2) < pos_threshold) {
                    if (distanceBetweenSegments(a1, a2, b1, b2) < pos_threshold) {
                        if (cos_angle < 0) {
                            std::swap(b1, b2);
                        }
                        combine2Segments(a1, a2, b1, b2, a1, a2);
                        length_a = distance(a1, a2);
                        processed[j] = true;
                    }
                }
            }
        }

        cv::Vec4f result;
        result[0] = static_cast<float>(a1.x);
        result[1] = static_cast<float>(a1.y);
        result[2] = static_cast<float>(a2.x);
        result[3] = static_cast<float>(a2.y);
        output.push_back(result);
    }
}
