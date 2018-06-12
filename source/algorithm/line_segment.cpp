#include "line_segment.h"
#include <math.h>
#include <stdlib.h>
#include <limits>

// 2次元の内積
static inline double dot(double ax, double ay, double bx, double by) {
	return ax * bx + ay * by;
}

// 2次元の外積
static inline double cross(double ax, double ay, double bx, double by) {
	return ax * by - ay * bx;
}

// 点が矩形の中にあるか判定する
// px, py : 点の座標
// ax, ay : 矩形のある角Aの座標
// bx, by : Aとは対角にある角Bの座標
static inline bool isPointInRectangle(double px, double py, double ax, double ay, double bx, double by) {
    if (bx < ax) {
        std::swap(ax, bx);
    }
    if (by < ay) {
        std::swap(ay, by);
    }
    return (ax <= px) && (px <= bx) && (ay <= py) && (py <= by);
}

// 点が矩形の中にあるか判定する
// p : 点の座標
// a : 矩形のある角Aの座標
// b : Aとは対角にある角Bの座標
static inline bool isPointInRectangle(const cv::Point2d &p, const cv::Point2d &a, const cv::Point2d &b) {
    return isPointInRectangle(p.x, p.y, a.x, a.y, b.x, b.y);
}

bool segmentIntersectCheck(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy) {
    return (cross(bx - ax, by - ay, cx - ax, cy - ay) * cross(bx - ax, by - ay, dx - ax, dy - ay) < 0)
        && (cross(dx - cx, dy - cy, ax - cx, ay - cy) * cross(dx - cx, dy - cy, bx - cx, by - cy) < 0);
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

double distanceBetweenSegmentAndPoint(double ax, double ay, double bx, double by, double px, double py){
	if (dot(px - ax, py - ay, bx - ax, by - ay) <= 0){
		return distance(ax, ay, px, py);
	}
	else if (dot(px - bx, py - by, ax - bx, ay - by) <= 0){
		return distance(bx, by, px, py);
	}
	else{
		return abs(cross(bx - ax, by - ay, px - ax, py - ay)) / distance(ax, ay, bx, by);
	}
}

double distanceBetweenSegments(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy){
	if (segmentIntersectCheck(ax, ay, bx, by, cx, cy, dx, dy) == true){
		return 0.0;
	}
	double ab_c = distanceBetweenSegmentAndPoint(ax, ay, bx, by, cx, cy);
	double ab_d = distanceBetweenSegmentAndPoint(ax, ay, bx, by, dx, dy);
	double cd_a = distanceBetweenSegmentAndPoint(cx, cy, dx, dy, ax, ay);
	double cd_b = distanceBetweenSegmentAndPoint(cx, cy, dx, dy, bx, by);
	return std::min(std::min(ab_c, ab_d), std::min(cd_a, cd_b));
}

double distanceBetweenSegmentsSimple(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy){
	double ex = (ax + bx) * 0.5;
	double ey = (ay + by) * 0.5;
	double fx = (cx + dx) * 0.5;
	double fy = (cy + dy) * 0.5;
	return distance(ex, ey, fx, fy) - (distance(ax, ay, bx, by) + distance(cx, cy, dx, dy)) * 0.5;
}

bool isPointOnLeftSideOfLine(const cv::Point2d &p, const cv::Point2d &a1, const cv::Point2d &a2) {
    return 0.0 <= (a2 - a1).cross(p - a1);
}

void reduceSegments(const std::vector<cv::Vec4f> &input, std::vector<cv::Vec4f> &output, double pos_threshold, double dir_threshold) {
    output.clear();

    std::vector<bool> processed(input.size(), false);
    for (int i = 0; i < static_cast<int>(input.size()); i++) {
        if (processed[i] == true) {
            continue;
        }
        processed[i] = true;

        double ax1 = input[i][0];
        double ay1 = input[i][1];
        double bx1 = input[i][2];
        double by1 = input[i][3];
        double w1 = distance(ax1, ay1, bx1, by1);

        // 傾き・座標の近い線分を探す
        for (int j = i + 1; j < static_cast<int>(input.size()); j++) {
            if (processed[j] == true) {
                continue;
            }
            double ax2 = input[j][0];
            double ay2 = input[j][1];
            double bx2 = input[j][2];
            double by2 = input[j][3];
            double w2 = distance(ax2, ay2, bx2, by2);

            // 傾きが近いかチェックする
            double cos_angle = dot(bx1 - ax1, by1 - ay1, bx2 - ax2, by2 - ay2) / (w1 * w2);
            if (dir_threshold < abs(cos_angle)) {
                // 距離が近いかチェックする
                if (distanceBetweenSegmentsSimple(ax1, ay1, bx1, by1, ax2, ay2, bx2, by2) < pos_threshold) {
                    if (distanceBetweenSegments(ax1, ay1, bx1, by1, ax2, ay2, bx2, by2) < pos_threshold) {
                        if (cos_angle < 0) {
                            double tx = ax2;
                            double ty = ay2;
                            ax2 = bx2;
                            ay2 = by2;
                            bx2 = tx;
                            by2 = ty;
                        }

                        double px = (w1 * (ax1 + bx1) + w2 * (ax2 + bx2)) * 0.5 / (w1 + w2);
                        double py = (w1 * (ay1 + by1) + w2 * (ay2 + by2)) * 0.5 / (w1 + w2);
                        double vx = (bx1 - ax1) + (bx2 - ax2);
                        double vy = (by1 - ay1) + (by2 - ay2);
                        double rlen = 1.0 / sqrt(vx * vx + vy * vy);
                        vx *= rlen;
                        vy *= rlen;

                        double t1 = dot(vx, vy, ax1 - px, ay1 - py);
                        double t2 = dot(vx, vy, bx1 - px, by1 - py);
                        double t3 = dot(vx, vy, ax2 - px, ay2 - py);
                        double t4 = dot(vx, vy, bx2 - px, by2 - py);

                        double min_t = std::min(std::min(t1, t2), std::min(t3, t4));
                        double max_t = std::max(std::max(t1, t2), std::max(t3, t4));

                        ax1 = px + min_t * vx;
                        ay1 = py + min_t * vy;
                        bx1 = px + max_t * vx;
                        by1 = py + max_t * vy;
                        w1 = distance(ax1, ay1, bx1, by1);

                        processed[j] = true;
                    }
                }
            }
        }

        cv::Vec4f result;
        result[0] = static_cast<float>(ax1);
        result[1] = static_cast<float>(ay1);
        result[2] = static_cast<float>(bx1);
        result[3] = static_cast<float>(by1);
        output.push_back(result);
    }
}
