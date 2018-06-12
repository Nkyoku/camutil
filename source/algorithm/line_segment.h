#pragma once

#include <vector>
#include <opencv2/core.hpp>

// 2つの点の距離あるいは線分の長さを計算する (double版)
// ax, ay : 点Aの座標あるいは線分の始点
// bx, by : 点Bの座標あるいは線分の終点
static inline double distance(double ax, double ay, double bx, double by) {
    return sqrt(pow(bx - ax, 2) + pow(by - ay, 2));
}

// 2つの点の距離あるいは線分の長さを計算する (cv::Point2d版)
// a : 点Aの座標あるいは線分の始点
// b : 点Bの座標あるいは線分の終点
static inline double distance(const cv::Point2d &a, const cv::Point2d &b) {
    return sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2));
}

// 2つの線分が交差しているか判定する
// ax, ay : 線分1の始点
// bx, by : 線分1の終点
// cx, cy : 線分2の始点
// dx, dy : 線分2の終点
bool segmentIntersectCheck(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy);

// 2つの直線の交点を計算する
// 直線が平行なときは(nan, nan)を返す
// a1, a2 : 直線Aの通る2点
// b1, b2 : 直線Bの通る2点
cv::Point2d lineIntersection(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2);

// 2つの線分の交点を計算する
// 線分が交差しない場合は直線とみなしたときの交点を返す
// 線分が平行なときは(nan, nan)を返す
// a1, a2 : 線分Aの始点，終点
// b1, b2 : 線分Bの始点，終点
// a_intersect : 交点が線分Aに含まれるときにtrueを返す
// b_intersect : 交点が線分Bに含まれるときにtrueを返す
cv::Point2d segmentIntersection(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2, bool *a_intersect, bool *b_intersect);

// 線分と点の距離を計算する
// ax, ay : 線分の始点
// bx, by : 線分の終点
// px, py : 点の座標
double distanceBetweenSegmentAndPoint(double ax, double ay, double bx, double by, double px, double py);

// 2つの線分の距離を計算する (double版)
// ax, ay : 線分1の始点
// bx, by : 線分1の終点
// cx, cy : 線分2の始点
// dx, dy : 線分2の終点
double distanceBetweenSegments(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy);

// 2つの線分の距離を計算する (cv::Point2d版)
// a1, a2 : 線分Aの始点，終点
// b1, b2 : 線分Bの始点，終点
static inline double distanceBetweenSegments(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2) {
    return distanceBetweenSegments(a1.x, a1.y, a2.x, a2.y, b1.x, b1.y, b2.x, b2.y);
}

// 2つの線分の距離を簡易的に計算する (double版)
// distanceBetweenSegments()の結果よりも常に小さい値を返す
// ax, ay : 線分1の始点
// bx, by : 線分1の終点
// cx, cy : 線分2の始点
// dx, dy : 線分2の終点
double distanceBetweenSegmentsSimple(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy);

// 2つの線分の距離を簡易的に計算する (cv::Point2d版)
// a1, a2 : 線分Aの始点，終点
// b1, b2 : 線分Bの始点，終点
static inline double distanceBetweenSegmentsSimple(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2) {
    return distanceBetweenSegmentsSimple(a1.x, a1.y, a2.x, a2.y, b1.x, b1.y, b2.x, b2.y);
}

// 点が直線の左右どちらにあるか調べる
// 左にあるか直線上にあるときtrueを返す
// p      : 点の座標
// a1, a2 : 直線の通る2点
bool isPointOnLeftSideOfLine(const cv::Point2d &p, const cv::Point2d &a1, const cv::Point2d &a2);

// 位置や傾きの近い線分を統合して線分数を削減する
// pos_threshold : 統合される座標のずれの閾値(px)
// dir_threshold : 統合される傾きのずれの閾値(cosθ)
void reduceSegments(const std::vector<cv::Vec4f> &input, std::vector<cv::Vec4f> &output, double pos_threshold = 5.0, double dir_threshold = 0.95);
