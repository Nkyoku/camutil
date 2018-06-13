#pragma once

#include <vector>
#include <opencv2/core.hpp>



////////////////////////////////////
////// 点との距離を求める関数 //////
////////////////////////////////////

// 2つの点の距離あるいは線分の長さを計算する
// a : 点Aの座標あるいは線分の始点
// b : 点Bの座標あるいは線分の終点
static inline double distance(const cv::Point2d &a, const cv::Point2d &b) {
    return sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2));
}



//////////////////////////////////////
////// 線分との距離を求める関数 //////
//////////////////////////////////////

// 線分と点の距離を計算する
// a1, a2 : 線分Aの始点，終点
// px, py : 点Pの座標
double distanceBetweenSegmentAndPoint(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &p);

// 2つの線分の距離を計算する
// a1, a2 : 線分Aの始点，終点
// b1, b2 : 線分Bの始点，終点
double distanceBetweenSegments(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2);

// 2つの線分の距離を簡易的に計算する
// a1, a2 : 線分Aの始点，終点
// b1, b2 : 線分Bの始点，終点
double distanceBetweenSegmentsSimple(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2);



//////////////////////////////////////
////// 直線との距離を求める関数 //////
//////////////////////////////////////

// 直線と点の距離を計算する
// a1, a2 : 直線Aの通る2点
// p      : 点Pの座標
double distanceBetweenLineAndPoint(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &p);

// 直線と線分の距離を計算する
// a1, a2 : 直線Aの通る2点
// b1, b2 : 線分Bの始点，終点
double distanceBetweenLineAndSegment(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2);



//////////////////////////////////////
////// 交差しているか求める関数 //////
//////////////////////////////////////

// 2つの線分が交差しているか判定する
// a1, a2 : 線分Aの始点，終点
// b1, b2 : 線分Bの始点，終点
bool segmentIntersectCheck(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2);



//////////////////////////////
////// 交点を求める関数 //////
//////////////////////////////

// 2つの線分の交点を計算する
// 線分が交差しない場合は直線とみなしたときの交点を返す
// 線分が平行なときは(nan, nan)を返す
// a1, a2 : 線分Aの始点，終点
// b1, b2 : 線分Bの始点，終点
// a_intersect : 交点が線分Aに含まれるときにtrueを返す
// b_intersect : 交点が線分Bに含まれるときにtrueを返す
cv::Point2d segmentIntersection(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2, bool *a_intersect, bool *b_intersect);

// 2つの直線の交点を計算する
// 直線が平行なときは(nan, nan)を返す
// a1, a2 : 直線Aの通る2点
// b1, b2 : 直線Bの通る2点
cv::Point2d lineIntersection(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2);


//////////////////////////////
////// 方向を求める関数 //////
//////////////////////////////

// 点が直線の左右どちらにあるか調べる
// 左にあるか直線上にあるときtrueを返す
// p      : 点Pの座標
// a1, a2 : 直線Aの通る2点
bool isPointOnLeftSideOfLine(const cv::Point2d &p, const cv::Point2d &a1, const cv::Point2d &a2);



//////////////////////
////// 応用関数 //////
//////////////////////

// ベクトルを正規化するとともに長さを返す
static inline double normalizeAndLength(cv::Point2d &a) {
    double length_a = sqrt(a.x * a.x + a.y * a.y);
    a *= 1.0 / length_a;
    return length_a;
}

// 2つの線分を合成する
// a1, a2 : 線分Aの始点，終点
// b1, b2 : 線分Bの始点，終点
// c1, c2 : 合成された線分Cの始点，終点
void combine2Segments(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2, cv::Point2d &c1, cv::Point2d &c2);

// 位置や傾きの近い線分を統合して線分数を削減する
// pos_threshold : 統合される座標のずれの閾値(px)
// dir_threshold : 統合される傾きのずれの閾値(cosθ)
void reduceSegments(const std::vector<cv::Vec4f> &input, std::vector<cv::Vec4f> &output, double pos_threshold = 5.0, double dir_threshold = 0.95);






