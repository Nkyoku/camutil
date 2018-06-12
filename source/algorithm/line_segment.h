#pragma once

#include <vector>
#include <opencv2/core.hpp>

// 2�̓_�̋������邢�͐����̒������v�Z���� (double��)
// ax, ay : �_A�̍��W���邢�͐����̎n�_
// bx, by : �_B�̍��W���邢�͐����̏I�_
static inline double distance(double ax, double ay, double bx, double by) {
    return sqrt(pow(bx - ax, 2) + pow(by - ay, 2));
}

// 2�̓_�̋������邢�͐����̒������v�Z���� (cv::Point2d��)
// a : �_A�̍��W���邢�͐����̎n�_
// b : �_B�̍��W���邢�͐����̏I�_
static inline double distance(const cv::Point2d &a, const cv::Point2d &b) {
    return sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2));
}

// 2�̐������������Ă��邩���肷��
// ax, ay : ����1�̎n�_
// bx, by : ����1�̏I�_
// cx, cy : ����2�̎n�_
// dx, dy : ����2�̏I�_
bool segmentIntersectCheck(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy);

// 2�̒����̌�_���v�Z����
// ���������s�ȂƂ���(nan, nan)��Ԃ�
// a1, a2 : ����A�̒ʂ�2�_
// b1, b2 : ����B�̒ʂ�2�_
cv::Point2d lineIntersection(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2);

// 2�̐����̌�_���v�Z����
// �������������Ȃ��ꍇ�͒����Ƃ݂Ȃ����Ƃ��̌�_��Ԃ�
// ���������s�ȂƂ���(nan, nan)��Ԃ�
// a1, a2 : ����A�̎n�_�C�I�_
// b1, b2 : ����B�̎n�_�C�I�_
// a_intersect : ��_������A�Ɋ܂܂��Ƃ���true��Ԃ�
// b_intersect : ��_������B�Ɋ܂܂��Ƃ���true��Ԃ�
cv::Point2d segmentIntersection(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2, bool *a_intersect, bool *b_intersect);

// �����Ɠ_�̋������v�Z����
// ax, ay : �����̎n�_
// bx, by : �����̏I�_
// px, py : �_�̍��W
double distanceBetweenSegmentAndPoint(double ax, double ay, double bx, double by, double px, double py);

// 2�̐����̋������v�Z���� (double��)
// ax, ay : ����1�̎n�_
// bx, by : ����1�̏I�_
// cx, cy : ����2�̎n�_
// dx, dy : ����2�̏I�_
double distanceBetweenSegments(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy);

// 2�̐����̋������v�Z���� (cv::Point2d��)
// a1, a2 : ����A�̎n�_�C�I�_
// b1, b2 : ����B�̎n�_�C�I�_
static inline double distanceBetweenSegments(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2) {
    return distanceBetweenSegments(a1.x, a1.y, a2.x, a2.y, b1.x, b1.y, b2.x, b2.y);
}

// 2�̐����̋������ȈՓI�Ɍv�Z���� (double��)
// distanceBetweenSegments()�̌��ʂ�����ɏ������l��Ԃ�
// ax, ay : ����1�̎n�_
// bx, by : ����1�̏I�_
// cx, cy : ����2�̎n�_
// dx, dy : ����2�̏I�_
double distanceBetweenSegmentsSimple(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy);

// 2�̐����̋������ȈՓI�Ɍv�Z���� (cv::Point2d��)
// a1, a2 : ����A�̎n�_�C�I�_
// b1, b2 : ����B�̎n�_�C�I�_
static inline double distanceBetweenSegmentsSimple(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2) {
    return distanceBetweenSegmentsSimple(a1.x, a1.y, a2.x, a2.y, b1.x, b1.y, b2.x, b2.y);
}

// �_�������̍��E�ǂ���ɂ��邩���ׂ�
// ���ɂ��邩������ɂ���Ƃ�true��Ԃ�
// p      : �_�̍��W
// a1, a2 : �����̒ʂ�2�_
bool isPointOnLeftSideOfLine(const cv::Point2d &p, const cv::Point2d &a1, const cv::Point2d &a2);

// �ʒu��X���̋߂������𓝍����Đ��������팸����
// pos_threshold : �����������W�̂����臒l(px)
// dir_threshold : ���������X���̂����臒l(cos��)
void reduceSegments(const std::vector<cv::Vec4f> &input, std::vector<cv::Vec4f> &output, double pos_threshold = 5.0, double dir_threshold = 0.95);
