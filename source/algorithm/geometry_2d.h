#pragma once

#include <vector>
#include <opencv2/core.hpp>



////////////////////////////////////
////// �_�Ƃ̋��������߂�֐� //////
////////////////////////////////////

// 2�̓_�̋������邢�͐����̒������v�Z����
// a : �_A�̍��W���邢�͐����̎n�_
// b : �_B�̍��W���邢�͐����̏I�_
static inline double distance(const cv::Point2d &a, const cv::Point2d &b) {
    return sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2));
}



//////////////////////////////////////
////// �����Ƃ̋��������߂�֐� //////
//////////////////////////////////////

// �����Ɠ_�̋������v�Z����
// a1, a2 : ����A�̎n�_�C�I�_
// px, py : �_P�̍��W
double distanceBetweenSegmentAndPoint(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &p);

// 2�̐����̋������v�Z����
// a1, a2 : ����A�̎n�_�C�I�_
// b1, b2 : ����B�̎n�_�C�I�_
double distanceBetweenSegments(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2);

// 2�̐����̋������ȈՓI�Ɍv�Z����
// a1, a2 : ����A�̎n�_�C�I�_
// b1, b2 : ����B�̎n�_�C�I�_
double distanceBetweenSegmentsSimple(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2);



//////////////////////////////////////
////// �����Ƃ̋��������߂�֐� //////
//////////////////////////////////////

// �����Ɠ_�̋������v�Z����
// a1, a2 : ����A�̒ʂ�2�_
// p      : �_P�̍��W
double distanceBetweenLineAndPoint(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &p);

// �����Ɛ����̋������v�Z����
// a1, a2 : ����A�̒ʂ�2�_
// b1, b2 : ����B�̎n�_�C�I�_
double distanceBetweenLineAndSegment(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2);



//////////////////////////////////////
////// �������Ă��邩���߂�֐� //////
//////////////////////////////////////

// 2�̐������������Ă��邩���肷��
// a1, a2 : ����A�̎n�_�C�I�_
// b1, b2 : ����B�̎n�_�C�I�_
bool segmentIntersectCheck(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2);



//////////////////////////////
////// ��_�����߂�֐� //////
//////////////////////////////

// 2�̐����̌�_���v�Z����
// �������������Ȃ��ꍇ�͒����Ƃ݂Ȃ����Ƃ��̌�_��Ԃ�
// ���������s�ȂƂ���(nan, nan)��Ԃ�
// a1, a2 : ����A�̎n�_�C�I�_
// b1, b2 : ����B�̎n�_�C�I�_
// a_intersect : ��_������A�Ɋ܂܂��Ƃ���true��Ԃ�
// b_intersect : ��_������B�Ɋ܂܂��Ƃ���true��Ԃ�
cv::Point2d segmentIntersection(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2, bool *a_intersect, bool *b_intersect);

// 2�̒����̌�_���v�Z����
// ���������s�ȂƂ���(nan, nan)��Ԃ�
// a1, a2 : ����A�̒ʂ�2�_
// b1, b2 : ����B�̒ʂ�2�_
cv::Point2d lineIntersection(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2);


//////////////////////////////
////// ���������߂�֐� //////
//////////////////////////////

// �_�������̍��E�ǂ���ɂ��邩���ׂ�
// ���ɂ��邩������ɂ���Ƃ�true��Ԃ�
// p      : �_P�̍��W
// a1, a2 : ����A�̒ʂ�2�_
bool isPointOnLeftSideOfLine(const cv::Point2d &p, const cv::Point2d &a1, const cv::Point2d &a2);



//////////////////////
////// ���p�֐� //////
//////////////////////

// �x�N�g���𐳋K������ƂƂ��ɒ�����Ԃ�
static inline double normalizeAndLength(cv::Point2d &a) {
    double length_a = sqrt(a.x * a.x + a.y * a.y);
    a *= 1.0 / length_a;
    return length_a;
}

// 2�̐�������������
// a1, a2 : ����A�̎n�_�C�I�_
// b1, b2 : ����B�̎n�_�C�I�_
// c1, c2 : �������ꂽ����C�̎n�_�C�I�_
void combine2Segments(const cv::Point2d &a1, const cv::Point2d &a2, const cv::Point2d &b1, const cv::Point2d &b2, cv::Point2d &c1, cv::Point2d &c2);

// �ʒu��X���̋߂������𓝍����Đ��������팸����
// pos_threshold : �����������W�̂����臒l(px)
// dir_threshold : ���������X���̂����臒l(cos��)
void reduceSegments(const std::vector<cv::Vec4f> &input, std::vector<cv::Vec4f> &output, double pos_threshold = 5.0, double dir_threshold = 0.95);






