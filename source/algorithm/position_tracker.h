#pragma once

#include <opencv2/core.hpp>
#include "undistort.h"

// 姿勢位置推定を行う
class PositionTracker {
public:
    // 主要な白線の太さ [m]
    static constexpr double kThickness1 = 0.1;

    // フィールドの白線のテンプレート
    static const std::vector<cv::Vec4f> kLineTemplate;

    // コンストラクタ
    PositionTracker(void);

    // 推定を行う
    void estimate(const std::vector<cv::Vec4f> &left_lines, const std::vector<cv::Vec4f> &right_lines, const Undistort &undistort, cv::Mat &debug, int width, int height, int vanishing_y);





private:
    // 同一の線分だと見なす成す角の閾値1(cosθ)
    static constexpr double kSameSegmentAngle1 = 31.0 / 32.0;
    
    // 同一の線分だと見なす成す角の閾値2(cosθ)
    // 線分がほとんど重なっているときに用いる
    static constexpr double kSameSegmentAngle2 = 255.0 / 256.0;

    // 同一の線分だと見なす水平距離の閾値(入力画像サイズに対する割合)
    static constexpr double kSameSegmentHorizontalDistance = 1.0 / 32.0;

    // 同一の線分だと見なすユークリッド距離の閾値(入力画像サイズに対する割合)
    static constexpr double kSameSegmentDistance = 1.0 / 64.0;

    // 長さが等しいと見なす閾値(線分の長さに対する比率)
    static constexpr double kSameLengthForSegment = 1.0 / 64.0;

    // 長さが等しいと見なす閾値(入力画像サイズに対する割合)
    static constexpr double kSameLengthForImage = 1.0 / 64.0;






};
