#pragma once

#include <opencv2/core.hpp>

// ラドン変換
class RadonTransform {
public:
    // コンストラクタ
    RadonTransform(int angle_resolution);

    // 変換を行う
    // src  : CV_8U型
    // dest : CV_32F型
    void compute(const cv::Mat &src, cv::Mat &dest);

    // 角度を取得する [rad]
    double getAngle(int angle_index) {
        return m_AngleTable[angle_index];
    }

private:
    // 角度テーブル
    std::vector<double> m_AngleTable;

    // 三角関数テーブル
    std::vector<double> m_CosTable, m_SinTable;
};
