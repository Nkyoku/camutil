#pragma once

#include <opencv2/core.hpp>
#include "gradient_based_stereo_matching.h"
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

	// 既知のロール角とピッチ角を与える
	// ロール角が正のとき視点は右に傾いている、負のとき左に傾いている
	// ピッチ角が正のとき視点は下に傾いている、負のとき上に傾いている
	void setKnownRollAndPitchAngle(double roll_angle, double pitch_angle);

	// 既知のロール角とピット角をクリアする
	void unsetKnownRollAndPitchAngle(void);

	// 白線の見え方からフィールドの平面を推定する
	// 既知のロール角とピッチ角が与えられていると正確な平面推定が可能になる
    // 推定誤差を返す(0.0～1.0)
    double estimateFieldPlane(const std::vector<cv::Vec4f> &line_segments, const GradientBasedStereoMatching &stereo, const Undistort &undistort, cv::Point3d *field_centroid, cv::Point3d *field_normal);

	// 平面推定結果からロール角を求める
	// ロール角が既知であるときはその値を返す
	double calculateRollAngle(void) const;

	// 平面推定結果からピッチ角を求める
	// ピッチ角が既知であるときはその値を返す
	double calculatePitchAngle(void) const;

    // 平面推定結果から視点の高さを求める
	double calculateHeight(void) const;

	

	cv::Mat m_PCAMeans;
	cv::Mat m_PCAEigenVectors;
	cv::Mat m_PCAEigenValues;
    std::vector<cv::Point3f> m_UsedPoints;
private:
	// 視差の最大値
	static constexpr int kMaximumDisparity = 64;

	// 線分が水平だと見なす角度cosθ
	static constexpr double kHorizontalCosAngle = 63.0 / 64.0;

	// RANSACに使用するサンプル数
	static constexpr int kNumberOfRansacSamples = 64;

	// RANSACの試行回数
	static constexpr int kRansacTrials = 128;

	// 面に含まれると判定する点と面の距離 [m]
	static constexpr double kRansacThreshold = 1.0 / 4.0;

	// 垂直に近い線分リスト
	std::vector<cv::Vec4f> m_VerticalLineSegments;

	// 線分上の点の座標と視差のリスト
	std::vector<cv::Point3f> m_DisparitiesOfPoints;

	// 線分上の点の三次元座標
	std::vector<cv::Point3f> m_Points3d;

	// 各エッジ線分が持つ点の数のリスト
	std::vector<int> m_NumberOfPointsOnLines;

	// PCAによって推定されたフィールドの重心
	cv::Point3d m_FieldMean;

	// PCAによって推定されたフィールドの法線ベクトル
	cv::Point3d m_FieldNormal;

	// 事前情報として与えられた既知のロール角、ピッチ角
	double m_KnownRollAngle = 0.0, m_KnownPitchAngle = 0.0;

	// 既知のロール角とピッチ角が存在する
	bool m_IsKnownRollAndPitchAngleProvided = false;





	// ブレゼンハムのアルゴリズムで線分上の点の位置を出力する
	// 出力した点の数を返す
	static int getPointsOnLineSegment(const cv::Vec4f &line_segment, std::vector<cv::Point3f> &points, int width, int height);
};
