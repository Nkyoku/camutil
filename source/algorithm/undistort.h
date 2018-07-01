#pragma once

#include <opencv2/core.hpp>

class Undistort {
public:
    // コンストラクタ
    Undistort(void);

    // カメラ解像度を指定して補正情報を読み込む
    // 読み込みに失敗した場合でも現在の補正情報は失われる
    bool load(int width, int height);

    // 現在の補正情報を保存する
    // new_width,new_heightに値を指定するとカメラ行列がその解像度に変換されて保存される
    bool save(int new_width = 0, int new_height = 0);

    // カメラ単体のキャリブレーションが行われているか取得する
    // side==0で左カメラ、side==1で右カメラについて取得する
    bool isCalibrated(bool side) const {
        return m_IsCalibrated[side];
    }

    // ステレオ平行化が行われているか取得する
    bool isStereoRectified(void) const {
        return m_IsStereoRectified;
    }

    // 補正を行う
    // side==0で左カメラ、side==1で右カメラの補正を行う
    // 補正が行われないときfalseを返し、undistorted_imageには空のMatを返す
    bool undistort(const cv::Mat &distorted_image, cv::Mat &undistorted_image, int side) const;

    // 補正を行う
    // side==0で左カメラ、side==1で右カメラの補正を行う
    // 補正が行われないとき、empty_if_uncalibrated==falseであればオリジナルの画像を返し、trueであれば空のcv::Matを返す
    cv::Mat undistort(const cv::Mat &distorted_image, int side, bool empty_if_uncalibrated = false) const;

    // 視差マップから実空間の座標マップを計算する
    bool reprojectImageTo3D(const cv::Mat &disparity, cv::Mat &output, bool missing_value = false) const;

    // 座標と視差の組み合わせから実空間の座標を計算する
    bool reprojectPointsTo3D(const std::vector<cv::Point3f> &disparities, std::vector<cv::Point3f> &output) const;

    // 実空間の座標からカメラに投影される座標を計算する
    bool projectPointsTo2D(const std::vector<cv::Point3f> &points3d, std::vector<cv::Point2f> &points2d, int side = 0) const;

    // チェスボードのパターンからカメラ単体のキャリブレーションを行う
    // side==0で左カメラ、side==1で右カメラについて行う
    bool calibrate(int side, int width, int height, const std::vector<std::vector<cv::Point3f>> &object_points, const std::vector<std::vector<cv::Point2f>> &image_points);

    // チェスボードのパターンからステレオ平行化を行う
    // 事前にカメラ単体のキャリブレーションが完了している必要がある
    bool stereoRectify(int width, int height, const std::vector<std::vector<cv::Point3f>> &object_points, const std::vector<std::vector<cv::Point2f>> &image_points_left, const std::vector<std::vector<cv::Point2f>> &image_points_right);

    // ステレオ平行化の微調整を行う
    bool adjustRectification(double roll, double pitch);

    // 補正情報を破棄する
    void destroy(void);

	// 横の解像度を取得する
	int width(void) const {
		return m_Width;
	}

	// 縦の解像度を取得する
	int height(void) const {
		return m_Height;
	}

    // 各カメラのカメラ行列(3x3)を取得する
    const cv::Mat& cameraMatrix(int side) const {
        return m_CameraMatrix[side];
    }

    // 各カメラのプロジェクション行列(4x3)を取得する
    const cv::Mat& projectionMatrix(int side) const {
        return m_ProjectionMatrix[side];
    }

    // カメラのベースライン長を取得する
    double baselineLength(void) const;

    // 角度調整パラメータを取得する
    void adjustmentParameters(double *roll, double *pitch) const {
        *roll = m_RollAdjustment;
        *pitch = m_PitchAdjustment;
    }

private:
    // 単体キャリブレーションが完了している
    bool m_IsCalibrated[2] = { false, false };

    // ステレオ平行化が完了している
    bool m_IsStereoRectified = false;

    // 解像度
    int m_Width = 0, m_Height = 0;

    // カメラの角度調整パラメータ
    double m_RollAdjustment = 0.0, m_PitchAdjustment = 0.0;

    // カメラの内部パラメータ行列
    cv::Mat m_RawCameraMatrix[2], m_CameraMatrix[2];

    // 歪み係数
    cv::Mat m_RawDistortionCoefficients[2], m_DistortionCoefficients[2];

    // 平行化変換行列
    cv::Mat m_RectificationMatrix[2];

    // 投影行列
    cv::Mat m_ProjectionMatrix[2];

    // 偏差と深度の変換行列
    cv::Mat m_DisparityMatrix;

    // 2つのカメラの回転行列
    cv::Mat m_RotationMatrix;

    // 2つのカメラの並進ベクトル
    cv::Mat m_TranslationVector;

    // 歪み補正マップ
    cv::Mat m_Map1[2], m_Map2[2];

    // カメラ単体の歪み補正マップを生成する
    void generateMap(int side);

    // ステレオの歪み補正マップを生成する
    void generateMapStereo(void);

    // 解像度に応じた補正情報ファイル名を生成する
    static std::string generateFileName(int width, int height);

};
