﻿#pragma once

#include <opencv2/core.hpp>

class Undistort {
public:
    // コンストラクタ
    Undistort(void);

    // 補正情報を読み込む
    bool load(int width, int height);

    // 現在の補正情報を保存する
    bool save(void);

    // キャリブレーションが行われているか取得する
    bool isCalibrated(void) const {
        return m_IsCalibrated;
    }

    // 補正を行う
    // side==0で左カメラ、side==1で右カメラの補正を行う
    // 補正が行われないときfalseを返し、undistorted_imageには空のMatを返す
    bool undistort(const cv::Mat &distorted_image, cv::Mat &undistorted_image, int side);

    // 補正を行う
    // side==0で左カメラ、side==1で右カメラの補正を行う
    // 補正が行われないとき、empty_if_uncalibrated==falseであればオリジナルの画像を返し、trueであれば空のMatを返す
    cv::Mat undistort(const cv::Mat &distorted_image, int side, bool empty_if_uncalibrated = false);

    // 視差マップから実空間の座標マップを計算する
    bool reprojectImageTo3D(const cv::Mat &disparity, cv::Mat &output, bool missing_value = false);

    // 座標と視差の組み合わせから実空間の座標を計算する
    bool reprojectPointsTo3D(const cv::Mat &disparity, cv::Mat &output);

    // キャリブレーション結果から補正マップを生成する
    bool calibrate(int width, int height, const std::vector<std::vector<cv::Point3f>> &object_points, const std::vector<std::vector<cv::Point2f>> &image_points_left, const std::vector<std::vector<cv::Point2f>> &image_points_right);

    // 補正情報を破棄する
    void destroy(void);

private:
    // キャリブレーションが完了している
    bool m_IsCalibrated = false;

    // 解像度
    int m_Width = 0, m_Height = 0;

    // カメラの内部パラメータ行列
    cv::Mat m_CameraMatrix[2];

    // 歪み係数
    cv::Mat m_DistortionCoefficients[2];

    // 平行化変換行列
    cv::Mat m_RectificationMatrix[2];

    // 投影行列
    cv::Mat m_ProjectionMatrix[2];

    // 偏差と深度の変換行列
    cv::Mat m_DisparityMatrix;

    // 歪み補正マップ
    cv::Mat m_Map1[2], m_Map2[2];

    // 補正行列から歪み補正マップを生成する
    void generateMap(const cv::Size &size);

    // 解像度に応じた補正情報ファイル名を生成する
    static std::string generateFileName(int width, int height);

};
