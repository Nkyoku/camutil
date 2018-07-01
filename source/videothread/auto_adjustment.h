#pragma once

#include "videothread.h"
#include "algorithm/undistort.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <random>

QT_FORWARD_DECLARE_CLASS(Ui_AutoAdjustment);
QT_FORWARD_DECLARE_CLASS(ImageViewGl);

class VideoAutoAdjustmentThread : public VideoThread{
	Q_OBJECT

public:
    VideoAutoAdjustmentThread(VideoInput *video_input);

    virtual ~VideoAutoAdjustmentThread();

    virtual QString initializeOnce(QWidget *parent) override;

    virtual void initialize(QWidget *parent) override;

    virtual void uninitialize(void) override;

    virtual void restoreSettings(const QSettings &settings) override;

    virtual void saveSettings(QSettings &settings) const override;

protected:
    virtual void processImage(const cv::Mat &input_image) override;

private:
    // 角度の最大値 [deg]
    static constexpr double kMaximumAngle = 1.0;

    // 最小値の探索の際の角度の分解能 [deg]
    static constexpr double kDeltaAngle = 0.001;

    // タブに表示するUI
    Ui_AutoAdjustment * m_ui;

    // 歪み補正器
    Undistort m_Undistort;

    // 特徴量検知器
    cv::Ptr<cv::AKAZE> m_Akaze;

    // 特徴点のマッチング器
    cv::Ptr<cv::DescriptorMatcher> m_Matcher;

    // ステレオマッチング器
    cv::Ptr<cv::StereoSGBM> m_StereoSgbm;

    // 乱数生成器
    std::mt19937 m_Random;
    
    // 角度の調整値
    double m_RollAngle = 0.0, m_PitchAngle = 0.0;

    // 調整値のキャプチャ中フラグ
    bool m_IsCapturing = false;

    // キャプチャした特徴点のずれとX座標のタプルのリスト
    std::vector<cv::Vec2f> m_CaptureData;

    // 表示画像
    ImageViewGl *m_OriginalOutput[2], *m_StereoOutput;

    // オリジナル画像
    cv::Mat m_OriginalImage[2];

    // グレースケール画像
    cv::Mat m_GrayscaleImage[2];

    // ステレオ画像
    cv::Mat m_StereoFixedPoint, m_Stereo8Bit;

    // ロール角を変更する
    Q_SLOT void setRollAngle(double roll);

    // ピッチ角を変更する
    Q_SLOT void setPitchAngle(double pitch);

    // キャプチャを開始する
    Q_SLOT void startCapture(void);

    // キャプチャを終了する
    Q_SLOT void finishCapture(bool discard = false);

    // 指定した角度での分散と平均を計算する
    void calculateVariance(double angle, double *mean, double *variance);

    // 補正情報適用する
    Q_SLOT void applyAdjustment(void);

    // 補正情報を保存する
    Q_SLOT void saveAdjustment(void);
};
