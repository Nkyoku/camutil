#pragma once

#include "../videothread.h"

QT_FORWARD_DECLARE_CLASS(Ui_LensCalibration);
QT_FORWARD_DECLARE_CLASS(ImageViewGl);

class VideoLensCalibrationThread : public VideoThread{
	Q_OBJECT

public:
    VideoLensCalibrationThread(VideoInput *video_input) : VideoThread(video_input) {};

    virtual ~VideoLensCalibrationThread();

    virtual QString initializeOnce(QWidget *parent) override;

    virtual void initialize(QWidget *parent) override;

    virtual void restoreSettings(const QSettings &settings) override;

    virtual void saveSettings(QSettings &settings) const override;

protected:
    virtual void processImage(const cv::Mat &input_image) override;

private:
    // CaptureListに表示するプレビューのサイズ
    static const int kPreviewSize = 64;

    // タブに表示するUI
    Ui_LensCalibration *m_ui;

    // 各ボタンが押された
    bool m_IsCaptuedPushed = false, m_IsCalibratePushed = false, m_IsClearPushed = false;

    // チェスボードの点群情報
    std::vector<std::vector<cv::Point3f>> m_ObjectPoints;

    // 撮影された点群情報
    std::vector<std::vector<cv::Point2f>> m_ImagePoints;

    // カメラの内部パラメータ行列
    cv::Mat m_CameraMatrix[2];

    // 歪み係数
    cv::Mat m_DistortionCoefficients[2];

    // 表示画像
    ImageViewGl *m_Original, *m_Undistort;
};
