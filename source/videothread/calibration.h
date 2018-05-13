#pragma once

#include "../videothread.h"
#include "undistort.h"

QT_FORWARD_DECLARE_CLASS(Ui_Calibration);
QT_FORWARD_DECLARE_CLASS(ImageViewGl);

class VideoCalibrationThread : public VideoThread{
	Q_OBJECT

public:
    VideoCalibrationThread(VideoInput *video_input) : VideoThread(video_input) {};

    virtual ~VideoCalibrationThread();

    virtual QString initializeOnce(QWidget *parent) override;

    virtual void initialize(QWidget *parent) override;

    virtual void uninitialize(void) override;

    virtual void restoreSettings(const QSettings &settings) override;

    virtual void saveSettings(QSettings &settings) const override;

protected:
    virtual void processImage(const cv::Mat &input_image) override;

private:
    // CaptureListに表示するプレビューのサイズ
    static const int kPreviewSize = 64;

    // Captureボタンが押されてから撮影が行われるまでの猶予フレーム数
    static const int kCaptureDeadline = 5;

    // タブに表示するUI
    Ui_Calibration *m_ui;

    // 撮影が行える猶予フレーム数
    int m_CaptureCounter = 0;

    // チェスボードの点群情報
    std::vector<std::vector<cv::Point3f>> m_ObjectPoints;

    // 撮影された点群情報
    std::vector<std::vector<cv::Point2f>> m_ImagePoints[2];

    // 歪み補正器
    Undistort m_Undistort;

    // 表示画像
    ImageViewGl *m_Original[2], *m_Undistorted[2];

    // 取得した点群情報を破棄する
    Q_SLOT void clearAllPoints(void);

    // 取得した点群情報から歪み補正を行う
    Q_SLOT void applyCalibration(void);

    // 補正情報を保存する
    Q_SLOT void saveCalibration(void);
};
