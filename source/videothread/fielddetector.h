#pragma once

#include "videothread.h"
#include "algorithm/undistort.h"
#include "algorithm/enhancement_filter.h"
#include "algorithm/field_detector.h"
#include "algorithm/gradient_based_stereo_matching.h"
#include "algorithm/position_tracker.h"
#include <opencv2/imgproc.hpp>
#include <QtWidgets/QSpinBox>

QT_FORWARD_DECLARE_CLASS(ImageViewGl);

class VideoFieldDetectorThread : public VideoThread{
	Q_OBJECT

public:
    VideoFieldDetectorThread(VideoInput *video_input);

    virtual ~VideoFieldDetectorThread();

    virtual QString initializeOnce(QWidget *parent) override;

    virtual void initialize(QWidget *parent) override;

    virtual void uninitialize(void) override;

    virtual void restoreSettings(const QSettings &settings) override;

    virtual void saveSettings(QSettings &settings) const override;

protected:
    virtual void processImage(const cv::Mat &input_image) override;

private:
    // 縮小比率
    static const int kScaleFactor = 8;

    // 歪み補正器
    Undistort m_Undistort;

    // 白線強調フィルタ
    EnhancementFilter m_EnhancementFilter;

    // フィールド検知器
    FieldDetector m_FieldDetector;

    // ステレオマッチング
    GradientBasedStereoMatching m_Stereo;

    // 姿勢位置推定器
    PositionTracker m_PositionTracker;

    // 画像を表示するウィジェット
    ImageViewGl *m_Output[4];

    // 色情報を表示したい座標
    int m_WatchPointX = -1, m_WatchPointY = -1;

    // オリジナルカラー画像
    cv::Mat m_ColorImage[2];

    // オリジナルグレースケール画像
    cv::Mat m_GrayscaleImage[2];

    // L*a*b*変換された強調画像
    cv::Mat m_LabEnhancedImage;

    // L*a*b*変換された中間値画像
    cv::Mat m_LabMedianedImage;

    // 視差画像
    cv::Mat m_DisparityMap;

    // 強調されたカラー画像
    cv::Mat m_EnhancedColorImage;

    // 検知された白線の画像
    cv::Mat m_WhiteLineImage;

    // 傾きを補正した画像
    cv::Mat m_UnrollImage;

    // 奥行を補正した画像
    cv::Mat m_UnpitchImage;

    // 指定した座標の色情報を表示する
    Q_SLOT void showColor(int x, int y);


};
