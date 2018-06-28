#pragma once

#include "videothread.h"
#include "algorithm/undistort.h"
#include "algorithm/field_detector.h"
#include <opencv2/imgproc.hpp>
#include <QtWidgets/QDoubleSpinBox>

QT_FORWARD_DECLARE_CLASS(ImageViewGl);

class VideoLineStereoThread : public VideoThread{
	Q_OBJECT

public:
    VideoLineStereoThread(VideoInput *video_input);

    virtual ~VideoLineStereoThread();

    virtual QString initializeOnce(QWidget *parent) override;

    virtual void initialize(QWidget *parent) override;

    virtual void uninitialize(void) override;

    virtual void restoreSettings(const QSettings &settings) override;

    virtual void saveSettings(QSettings &settings) const override;

protected:
    virtual void processImage(const cv::Mat &input_image) override;

private:
    // 2本の直線が平行だと見なすcosθ
    static constexpr double kParallelAngle = 255.0 / 256.0;
    
    // 歪み補正器
    Undistort m_Undistort;

    // フィールド検知器
    FieldDetector m_FieldDetector;

    // LineSegmentDetector
    cv::Ptr<cv::LineSegmentDetector> m_Lsd;

    // 画像を表示するウィジェット
    ImageViewGl *m_Output[4];

    // 角度を入力するウィジェット
    QDoubleSpinBox *m_PitchAngleInput, *m_RollAngleInput;

    // 高さを入力するウィジェット
    QDoubleSpinBox *m_HeightInput;

    // オリジナルカラー画像
    cv::Mat m_OriginalBgrImage[2];

    // オリジナルグレースケール画像
    cv::Mat m_OriginalGrayscaleImage[2];

    // オリジナルL*a*b*画像
    cv::Mat m_OriginalLabImage[2];


    

    // 平行な線分を太さを持つ線分に合成する
    static void combineParallelSegments(const std::vector<cv::Vec4f> &input_segments, std::vector<cv::Vec4f> &output_segments, double threshold);

    // 画像に線分リストを描画する
    static void drawSegments(cv::Mat &image, const std::vector<cv::Vec4f> &line_segments, const cv::Scalar &color, int thickness = 1);
};
