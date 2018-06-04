#pragma once

#include "videothread.h"
#include "undistort.h"
#include <opencv2/calib3d.hpp>

QT_FORWARD_DECLARE_CLASS(ImageViewGl);

class VideoSgbmThread : public VideoThread{
	Q_OBJECT

public:
    VideoSgbmThread(VideoInput *video_input);

    virtual ~VideoSgbmThread();

    virtual QString initializeOnce(QWidget *parent) override;

    virtual void initialize(QWidget *parent) override;

    virtual void uninitialize(void) override;

    virtual void restoreSettings(const QSettings &settings) override;

    virtual void saveSettings(QSettings &settings) const override;

protected:
    virtual void processImage(const cv::Mat &input_image) override;

private:
    // 歪み補正器
    Undistort m_Undistort;

    // SGBM法
    cv::Ptr<cv::StereoSGBM> m_Sgbm;

    // 設定された最大偏差
    int m_MaxDisparity = 64;

    // 表示画像
    ImageViewGl *m_Color[2], *m_Depth;
};
