#pragma once

#include "videothread.h"
#include "algorithm/undistort.h"
#include "algorithm/gaussian_dog.h"
#include "algorithm/gradient_based_stereo_matching.h"
#include <opencv2/core/ocl.hpp>

QT_FORWARD_DECLARE_CLASS(ImageViewGl);

class VideoGradientThread : public VideoThread{
	Q_OBJECT

public:
    VideoGradientThread(VideoInput *video_input);

    virtual ~VideoGradientThread();

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

    // フィルタ
    //GaussianDoG m_GaussianDoG;

    // ステレオマッチング
    GradientBasedStereoMatching m_StereoMatching;

    // 表示画像
    ImageViewGl *m_Color[2], *m_Gradient[2];

    // オリジナル画像
    cv::Mat m_OriginalImage[2];

    // グレースケール画像
    cv::Mat m_GrayscaleImage[2];

};
