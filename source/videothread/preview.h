#pragma once

#include "videothread.h"
#include "algorithm/undistort.h"

QT_FORWARD_DECLARE_CLASS(ImageViewGl);

class VideoPreviewThread : public VideoThread{
	Q_OBJECT

public:
    VideoPreviewThread(VideoInput *video_input) : VideoThread(video_input) {};

    virtual ~VideoPreviewThread();

    virtual QString initializeOnce(QWidget *parent) override;

    virtual void initialize(QWidget *parent) override;

    virtual void uninitialize(void) override;

protected:
    virtual void processImage(const cv::Mat &input_image) override;

private:
    Undistort m_Undistort;

    ImageViewGl *m_Left, *m_Right;
};
