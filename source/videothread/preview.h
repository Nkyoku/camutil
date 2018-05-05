#pragma once

#include "../videothread.h"

QT_FORWARD_DECLARE_CLASS(ImageViewGl);

class VideoPreviewThread : public VideoThread{
	Q_OBJECT

public:
    VideoPreviewThread(VideoInput *video_input);

    virtual ~VideoPreviewThread();

    virtual QString initializeOnce(QWidget *parent) override;

    virtual void initialize(QWidget *parent) override;

protected:
    virtual void processImage(const cv::Mat &input_image) override;

private:
    ImageViewGl *m_Left, *m_Right;
    ImageViewGl *m_Left2, *m_Right2;

};
