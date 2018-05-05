#include "preview.h"
#include "../imageviewgl.h"
#include <QtWidgets/QGridLayout>
#include <opencv2/imgproc.hpp>

VideoPreviewThread::VideoPreviewThread(VideoInput *video_input)
    : VideoThread(video_input)
{
	
}

VideoPreviewThread::~VideoPreviewThread(){
    quitThread();
}

QString VideoPreviewThread::initializeOnce(QWidget *parent) {
    return tr("Preview");
}

void VideoPreviewThread::initialize(QWidget *parent) {
    QGridLayout *grid_layout = new QGridLayout;
    parent->setLayout(grid_layout);

    m_Left = new ImageViewGl;
    grid_layout->addWidget(m_Left);

    connect(this, &VideoThread::update, m_Left, QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    //connect(this, &VideoThread::update, m_Right, QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    //connect(this, &VideoThread::update, m_Left2, QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    //connect(this, &VideoThread::update, m_Right2, QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
}

void VideoPreviewThread::processImage(const cv::Mat &input_image) {
    m_Left->setImage(input_image);
    //m_Right->setImage(input_image);
    //m_Left2->setImage(input_image);
    //m_Right2->setImage(input_image);
}
