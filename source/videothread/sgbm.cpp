#include "sgbm.h"
#include "../imageviewgl.h"
#include <QtWidgets/QGridLayout>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

VideoSgbmThread::VideoSgbmThread(VideoInput *video_input)
    : VideoThread(video_input)
{
    m_Sgbm = cv::StereoSGBM::create(0, m_MaxDisparity, 5);
};

VideoSgbmThread::~VideoSgbmThread(){
    quitThread();
}

QString VideoSgbmThread::initializeOnce(QWidget *parent) {
    return tr("SGBM");
}

void VideoSgbmThread::initialize(QWidget *parent) {
    uninitialize();

    QSize size = m_VideoInput->sourceResolution();
    m_Undistort.load(size.width() / 2, size.height());

    // カラー画像と深度画像を表示するImageViewGlを生成
    QGridLayout *grid_layout = new QGridLayout;
    parent->setLayout(grid_layout);
    for (int side = 0; side < 2; side++) {
        m_Color[side] = new ImageViewGl;
        m_Color[side]->convertBgrToRgb();
        grid_layout->addWidget(m_Color[side], 0, side);
        connect(this, &VideoThread::update, m_Color[side], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
        
    }
    m_Depth = new ImageViewGl;
    grid_layout->addWidget(m_Depth, 1, 0, 1, 2);
    connect(this, &VideoThread::update, m_Depth, QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
}

void VideoSgbmThread::uninitialize(void) {
    m_Undistort.destroy();
}

void VideoSgbmThread::restoreSettings(const QSettings &settings) {
    
}

void VideoSgbmThread::saveSettings(QSettings &settings) const {
    
}

void VideoSgbmThread::processImage(const cv::Mat &input_image) {
    int width = input_image.cols / 2;
    int height = input_image.rows;
    
    cv::Mat color[2];
    m_Undistort.undistort(cv::Mat(input_image, cv::Rect(0, 0, width, height)), color[0], 0);
    m_Undistort.undistort(cv::Mat(input_image, cv::Rect(width, 0, width, height)), color[1], 1);

    cv::Mat gray[2];
    cv::cvtColor(color[0], gray[0], cv::COLOR_BGR2GRAY);
    cv::cvtColor(color[1], gray[1], cv::COLOR_BGR2GRAY);

    cv::Mat depth, depth8;
    m_Sgbm->compute(gray[0], gray[1], depth);

    double min, max;
    cv::minMaxLoc(depth, &min, &max);
    depth.convertTo(depth8, CV_8UC1, 255.0 / (max - min), -255.0 * min / (max - min));

    m_Color[0]->setImage(color[0]);
    m_Color[1]->setImage(color[1]);
    m_Depth->setImage(depth8);
}
