#include "gradient.h"
#include "imageviewgl.h"
#include <QtWidgets/QGridLayout>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <fstream>

VideoGradientThread::VideoGradientThread(VideoInput *video_input)
    : VideoThread(video_input)
{
   
}

VideoGradientThread::~VideoGradientThread(){
    quitThread();
}

QString VideoGradientThread::initializeOnce(QWidget *parent) {
    m_StereoMatching.setMaximumDisparity(32);
    return tr("Gradient");
}

void VideoGradientThread::initialize(QWidget *parent) {
    uninitialize();

    QSize size = m_VideoInput->sourceResolution();
    m_Undistort.load(size.width() / 2, size.height());

    // カラー画像と勾配画像を表示するImageViewGlを生成
    QGridLayout *grid_layout = new QGridLayout;
    parent->setLayout(grid_layout);
    for (int side = 0; side < 2; side++) {
        m_Color[side] = new ImageViewGl;
        m_Color[side]->convertBgrToRgb();
        grid_layout->addWidget(m_Color[side], 0, side);
        connect(this, &VideoThread::update, m_Color[side], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
        
    }
    for (int side = 0; side < 2; side++) {
        m_Gradient[side] = new ImageViewGl;
        grid_layout->addWidget(m_Gradient[side], 1, side);
        connect(this, &VideoThread::update, m_Gradient[side], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);

    }
}

void VideoGradientThread::uninitialize(void) {
    m_Undistort.destroy();
}

void VideoGradientThread::restoreSettings(const QSettings &settings) {
    
}

void VideoGradientThread::saveSettings(QSettings &settings) const {
    
}

void VideoGradientThread::processImage(const cv::Mat &input_image) {
    int width = input_image.cols / 2;
    int height = input_image.rows;
    
    m_Undistort.undistort(cv::Mat(input_image, cv::Rect(0, 0, width, height)), m_OriginalImage[0], 0);
    m_Undistort.undistort(cv::Mat(input_image, cv::Rect(width, 0, width, height)), m_OriginalImage[1], 1);

    cv::Mat disparity_map, likelihood_map;
    m_StereoMatching.compute(m_OriginalImage[0], m_OriginalImage[1], disparity_map, likelihood_map);

    cv::Mat disparity_map_8bit, likelihood_map_8bit;
    disparity_map.convertTo(disparity_map_8bit, CV_8U, 8.0);
    likelihood_map.convertTo(likelihood_map_8bit, CV_8U, 10.0);
    
    m_Color[0]->setImage(m_OriginalImage[0]);
    m_Color[1]->setImage(m_OriginalImage[1]);
    m_Gradient[0]->setImage(disparity_map_8bit);
    m_Gradient[1]->setImage(likelihood_map_8bit);
}
