#include "gradient.h"
#include "imageviewgl.h"
#include <QtWidgets/QGridLayout>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#define _USE_MATH_DEFINES
#include <math.h>

VideoGradientThread::VideoGradientThread(VideoInput *video_input)
    : VideoThread(video_input)
{
    m_Lsd = cv::createLineSegmentDetector();
}

VideoGradientThread::~VideoGradientThread(){
    quitThread();
}

QString VideoGradientThread::initializeOnce(QWidget *parent) {
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

    m_Gradient[0]->useMouse();
    connect(m_Gradient[0], &ImageViewGl::mouseMoved, this, &VideoGradientThread::watch, Qt::QueuedConnection);
}

void VideoGradientThread::uninitialize(void) {
    m_Undistort.destroy();
    m_WatchPointX = -1;
    m_WatchPointY = -1;
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

    cv::Mat gray_left, gray_right;
    cv::cvtColor(m_OriginalImage[0], gray_left, cv::COLOR_BGR2GRAY);
    cv::cvtColor(m_OriginalImage[1], gray_right, cv::COLOR_BGR2GRAY);

    cv::Mat disparity_map;
    m_StereoMatching.precompute(gray_left, gray_right);
    
    m_StereoMatching.compute(disparity_map, 32);
    cv::Mat disparity_map_8bit, disparity_map_8bit_color;
    disparity_map.convertTo(disparity_map_8bit, CV_8U, 8.0);

    cv::Mat gradient_l_x, gradient_r_x;
    cv::cvtColor(m_StereoMatching.m_GaussianDoG[0]->gradient0deg(), gradient_l_x, cv::COLOR_GRAY2RGB);
    cv::cvtColor(m_StereoMatching.m_GaussianDoG[1]->gradient0deg(), gradient_r_x, cv::COLOR_GRAY2RGB);

    m_Color[0]->setImage(gray_left);
    m_Color[1]->setImage(disparity_map_8bit);
    m_Gradient[0]->setImage(gradient_l_x);
    m_Gradient[1]->setImage(gradient_r_x);//*/

    
    /*
    cv::Mat gray;
    cv::cvtColor(m_OriginalImage[0], gray, cv::COLOR_BGR2GRAY);
    m_GaussianDoG.compute(gray);

    /*cv::Mat dominant_0(height, width, CV_8U);
    cv::Mat dominant_45(height, width, CV_8U);
    cv::Mat dominant_90(height, width, CV_8U);
    cv::Mat dominant_135(height, width, CV_8U);
    for (int y = 0; y < height; y++) {
        uint8_t *dominant_0_ptr = dominant_0.ptr(y);
        uint8_t *dominant_45_ptr = dominant_45.ptr(y);
        uint8_t *dominant_90_ptr = dominant_90.ptr(y);
        uint8_t *dominant_135_ptr = dominant_135.ptr(y);
        const uint8_t *dog_0_ptr = m_GaussianDoG.dog0deg().ptr(y);
        const uint8_t *dog_45_ptr = m_GaussianDoG.dog45deg().ptr(y);
        const uint8_t *dog_90_ptr = m_GaussianDoG.dog90deg().ptr(y);
        const uint8_t *dog_135_ptr = m_GaussianDoG.dog135deg().ptr(y);
        for (int x = 0; x < width; x++) {
            int max_value = 0;
            max_value = std::max(max_value, abs(dog_0_ptr[x] - 128));
            max_value = std::max(max_value, abs(dog_45_ptr[x] - 128));
            max_value = std::max(max_value, abs(dog_90_ptr[x] - 128));
            max_value = std::max(max_value, abs(dog_135_ptr[x] - 128));
            if (max_value < 2) {
                max_value = -1;
            }
            dominant_0_ptr[x] = (abs(dog_0_ptr[x] - 128) == max_value) ? 255 : 0;
            dominant_45_ptr[x] = (abs(dog_45_ptr[x] - 128) == max_value) ? 255 : 0;
            dominant_90_ptr[x] = (abs(dog_90_ptr[x] - 128) == max_value) ? 255 : 0;
            dominant_135_ptr[x] = (abs(dog_135_ptr[x] - 128) == max_value) ? 255 : 0;
        }
    }

    m_Color[0]->setImage(dominant_0);
    m_Color[1]->setImage(dominant_45);
    m_Gradient[0]->setImage(dominant_90);
    m_Gradient[1]->setImage(dominant_135);
    //*/

    /*
    m_Color[0]->setImage(m_GaussianDoG.dog0deg());
    m_Color[1]->setImage(m_GaussianDoG.dog90deg());
    m_Gradient[0]->setImage(m_GaussianDoG.dogNarrow0deg());
    m_Gradient[1]->setImage(m_GaussianDoG.dogNarrow90deg());
    //*/
}
