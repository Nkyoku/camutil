#include "stereo.h"
#include "../imageviewgl.h"
#include "../3rdparty/fastguidedfilter.h"
#include "../3rdparty/JointWMF.h"
#include <QtWidgets/QGridLayout>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

VideoStereoThread::VideoStereoThread(VideoInput *video_input)
    : VideoThread(video_input)
{
    m_StereoMatching.setMaximumDisparity(m_MaxDisparity);
}

VideoStereoThread::~VideoStereoThread(){
    quitThread();
}

QString VideoStereoThread::initializeOnce(QWidget *parent) {
    return tr("Stereo");
}

void VideoStereoThread::initialize(QWidget *parent) {
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
    m_Depth[0] = new ImageViewGl;
    m_Depth[0]->useMouse();
    grid_layout->addWidget(m_Depth[0], 1, 0);
    connect(this, &VideoThread::update, m_Depth[0], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    connect(m_Depth[0], &ImageViewGl::mouseMoved, this, &VideoStereoThread::watch, Qt::QueuedConnection);
    m_Depth[1] = new ImageViewGl;
    grid_layout->addWidget(m_Depth[1], 1, 1);
    connect(this, &VideoThread::update, m_Depth[1], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
}

void VideoStereoThread::uninitialize(void) {
    m_Undistort.destroy();
    m_WatchPointX = -1;
    m_WatchPointY = -1;
}

void VideoStereoThread::restoreSettings(const QSettings &settings) {
    
}

void VideoStereoThread::saveSettings(QSettings &settings) const {
    
}

void VideoStereoThread::processImage(const cv::Mat &input_image) {
    int width = input_image.cols / 2;
    int height = input_image.rows;
    int coarse_width = width / kCoarseRatio;
    int coarse_height = height / kCoarseRatio;

    m_Undistort.undistort(cv::Mat(input_image, cv::Rect(0, 0, width, height)), m_ColorImage[0], 0);
    m_Undistort.undistort(cv::Mat(input_image, cv::Rect(width, 0, width, height)), m_ColorImage[1], 1);

    // ステレオマッチングを行う
    // reprojectPointsTo3Dは偏差の4倍の値を受け取るのでスケールする
    cv::resize(m_ColorImage[0], m_CoarseColorImage[0], cv::Size(coarse_width, coarse_height));
    cv::resize(m_ColorImage[1], m_CoarseColorImage[1], cv::Size(coarse_width, coarse_height));
    m_StereoMatching.compute(m_CoarseColorImage[0], m_CoarseColorImage[1], m_CoarseDepthMap);
    m_CoarseDepthMap *= 4;

    // 深度マップを表示用に8ビットカラー画像に変換する
    cv::Mat depth_8bit, depth_scaled_8bit, depth_display;
    m_CoarseDepthMap.convertTo(depth_8bit, CV_8U, 2.0);
    cv::resize(depth_8bit, depth_scaled_8bit, m_Depth[0]->optimumImageSize(m_CoarseDepthMap.cols, m_CoarseDepthMap.rows));
    cv::cvtColor(depth_scaled_8bit, depth_display, cv::COLOR_GRAY2RGB);

    // 深度マップに情報を表示する
    cv::circle(depth_display, cv::Point(m_WatchPointX, m_WatchPointY), 5, cv::Scalar(255, 0, 0), 2);
    {
        double image_x = static_cast<double>(m_WatchPointX) * m_CoarseDepthMap.cols / depth_display.cols;
        double image_y = static_cast<double>(m_WatchPointY) * m_CoarseDepthMap.rows / depth_display.rows;
        double actual_x = image_x * kCoarseRatio;
        double actual_y = image_y * kCoarseRatio;
        if ((0.0 <= image_x) && (0.0 <= image_y) && (image_x < coarse_width) && (image_y < coarse_height)) {
            cv::Mat points_input(1, 1, CV_32FC3), points_output;
            points_input.at<Vec3f>(0)[0] = actual_x;
            points_input.at<Vec3f>(0)[1] = actual_y;
            points_input.at<Vec3f>(0)[2] = m_CoarseDepthMap.at<float>(image_y, image_x);
            //double likelihood = m_CoarseLikelihoodMap.at<float>(image_y, image_x);
            m_Undistort.reprojectPointsTo3D(points_input, points_output);
            if (points_output.rows == 1) {
                Vec3f point = points_output.at<Vec3f>(0);
                std::ostringstream text1, text2;
                text1 << "(" << point[0] << ", " << point[1] << ", " << point[2] << ")";
                //text2 << "l=" << likelihood;
                //text << "(" << actual_x << ", " << actual_y << ", " << m_CoarseDepthMap.at<float>(image_y, image_x) << ")";
                cv::putText(depth_display, text1.str(), cv::Point(4, 16), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
                //cv::putText(depth_display, text2.str(), cv::Point(4, 32), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
            }
        }
    }

    m_Color[0]->setImage(m_ColorImage[0]);
    m_Color[1]->setImage(m_ColorImage[1]);
    m_Depth[0]->setImage(depth_display);
    //m_Depth[1]->setImage(likelihood_8bit);
    return;
}
