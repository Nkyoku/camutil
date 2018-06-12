#include "stereo.h"
#include "../imageviewgl.h"
#include "../3rdparty/fastguidedfilter.h"
#include <QtWidgets/QGridLayout>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

VideoStereoThread::VideoStereoThread(VideoInput *video_input)
    : VideoThread(video_input)
{
    m_StereoMatching[0].setMaximumDisparity(kMaxDisparity);
    m_StereoMatching[1].setMaximumDisparity(kMaxDisparity);
    m_StereoMatching[2].setMaximumDisparity(kMaxDisparity);
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
    m_Depth = new ImageViewGl;
    grid_layout->addWidget(m_Depth, 0, 2);
    connect(this, &VideoThread::update, m_Depth, QOverload<>::of(&QWidget::update), Qt::QueuedConnection);

    for (int index = 0; index < 3; index++) {
        m_Disparity[index] = new ImageViewGl;
        grid_layout->addWidget(m_Disparity[index], 1, index);
        connect(this, &VideoThread::update, m_Disparity[index], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    }
    m_Disparity[0]->useMouse();
    connect(m_Disparity[0], &ImageViewGl::mouseMoved, this, &VideoStereoThread::watch, Qt::QueuedConnection);

    for (int index = 0; index < 3; index++) {
        m_Likelihood[index] = new ImageViewGl;
        grid_layout->addWidget(m_Likelihood[index], 2, index);
        connect(this, &VideoThread::update, m_Likelihood[index], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    }
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
    int original_width = input_image.cols / 2;
    int original_height = input_image.rows;
    int width = original_width / kHScale1;
    int height = original_height / kVScale1;

    m_Undistort.undistort(cv::Mat(input_image, cv::Rect(0, 0, original_width, original_height)), m_OriginalImage[0], 0);
    m_Undistort.undistort(cv::Mat(input_image, cv::Rect(original_width, 0, original_width, original_height)), m_OriginalImage[1], 1);

    // 縮小する
    for (int side = 0; side < 2; side++) {
        cv::resize(m_OriginalImage[side], m_ColorImage[side], cv::Size(), 1.0 / kHScale1, 1.0 / kVScale1);
        
        //m_ColorImage[side] = fastGuidedFilter(m_ColorImage[side], m_ColorImage[side], 5, 1000, 1);
        
        cv::resize(m_ColorImage[side], m_ColorImage[2 + side], cv::Size(), static_cast<double>(kHScale1) / kHScale2, static_cast<double>(kVScale1) / kVScale2);
        cv::resize(m_ColorImage[2 + side], m_ColorImage[4 + side], cv::Size(), static_cast<double>(kHScale2) / kHScale3, static_cast<double>(kVScale2) / kVScale3);
    }

    // ステレオマッチングを行う
    for (int level = 0; level < 3; level++) {
        m_StereoMatching[level].compute(m_ColorImage[2 * level], m_ColorImage[2 * level + 1], m_DisparityMap[level], m_LikelihoodMap[level]);
    }

    cv::GaussianBlur(m_LikelihoodMap[0], m_LikelihoodMap[0], cv::Size(9, 9), 1.25, 1.25);
    cv::GaussianBlur(m_LikelihoodMap[1], m_LikelihoodMap[1], cv::Size(5, 5), 1.25, 1.25);
    cv::GaussianBlur(m_LikelihoodMap[2], m_LikelihoodMap[2], cv::Size(3, 3), 1.25, 1.25);


    //m_DisparityMap[0] = fastGuidedFilter(m_ColorImage[0], m_DisparityMap[0], 5, 1000, 1);
    //m_DisparityMap[1] = fastGuidedFilter(m_ColorImage[2], m_DisparityMap[1], 5, 1000, 1);
    //m_DisparityMap[2] = fastGuidedFilter(m_ColorImage[4], m_DisparityMap[2], 5, 1000, 1);

    // 偏差マップを拡大する
    cv::resize(m_DisparityMap[1], m_MagnifiedDisparityMap[0], cv::Size(), static_cast<double>(kHScale2) / kHScale1, static_cast<double>(kVScale2) / kVScale1);
    cv::resize(m_DisparityMap[2], m_MagnifiedDisparityMap[1], cv::Size(), static_cast<double>(kHScale3) / kHScale1, static_cast<double>(kVScale3) / kVScale1);

    // 偏差マップを合成する
    double weight1 = 1.0;// / sqrt(kHScale1);
    double weight2 = 1.0;// / sqrt(kHScale2);
    double weight3 = 1.0;// / sqrt(kHScale3);
    m_DepthMap.create(height, width, CV_32F);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double disparity1 = m_DisparityMap[0].at<float>(y, x);
            double disparity2 = m_MagnifiedDisparityMap[0].at<float>(y, x);
            double disparity3 = m_MagnifiedDisparityMap[1].at<float>(y, x);
            double likelihood1 = 0;// weight1 * m_LikelihoodMap[0].at<float>(y, x);
            double likelihood2 = weight2 * m_LikelihoodMap[1].at<float>(y * kVScale1 / kVScale2, x * kHScale1 / kHScale2);
            double likelihood3 = weight3 * m_LikelihoodMap[2].at<float>(y * kVScale1 / kVScale3, x * kHScale1 / kHScale3);
            double total_likelihood = likelihood1 + likelihood2 + likelihood3;
            double result;
            if (total_likelihood != 0.0) {
                result = likelihood1 * disparity1 + likelihood2 * (disparity2 * kHScale2 / kHScale1) + likelihood3 * (disparity3 * kHScale3 / kHScale1);
                result /= total_likelihood;
            } else {
                result = 0.0;
            }
            m_DepthMap.at<float>(y, x) = result / (kMaxDisparity * kHScale3 / kHScale1);
            
            //m_DepthMap.at<float>(y, x) = std::max(disparity1, std::max(disparity2 * kHScale2 / kHScale1, disparity3 * kHScale3 / kHScale1)) / (kMaxDisparity * kHScale3 / kHScale1);

        }
    }

    //double max_val;
    //cv::minMaxLoc(m_DepthMap, nullptr, &max_val);
    //m_DepthMap *= 1.0 / max_val;
    cv::Mat magnified_depth;
    cv::resize(m_DepthMap, magnified_depth, cv::Size(), kHScale1, kVScale1);
   


    //cv::minMaxLoc(m_DisparityMap[0], nullptr, &max_val);
    m_DisparityMap[0] *= 1.0 / kMaxDisparity;
    //cv::minMaxLoc(m_DisparityMap[1], nullptr, &max_val);
    m_DisparityMap[1] *= 1.0 / kMaxDisparity;
    //cv::minMaxLoc(m_DisparityMap[2], nullptr, &max_val);
    m_DisparityMap[2] *= 1.0 / kMaxDisparity;

    

    /*cv::Mat disparity_map1, disparity_map2, disparity_map3;
    m_DisparityMap[0].convertTo(disparity_map1, CV_8U, 255.0);
    m_DisparityMap[1].convertTo(disparity_map2, CV_8U, 255.0);
    m_DisparityMap[2].convertTo(disparity_map3, CV_8U, 255.0);*/

    //cv::Mat inter_depth;
    //m_Interpolation.compute(m_CoarseColorImage[0], m_CoarseDepthMap, inter_depth);

    //cv::Mat inter_depth_8bit;
    //inter_depth.convertTo(inter_depth_8bit, CV_8U, 8.0);


    // 深度マップを表示用に8ビットカラー画像に変換する
    //cv::Mat depth_8bit, depth_scaled_8bit, depth_display;
    //m_CoarseDepthMap.convertTo(depth_8bit, CV_8U, 8.0);
    //cv::resize(depth_8bit, depth_scaled_8bit, m_Depth[0]->optimumImageSize(m_CoarseDepthMap.cols, m_CoarseDepthMap.rows));
    //cv::cvtColor(depth_scaled_8bit, depth_display, cv::COLOR_GRAY2RGB);

    

    /*// 深度マップに情報を表示する
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
            points_input.at<Vec3f>(0)[2] = m_CoarseDepthMap.at<float>(image_y, image_x) * kCoarseRatio;
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
    }*/

    m_Color[0]->setImage(m_OriginalImage[0]);
    m_Color[1]->setImage(m_OriginalImage[1]);
    m_Depth->setImage(magnified_depth);
    m_Disparity[0]->setImage(m_DisparityMap[0]);
    m_Disparity[1]->setImage(m_DisparityMap[1]);
    m_Disparity[2]->setImage(m_DisparityMap[2]);
    m_Likelihood[0]->setImage(m_LikelihoodMap[0]);
    m_Likelihood[1]->setImage(m_LikelihoodMap[1]);
    m_Likelihood[2]->setImage(m_LikelihoodMap[2]);
}
