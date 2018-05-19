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
    
};

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

    // 低解像度マップを生成する
    cv::resize(m_ColorImage[0], m_CoarseColorImage[0], cv::Size(coarse_width, coarse_height));
    cv::resize(m_ColorImage[1], m_CoarseColorImage[1], cv::Size(coarse_width, coarse_height));
    cv::cvtColor(m_CoarseColorImage[0], m_CoarseGrayImage[0], cv::COLOR_BGR2GRAY);
    cv::cvtColor(m_CoarseColorImage[1], m_CoarseGrayImage[1], cv::COLOR_BGR2GRAY);
    stereoMatching();





    // 低解像度のデプスマップを標準解像度にもどし、JointWMFによって補間する
    //cv::resize(m_CoarseDepthMap, m_NormalDepthMap, cv::Size(), kCoarseRatio, kCoarseRatio);
    
    


    /*cv::Mat magnified_disparity_l, filtered_disparity_l;
    cv::resize(m_CoarseDepthMap, magnified_disparity_l, cv::Size(), kCoarseRatio, kCoarseRatio);
    filtered_disparity_l = JointWMF::filter(magnified_disparity_l, cv::Mat(m_ColorImage[0], cv::Rect(m_MaxDisparity, 0, width - m_MaxDisparity, height)), 5);

    cv::Mat disparity8_l, filtered_disparity8_l;
    m_CoarseDepthMap.convertTo(disparity8_l, CV_8U, 1, 0);
    filtered_disparity_l.convertTo(filtered_disparity8_l, CV_8U, 1, 0);

    cv::Mat disparity8_org_l, filtered_disparity8_org_l;
    cv::resize(disparity8_l, disparity8_org_l, cv::Size(), kCoarseRatio, kCoarseRatio, INTER_LANCZOS4);
    cv::resize(filtered_disparity8_l, filtered_disparity8_org_l, cv::Size(), kCoarseRatio, kCoarseRatio, INTER_LANCZOS4);*/

    cv::Mat depth_scaled, depth_scaled_color;
    cv::resize(m_CoarseDepthMap, depth_scaled, m_Depth[0]->optimumImageSize(coarse_width - m_MaxDisparity, coarse_height));
    cv::cvtColor(depth_scaled, depth_scaled_color, cv::COLOR_GRAY2RGB);
    cv::circle(depth_scaled_color, cv::Point(m_WatchPointX, m_WatchPointY), 5, cv::Scalar(255, 0, 0), 2);
    
    m_Color[0]->setImage(m_ColorImage[0]);
    m_Color[1]->setImage(m_ColorImage[1]);
    m_Depth[0]->setImage(depth_scaled_color);
    //m_Depth[1]->setImage(filtered_disparity8_org_l);
}

void VideoStereoThread::findDisparity(const std::vector<cv::Mat> &cost_volume, cv::Mat &disparity, double scale) {
    if (cost_volume.empty() == true) {
        return;
    }
    int max_disparity = static_cast<int>(cost_volume.size());
    int width = cost_volume[0].cols;
    int height = cost_volume[0].rows;
    disparity.create(height, width, CV_8UC1);
    for (int y = 0; y < height; y++) {
        uint8_t *disp_ptr = disparity.ptr(y);
        for (int x = 0; x < width; x++) {
            int min_p_d = -1;
            double min_p = 65535;
            for (int d = 0; d < max_disparity; d++) {
                double p = cost_volume[d].at<float>(y, x);
                if (p < min_p) {
                    min_p_d = d;
                    min_p = p;
                }
            }
            disp_ptr[x] = static_cast<int>(round(min_p_d * scale));
        }
    }
}

void VideoStereoThread::findDisparitySubPixel(const std::vector<cv::Mat> &cost_volume, cv::Mat &disparity, double scale) {
    if (cost_volume.empty() == true) {
        return;
    }
    int max_disparity = static_cast<int>(cost_volume.size());
    int width = cost_volume[0].cols;
    int height = cost_volume[0].rows;
    std::vector<double> p_list(max_disparity);
    disparity.create(height, width, CV_8UC1);
    for (int y = 0; y < height; y++) {
        uint8_t *disp_ptr = disparity.ptr(y);
        for (int x = 0; x < width; x++) {
            int d_min = 0;
            double p_min = cost_volume[0].at<float>(y, x);
            p_list[0] = p_min;
            for (int d = 1; d < max_disparity; d++) {
                double p = cost_volume[d].at<float>(y, x);
                p_list[d] = p;
                if (p < p_min) {
                    d_min = d;
                    p_min = p;
                }
            }
            double d_frac = d_min;
            if ((d_min != 0) && (d_min != (max_disparity - 1))) {
                double p_prev = p_list[d_min - 1];
                double p_next = p_list[d_min + 1];
                if (p_next <= p_prev) {
                    if (d_min < (max_disparity - 2)) {
                        double p_next2 = p_list[d_min + 2];
                        d_frac += (p_prev - p_next) / (p_prev - p_min - p_next + p_next2);
                    } else {
                        d_frac += (p_prev - p_next) / (p_prev - p_min) * 0.5;
                    }
                } else {
                    if (1 < d_min) {
                        double p_prev2 = p_list[d_min - 2];
                        d_frac += (p_prev - p_next) / (p_prev2 - p_prev - p_min + p_next);
                    } else {
                        d_frac += (p_prev - p_next) / (p_next - p_min) * 0.5;
                    }
                }
            }
            disp_ptr[x] = std::max(std::min(static_cast<int>(round(d_frac * scale)), 255), 0);
        }
    }
}

void VideoStereoThread::stereoMatching(void) {
    static const int kGifRadius = 5;
    static const double kGifEps = 1000;
    static const int kTc = 7;
    static const int kTg = 4;
    static const double kAlpha = 0.2;

    int width = m_CoarseColorImage[0].cols;
    int height = m_CoarseColorImage[0].rows;
    
    // Census変換を行う
    m_CoarseCensusImage[0].create(width, height, CV_32SC1);
    m_CoarseCensusImage[1].create(width, height, CV_32SC1);
    doCensus5x5Transform(m_CoarseGrayImage[0], m_CoarseCensusImage[0]);
    doCensus5x5Transform(m_CoarseGrayImage[1], m_CoarseCensusImage[1]);

    // GuidedFilterを作成する
    cv::Mat guide(m_CoarseGrayImage[0], cv::Rect(m_MaxDisparity, 0, width - m_MaxDisparity, height));
    FastGuidedFilter filter(guide, kGifRadius, kGifEps, 1);

    // コストボリュームを生成する
    m_CostVolume.resize(m_MaxDisparity);
#pragma omp parallel for
    for (int d = 0; d < m_MaxDisparity; d++) {
        cv::Mat cost(height, width - m_MaxDisparity, CV_32FC1);
        for (int y = 0; y < height; y++) {
            float *cost_ptr = cost.ptr<float>(y);
            for (int x = m_MaxDisparity; x < width; x++) {
                const uint8_t *rgb_l_ptr = m_CoarseColorImage[0].ptr(y);
                const uint8_t *rgb_r_ptr = m_CoarseColorImage[1].ptr(y);
                const uint32_t *census_l_ptr = m_CoarseCensusImage[0].ptr<uint32_t>(y);
                const uint32_t *census_r_ptr = m_CoarseCensusImage[1].ptr<uint32_t>(y);
                int color_diff =
                    abs(rgb_l_ptr[3 * x] - rgb_r_ptr[3 * (x - d)])
                    + abs(rgb_l_ptr[3 * x + 1] - rgb_r_ptr[3 * (x - d) + 1])
                    + abs(rgb_l_ptr[3 * x + 2] - rgb_r_ptr[3 * (x - d) + 2]);
                int grad_diff =
                    __popcnt(census_l_ptr[x] ^ census_r_ptr[x - d]);
                *cost_ptr++ = static_cast<float>(kAlpha * std::min(kTc * 16, color_diff) + (1 - kAlpha) * std::min(kTg * 16, grad_diff));
            }
        }
        m_CostVolume[d] = filter.filter(cost);
    }

    // 視差を計算して返す
    findDisparitySubPixel(m_CostVolume, m_CoarseDepthMap, 8);
}

void VideoStereoThread::doCensus5x5Transform(const cv::Mat &src, cv::Mat &dst) {
    int width = src.cols;
    int height = src.rows;
    dst.create(height, width, CV_32SC1);
    for (int y00 = 0; y00 < height; y00++) {
        int y2n = std::max(y00 - 2, 0);
        int y1n = std::max(y00 - 1, 0);
        int y1p = std::min(y00 + 1, height - 1);
        int y2p = std::min(y00 + 2, height - 1);
        int x2n = 0, x1n = 0, x1p = 0, x2p = 0;
        for (int x00 = 0; x00 < width; x00++) {
            x2n = std::max(x00 - 2, 0);
            x1n = std::max(x00 - 1, 0);
            x1p = std::min(x00 + 1, width - 1);
            x2p = std::min(x00 + 2, width - 1);
            int bitmap = 0;
            uint8_t center = src.at<uint8_t>(y00, x00);
            bitmap |= (center < src.at<uint8_t>(y2n, x2n)) ? 0x1 : 0;
            bitmap |= (center < src.at<uint8_t>(y2n, x1n)) ? 0x2 : 0;
            bitmap |= (center < src.at<uint8_t>(y2n, x00)) ? 0x4 : 0;
            bitmap |= (center < src.at<uint8_t>(y2n, x1p)) ? 0x8 : 0;
            bitmap |= (center < src.at<uint8_t>(y2n, x2p)) ? 0x10 : 0;
            bitmap |= (center < src.at<uint8_t>(y1n, x2n)) ? 0x20 : 0;
            bitmap |= (center < src.at<uint8_t>(y1n, x1n)) ? 0x40 : 0;
            bitmap |= (center < src.at<uint8_t>(y1n, x00)) ? 0x80 : 0;
            bitmap |= (center < src.at<uint8_t>(y1n, x1p)) ? 0x100 : 0;
            bitmap |= (center < src.at<uint8_t>(y1n, x2p)) ? 0x200 : 0;
            bitmap |= (center < src.at<uint8_t>(y00, x2n)) ? 0x400 : 0;
            bitmap |= (center < src.at<uint8_t>(y00, x1n)) ? 0x800 : 0;
            bitmap |= (center < src.at<uint8_t>(y00, x1p)) ? 0x1000 : 0;
            bitmap |= (center < src.at<uint8_t>(y00, x2p)) ? 0x2000 : 0;
            bitmap |= (center < src.at<uint8_t>(y1p, x2n)) ? 0x4000 : 0;
            bitmap |= (center < src.at<uint8_t>(y1p, x1n)) ? 0x8000 : 0;
            bitmap |= (center < src.at<uint8_t>(y1p, x00)) ? 0x10000 : 0;
            bitmap |= (center < src.at<uint8_t>(y1p, x1p)) ? 0x20000 : 0;
            bitmap |= (center < src.at<uint8_t>(y1p, x2p)) ? 0x40000 : 0;
            bitmap |= (center < src.at<uint8_t>(y2p, x2n)) ? 0x80000 : 0;
            bitmap |= (center < src.at<uint8_t>(y2p, x1n)) ? 0x100000 : 0;
            bitmap |= (center < src.at<uint8_t>(y2p, x00)) ? 0x200000 : 0;
            bitmap |= (center < src.at<uint8_t>(y2p, x1p)) ? 0x400000 : 0;
            bitmap |= (center < src.at<uint8_t>(y2p, x2p)) ? 0x800000 : 0;
            dst.at<int>(y00, x00) = bitmap;
        }
    }
}

void VideoStereoThread::watch(int x, int y) {
    m_WatchPointX = x;
    m_WatchPointY = y;
}
