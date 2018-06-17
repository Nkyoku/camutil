#include "fielddetector.h"
#include "../imageviewgl.h"
#include <QtWidgets/QGridLayout>
#define _USE_MATH_DEFINES
#include <math.h>
#include <new>
#include <sstream>
//#include <QDebug>

VideoFieldDetectorThread::VideoFieldDetectorThread(VideoInput *video_input)
    : VideoThread(video_input)
{
    new(&m_EnhancementFilter[0]) EnhancementFilter(kScaleFactor, 5);
    new(&m_EnhancementFilter[1]) EnhancementFilter(kScaleFactor, 5);
}

VideoFieldDetectorThread::~VideoFieldDetectorThread(){
    quitThread();
}

QString VideoFieldDetectorThread::initializeOnce(QWidget *parent) {
    QGridLayout *grid_layout = new QGridLayout;
    parent->setLayout(grid_layout);
    m_VanishSpinbox = new QSpinBox;
    m_VanishSpinbox->setMinimum(-10000);
    m_VanishSpinbox->setMaximum(10000);
    m_VanishSpinbox->setSingleStep(10);
    m_VanishSpinbox->setValue(0);
    grid_layout->addWidget(m_VanishSpinbox);
    return tr("FieldDetector");
}

void VideoFieldDetectorThread::initialize(QWidget *parent) {
    uninitialize();

    QSize size = m_VideoInput->sourceResolution();
    int width = size.width() / 2;
    int height = size.height();
    m_Undistort.load(width, height);

    // ImageViewGlを生成
    QGridLayout *grid_layout = new QGridLayout;
    parent->setLayout(grid_layout);
    for (int index = 0; index < 1; index++) {
        m_Output[index] = new ImageViewGl;
        grid_layout->addWidget(m_Output[index], index / 2, index % 2);
        connect(this, &VideoThread::update, m_Output[index], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    }

    //m_Output[0]->convertBgrToRgb();
    //m_Output[1]->convertBgrToRgb();

    //m_Output[1]->useMouse();
    //connect(m_Output[1], &ImageViewGl::mouseMoved, this, &VideoFieldDetectorThread::showColor, Qt::QueuedConnection);
}

void VideoFieldDetectorThread::uninitialize(void) {
    m_Undistort.destroy();
    m_WatchPointX = -1;
    m_WatchPointY = -1;
}

void VideoFieldDetectorThread::restoreSettings(const QSettings &settings) {
    
}

void VideoFieldDetectorThread::saveSettings(QSettings &settings) const {
    
}

void VideoFieldDetectorThread::processImage(const cv::Mat &input_image) {
    int width = input_image.cols / 2;
    int height = input_image.rows;
    
    m_Undistort.undistort(cv::Mat(input_image, cv::Rect(0, 0, width, height)), m_ColorImage[0], 0);
    m_Undistort.undistort(cv::Mat(input_image, cv::Rect(width, 0, width, height)), m_ColorImage[1], 1);

    /*// a*, b*成分の2次元ヒストグラムを作成する
    cv::Mat histogram(256, 256, CV_32SC1, cv::Scalar(0));
    lab.forEach<cv::Vec3b>([&](const cv::Vec3b &p, const int pos[2]) {
    histogram.at<int>(p[2], p[1])++;
    });*/

    // ヒストグラムを表示用に加工する
    /*cv::Mat histogram_show(256, 256, CV_8UC3);
    double min, max;
    cv::minMaxLoc(histogram, &min, &max);
    histogram.forEach<int>([&](int value, const int pos[2]) {
    uint8_t color = static_cast<int>(std::min(-log10(value / max) * 100.0, 255.0));
    histogram_show.at<cv::Vec3b>(pos[0], pos[1]) = cv::Vec3b(color, color, color);
    });*/

    cv::Mat white_display(height, width, CV_8UC3);
    white_display.setTo(0);
    cv::Mat green_display(height, width, CV_8UC3);
    std::vector<cv::Range> ranges;
    std::vector<cv::Vec4f> line_segments[2];
#pragma omp parallel for
    for (int side = 0; side < 1; side++) {
        // 画像に強調フィルタを掛けL*a*b*に変換する
        m_EnhancementFilter[side].compute(m_ColorImage[side], m_EnhancedColorImage[side]);
        cv::cvtColor(m_EnhancedColorImage[side], m_LabEnhancedImage[side], cv::COLOR_BGR2Lab);
        cv::cvtColor(m_EnhancementFilter[side].medianedImage(), m_LabMedianedImage[side], cv::COLOR_BGR2Lab);

        // 芝の検知を行う
        // 強調フィルタの処理過程で出た中間値画像を利用して白線を除く
        //std::vector<cv::Point2d> grass_rect;
        const cv::Mat &green = m_FieldDetector[side].detectGrassMoment(m_LabMedianedImage[side], &ranges, kScaleFactor);
        if (side == 0) {
            //cv::cvtColor(green, green_display, cv::COLOR_GRAY2RGB);
            for (int y = 0; y < green_display.rows; y++) {
                int start_x = ranges[y].start;
                int end_x = ranges[y].end;
                if (start_x != end_x) {
                    for (int x = 0; x < green_display.cols; x++) {
                        bool grass = green.at<uint8_t>(y / kScaleFactor, x / kScaleFactor);
                        bool inside = (start_x <= x) && (x < end_x);
                        if (grass == true) {
                            if (inside == true) {
                                green_display.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
                            } else {
                                green_display.at<cv::Vec3b>(y, x) = cv::Vec3b(128, 128, 128);
                            }
                        } else {
                            if (inside == true) {
                                green_display.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 128, 0);
                            } else {
                                green_display.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                            }
                        }
                    }
                } else {
                    memset(green_display.ptr(y), 0, 3 * width);
                }
            }

            /*cv::line(green_display, grass_rect[0] / kScaleFactor, grass_rect[1] / kScaleFactor, cv::Scalar(0, 128, 255));
            cv::line(green_display, grass_rect[1] / kScaleFactor, grass_rect[2] / kScaleFactor, cv::Scalar(0, 128, 255));
            cv::line(green_display, grass_rect[2] / kScaleFactor, grass_rect[3] / kScaleFactor, cv::Scalar(0, 128, 255));
            cv::line(green_display, grass_rect[3] / kScaleFactor, grass_rect[0] / kScaleFactor, cv::Scalar(0, 128, 255));*/
        }
        

        // 白線の検知を行う
        std::vector<cv::Vec4f> edge_line_segments;
        const cv::Mat &white = m_FieldDetector[side].detectLines(m_LabEnhancedImage[side], line_segments[side], &edge_line_segments);

        // 表示用画像に白線の線分を描画する
        //cv::cvtColor(white, white_display[side], cv::COLOR_GRAY2RGB);
        //cv::line(white_display, grass_rect[0], grass_rect[1], cv::Scalar(0, 255, 0));
        //cv::line(white_display, grass_rect[1], grass_rect[2], cv::Scalar(0, 255, 0));
        //cv::line(white_display, grass_rect[2], grass_rect[3], cv::Scalar(0, 255, 0));
        //cv::line(white_display, grass_rect[3], grass_rect[0], cv::Scalar(0, 255, 0));
        //for (const cv::Vec4f &segment : edge_line_segments) {
        //    cv::Point a(segment[0], segment[1]);
        //    cv::Point b(segment[2], segment[3]);
        //    cv::line(white_display[side], a, b, cv::Scalar(0, 128, 255), 2);
        //}
        for (const cv::Vec4f &segment : line_segments[side]) {
            cv::Point2d a(segment[0], segment[1]);
            cv::Point2d b(segment[2], segment[3]);
            cv::Point2d vector(b - a);
            //a -= vector;
            //b += vector;
            cv::line(white_display, a, b, (side == 0) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 128, 255), 2);
        }

        // 表示用画像に芝の領域を示す矩形を描画する
        /*cv::Mat green_show;
        cv::cvtColor(green, green_show, cv::COLOR_GRAY2RGB);
        cv::line(green_show, grass_rect[0] / kScaleFactor, grass_rect[1] / kScaleFactor, cv::Scalar(0, 255, 0));
        cv::line(green_show, grass_rect[1] / kScaleFactor, grass_rect[2] / kScaleFactor, cv::Scalar(0, 255, 0));
        cv::line(green_show, grass_rect[2] / kScaleFactor, grass_rect[3] / kScaleFactor, cv::Scalar(0, 255, 0));
        cv::line(green_show, grass_rect[3] / kScaleFactor, grass_rect[0] / kScaleFactor, cv::Scalar(0, 255, 0));*/
    }

    /*// ポイントされた場所の色をヒストグラムに表示する
    if ((0 <= m_WatchPointX) && (0 <= m_WatchPointY) && (m_WatchPointX < medianed.cols) && (m_WatchPointY < medianed.rows)) {
    cv::Vec3b color = lab_small.at<cv::Vec3b>(m_WatchPointY, m_WatchPointX);
    int L = color[0], a = color[1], b = color[2];
    cv::circle(histogram_show, cv::Point(a, b), 3, cv::Scalar(255, 0, 0));
    cv::circle(medianed, cv::Point(m_WatchPointX, m_WatchPointY), 1, cv::Scalar(0, 0, 255));
    cv::circle(green_show, cv::Point(m_WatchPointX, m_WatchPointY), 1, cv::Scalar(255, 0, 0));
    cv::circle(white_show, cv::Point(m_WatchPointX * kScaleFactor, m_WatchPointY * kScaleFactor), kScaleFactor, cv::Scalar(255, 0, 0), kScaleFactor);
    }*/

    // フィールドを描画する
    /*cv::Mat field_2d(800, 1100, CV_8UC3);
    field_2d.setTo(0);
    for (const cv::Vec4f &segment : PositionTracker::kLineTemplate) {
        int x1 = static_cast<int>(round(segment[0] * 100 + field_2d.cols * 0.5));
        int y1 = static_cast<int>(round(segment[1] * 100 + field_2d.rows * 0.5));
        int x2 = static_cast<int>(round(segment[2] * 100 + field_2d.cols * 0.5));
        int y2 = static_cast<int>(round(segment[3] * 100 + field_2d.rows * 0.5));
        int thickness = static_cast<int>(round(PositionTracker::kThickness1 * 100));
        cv::line(field_2d, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 255), thickness);
    }*/


    //cv::line(white_display[0], cv::Point(0, height - 1), cv::Point(width / 2, m_VanishSpinbox->value()), cv::Scalar(0, 128, 255), 2);
    //cv::line(white_display[0], cv::Point(width - 1, height - 1), cv::Point(width / 2, m_VanishSpinbox->value()), cv::Scalar(0, 128, 255), 2);


    //cv::Mat estimated;
    //m_PositionTracker.estimate(line_segments[0], line_segments[1], m_Undistort, estimated, width, height, 0);
    

    //m_Output[0]->setImage(m_ColorImage[0]);
    //m_Output[1]->setImage(m_ColorImage[1]);
    m_Output[0]->setImage(white_display);
    //m_Output[0]->setImage(estimated);
}

void VideoFieldDetectorThread::showColor(int x, int y) {
    m_WatchPointX = x;
    m_WatchPointY = y;
}
