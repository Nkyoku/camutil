#include "fielddetector.h"
#include "../imageviewgl.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <QtWidgets/QGridLayout>
#include <sstream>
#include <QDebug>

VideoFieldDetectorThread::VideoFieldDetectorThread(VideoInput *video_input)
    : VideoThread(video_input), m_EnhancementFilter(kScaleFactor, 5)
{
    //m_Lsd = cv::createLineSegmentDetector();
}

VideoFieldDetectorThread::~VideoFieldDetectorThread(){
    quitThread();
}

QString VideoFieldDetectorThread::initializeOnce(QWidget *parent) {
    QGridLayout *layout = new QGridLayout;
    parent->setLayout(layout);
    m_ThresholdInput = new QDoubleSpinBox;
    m_ThresholdInput->setMinimum(-180.0);
    m_ThresholdInput->setMaximum(180.0);
    m_ThresholdInput->setValue(120.0);
    layout->addWidget(m_ThresholdInput);
    m_Threshold2Input = new QDoubleSpinBox;
    m_Threshold2Input->setMinimum(-180.0);
    m_Threshold2Input->setMaximum(180.0);
    m_Threshold2Input->setValue(-150.0);
    layout->addWidget(m_Threshold2Input);
    m_Threshold3Input = new QDoubleSpinBox;
    m_Threshold3Input->setMinimum(0.0);
    m_Threshold3Input->setMaximum(180.0);
    m_Threshold3Input->setValue(10.0);
    layout->addWidget(m_Threshold3Input);
    return tr("FieldDetector");
}

void VideoFieldDetectorThread::initialize(QWidget *parent) {
    uninitialize();

    QSize size = m_VideoInput->sourceResolution();
    int width = size.width() / 2;
    int height = size.height();
    m_Undistort.load(width, height);

    // カラー画像と深度画像を表示するImageViewGlを生成
    QGridLayout *grid_layout = new QGridLayout;
    parent->setLayout(grid_layout);
    for (int side = 0; side < 2; side++) {
        //m_Color[side] = new ImageViewGl;
        //m_Color[side]->convertBgrToRgb();
        //grid_layout->addWidget(m_Color[side], 0, side);
        //connect(this, &VideoThread::update, m_Color[side], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
        
    }
    for (int side = 0; side < 1; side++) {
        m_Field[side] = new ImageViewGl;
        grid_layout->addWidget(m_Field[side], 1, side);
        connect(this, &VideoThread::update, m_Field[side], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    }

    //m_Color[1]->useMouse();
    //connect(m_Color[1], &ImageViewGl::mouseMoved, this, &VideoFieldDetectorThread::showColor, Qt::QueuedConnection);
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

    m_EnhancementFilter.compute(m_ColorImage[0], m_EnhancedColorImage[0]);
    const cv::Mat &medianed = m_EnhancementFilter.m_Medianed;



    cv::Mat lab;
    cv::cvtColor(m_EnhancedColorImage[0], lab, cv::COLOR_BGR2Lab);

    cv::Mat lab_small;
    cv::cvtColor(medianed, lab_small, cv::COLOR_BGR2Lab);

    // 芝の検知を行う
    std::vector<cv::Point2d> grass_rect;
    cv::Mat &green = m_FieldDetector.detectGrass(lab_small, kScaleFactor, &grass_rect);

    // 白線の検知を行う
    std::vector<cv::Vec4f> line_segments;
    cv::Mat &white = m_FieldDetector.detectLines(lab, &line_segments);
    












    // a*, b*成分の2次元ヒストグラムを作成する
    cv::Mat histogram(256, 256, CV_32SC1, cv::Scalar(0));
    lab.forEach<cv::Vec3b>([&](const cv::Vec3b &p, const int pos[2]) {
        histogram.at<int>(p[2], p[1])++;
    });

    // ヒストグラムを表示用に加工する
    cv::Mat histogram_show(256, 256, CV_8UC3);
    double min, max;
    cv::minMaxLoc(histogram, &min, &max);
    histogram.forEach<int>([&](int value, const int pos[2]) {
        uint8_t color = static_cast<int>(std::min(-log10(value / max) * 100.0, 255.0));
        histogram_show.at<cv::Vec3b>(pos[0], pos[1]) = cv::Vec3b(color, color, color);
    });

    


    

    // ヒストグラムに緑の領域を表示する
    //cv::ellipse(histogram_show, cv::Point(128, 128), cv::Size(green_region.chroma_min, green_region.chroma_min), 0.0, green_region.hue_min * 180.0 / M_PI, green_region.hue_max * 180.0 / M_PI, cv::Scalar(0, 192, 0));
    //cv::ellipse(histogram_show, cv::Point(128, 128), cv::Size(green_region.chroma_max, green_region.chroma_max), 0.0, green_region.hue_min * 180.0 / M_PI, green_region.hue_max * 180.0 / M_PI, cv::Scalar(0, 192, 0));
    //cv::line(histogram_show, cv::Point(128 + green_region.chroma_min * cos(green_region.hue_min), 128 + green_region.chroma_min * sin(green_region.hue_min)), cv::Point(128 + green_region.chroma_max * cos(green_region.hue_min), 128 + green_region.chroma_max * sin(green_region.hue_min)), cv::Scalar(0, 192, 0));
    //cv::line(histogram_show, cv::Point(128 + green_region.chroma_min * cos(green_region.hue_max), 128 + green_region.chroma_min * sin(green_region.hue_max)), cv::Point(128 + green_region.chroma_max * cos(green_region.hue_max), 128 + green_region.chroma_max * sin(green_region.hue_max)), cv::Scalar(0, 192, 0));

    // ポイントされた場所の色をヒストグラムに表示する
    cv::Mat green_show, white_show;
    cv::cvtColor(green, green_show, cv::COLOR_GRAY2RGB);
    cv::cvtColor(white, white_show, cv::COLOR_GRAY2RGB);
    if ((0 <= m_WatchPointX) && (0 <= m_WatchPointY) && (m_WatchPointX < medianed.cols) && (m_WatchPointY < medianed.rows)) {
        cv::Vec3b color = lab_small.at<cv::Vec3b>(m_WatchPointY, m_WatchPointX);
        int L = color[0], a = color[1], b = color[2];
        cv::circle(histogram_show, cv::Point(a, b), 3, cv::Scalar(255, 0, 0));
        cv::circle(medianed, cv::Point(m_WatchPointX, m_WatchPointY), 1, cv::Scalar(0, 0, 255));
        cv::circle(green_show, cv::Point(m_WatchPointX, m_WatchPointY), 1, cv::Scalar(255, 0, 0));
        cv::circle(white_show, cv::Point(m_WatchPointX * kScaleFactor, m_WatchPointY * kScaleFactor), kScaleFactor, cv::Scalar(255, 0, 0), kScaleFactor);
    }


    // 白線の線分を描画する
    cv::line(white_show, grass_rect[0], grass_rect[1], cv::Scalar(0, 255, 0));
    cv::line(white_show, grass_rect[1], grass_rect[2], cv::Scalar(0, 255, 0));
    cv::line(white_show, grass_rect[2], grass_rect[3], cv::Scalar(0, 255, 0));
    cv::line(white_show, grass_rect[3], grass_rect[0], cv::Scalar(0, 255, 0));
    for (int index = 0; index < static_cast<int>(m_FieldDetector.m_LongLineSegments.size()); index++) {
        const cv::Vec4f &segment = m_FieldDetector.m_LongLineSegments[index];
        cv::Point a(segment[0], segment[1]);
        cv::Point b(segment[2], segment[3]);
        bool edge = m_FieldDetector.m_EdgePolarity[index];
        if (b.y < a.y) {
            std::swap(a.x, b.x);
            std::swap(a.y, b.y);
            edge = !edge;
        }
        //cv::line(white_show, a, b, edge ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 255), 2);
        cv::line(white_show, a, b, cv::Scalar(255, 0, 0), 2);
    }
    for (const cv::Vec6f &white_line : m_FieldDetector.m_WhiteLines) {
        cv::Point a(white_line[0], white_line[1]);
        cv::Point b(white_line[2], white_line[3]);
        int thickness = std::max(static_cast<int>(round(white_line[4])), 1);
        cv::line(white_show, a, b, cv::Scalar(0, 255, 255), thickness);
    }



    // 芝の領域を示す矩形を描画する
    cv::line(green_show, grass_rect[0] / kScaleFactor, grass_rect[1] / kScaleFactor, cv::Scalar(0, 255, 0));
    cv::line(green_show, grass_rect[1] / kScaleFactor, grass_rect[2] / kScaleFactor, cv::Scalar(0, 255, 0));
    cv::line(green_show, grass_rect[2] / kScaleFactor, grass_rect[3] / kScaleFactor, cv::Scalar(0, 255, 0));
    cv::line(green_show, grass_rect[3] / kScaleFactor, grass_rect[0] / kScaleFactor, cv::Scalar(0, 255, 0));

    


    for (int index = 0; index < m_FieldDetector.m_LineIntersections.rows; index++) {
        double x = m_FieldDetector.m_LineIntersections.at<float>(index, 0);
        double y = m_FieldDetector.m_LineIntersections.at<float>(index, 1);
        cv::circle(white_show, cv::Point(x, y), 4, cv::Scalar(255, 255, 0), 2);
    }


    // 消失点を描画する
    /*cv::Mat vanish;
    if (3 <= m_FieldDetector.m_LineIntersections.rows) {
        double min_x, max_x, min_y, max_y;
        min_x = max_x = m_FieldDetector.m_LineIntersections.at<float>(0, 0);
        min_y = max_y = m_FieldDetector.m_LineIntersections.at<float>(0, 1);
        for (int index = 1; index < m_FieldDetector.m_LineIntersections.rows; index++) {
            double x = m_FieldDetector.m_LineIntersections.at<float>(index, 0);
            double y = m_FieldDetector.m_LineIntersections.at<float>(index, 1);
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
        }
        QSize window_size = m_Field[1]->size();
        vanish.create(window_size.height(), window_size.width(), CV_8UC3);
        vanish.setTo(0);
        int width = vanish.cols, height = vanish.rows;
        for (int index = 0; index < 4; index++) {
            int image_x1 = static_cast<int>((grass_rect[index].x - min_x) / (max_x - min_x) * width);
            int image_y1 = static_cast<int>((grass_rect[index].y - min_y) / (max_y - min_y) * height);
            int image_x2 = static_cast<int>((grass_rect[(index + 1) % 4].x - min_x) / (max_x - min_x) * width);
            int image_y2 = static_cast<int>((grass_rect[(index + 1) % 4].y - min_y) / (max_y - min_y) * height);
            cv::line(vanish, cv::Point(image_x1, image_y1), cv::Point(image_x2, image_y2), cv::Scalar(0, 255, 0));
        }
        for(const auto &segment : m_FieldDetector.m_LongLineSegments){
        }
        for (int index = 0; index < m_FieldDetector.m_LineIntersections.rows; index++) {
            double x = m_FieldDetector.m_LineIntersections.at<float>(index, 0);
            double y = m_FieldDetector.m_LineIntersections.at<float>(index, 1);
            int image_x = static_cast<int>((x - min_x) / (max_x - min_x) * width);
            int image_y = static_cast<int>((y - min_y) / (max_y - min_y) * height);
            cv::circle(vanish, cv::Point(image_x, image_y), 2, cv::Scalar(255, 255, 0));
        }
        if (m_FieldDetector.m_VanishingPoints.size() == 3) {
            for (int index = 0; index < 3; index++) {
                std::ostringstream text;
                text << m_FieldDetector.m_VanishingPoints[index].x << ", " << m_FieldDetector.m_VanishingPoints[index].y;
                cv::putText(vanish, text.str(), cv::Point(8, 16 + 16 * index), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
            }
        }
    }*/




    /*for (int i = 0; i < centers.rows; i++) {
        cv::Vec2f &color = centers.at<cv::Vec2f>(i, 0);
        if ((1 << i) & inside_green_flag) {
            cv::circle(histogram_show, cv::Point((int)color[0], (int)color[1]), 2, cv::Scalar(255, 128, 0));
        } else {
            cv::circle(histogram_show, cv::Point((int)color[0], (int)color[1]), 2, cv::Scalar(0, 255, 0));
        }
    }*/


    /*int threshold = m_ThresholdInput->value();
    cv::Mat binary[2];
    cv::adaptiveThreshold(gray[0], binary[0], 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, threshold);
    cv::adaptiveThreshold(gray[1], binary[1], 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, threshold);*/

    //m_Color[0]->setImage(m_ColorImage[0]);
    //m_Color[1]->setImage(medianed/*m_EnhancedColorImage[0]*/);
    //m_Field[0]->setImage(m_WhiteLineImage[0]);
    //m_Field[0]->setImage(histogram_show);
    //m_Field[0]->setImage(green_show);
    m_Field[0]->setImage(white_show);
}

void VideoFieldDetectorThread::showColor(int x, int y) {
    m_WatchPointX = x;
    m_WatchPointY = y;
}
