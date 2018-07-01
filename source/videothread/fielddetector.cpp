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
    new(&m_EnhancementFilter) EnhancementFilter(kScaleFactor, 5);
}

VideoFieldDetectorThread::~VideoFieldDetectorThread(){
    quitThread();
}

QString VideoFieldDetectorThread::initializeOnce(QWidget *parent) {
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
    for (int index = 0; index < 4; index++) {
        m_Output[index] = new ImageViewGl;
        grid_layout->addWidget(m_Output[index], index / 2, index % 2);
        connect(this, &VideoThread::update, m_Output[index], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    }

    m_Output[0]->convertBgrToRgb();
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

    // ステレオマッチングの前処理を行う
    cv::cvtColor(m_ColorImage[0], m_GrayscaleImage[0], cv::COLOR_BGR2GRAY);
    cv::cvtColor(m_ColorImage[1], m_GrayscaleImage[1], cv::COLOR_BGR2GRAY);
    m_Stereo.precompute(m_GrayscaleImage[0], m_GrayscaleImage[1]);

    cv::Mat disparity_map, disparity_map_color;
    m_Stereo.compute(m_DisparityMap, 32);
    m_DisparityMap.convertTo(disparity_map, CV_8U, 8.0);
    cv::cvtColor(disparity_map, disparity_map_color, cv::COLOR_GRAY2RGB);

    // 白線の検知を行う
    std::vector<cv::Vec4f> line_segments;
    std::vector<cv::Vec4f> edge_line_segments;
    const cv::Mat &white = m_FieldDetector.detect(m_ColorImage[0], line_segments, &edge_line_segments);

    static const cv::Scalar color_list[] = {
        cv::Scalar(255, 0, 0)/*,
        cv::Scalar(255, 128, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 255, 255),
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 0, 255),*/
    };
    int color_number = 0;
    cv::cvtColor(m_ColorImage[0], m_WhiteLineImage, cv::COLOR_BGR2RGB);
    for (const cv::Vec4f &segment : edge_line_segments) {
        cv::Point2f a(segment[0], segment[1]);
        cv::Point2f b(segment[2], segment[3]);
        cv::line(m_WhiteLineImage, a, b, color_list[color_number], 2);
        color_number = (color_number + 1) % (sizeof(color_list) / sizeof(cv::Scalar));
    }
    for (const cv::Vec4f &segment : line_segments) {
        cv::Point2f a(segment[0], segment[1]);
        cv::Point2f b(segment[2], segment[3]);
        cv::line(m_WhiteLineImage, a, b, cv::Scalar(0, 128, 255), 3);
    }
    
    // 検知した芝を表示する
    cv::Mat green_display(height, width, CV_8UC3);
    const cv::Mat &green = m_FieldDetector.grassBinaryImage();
    cv::Rect rect = m_FieldDetector.grassRectangle();
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            bool grass = green.at<uint8_t>(y, x);
            green_display.at<cv::Vec3b>(y, x) = (grass == true) ? cv::Vec3b(0, 255, 0) : cv::Vec3b(0, 0, 0);
        }
    }
    cv::rectangle(green_display, rect, cv::Scalar(255, 0, 0), 2);

    

	cv::Point3d field_centroid, field_normal;
	double estimation_error = m_PositionTracker.estimateFieldPlane(line_segments, m_Stereo, m_Undistort, &field_centroid, &field_normal);
	double roll_angle = m_PositionTracker.calculateRollAngle();
	double pitch_angle = m_PositionTracker.calculatePitchAngle();
    double viewpoint_height = m_PositionTracker.calculateHeight();

    // 表示用画像から芝の範囲外を黒塗りする
    cv::cvtColor(m_ColorImage[0], m_WhiteLineImage, cv::COLOR_BGR2RGB);
    m_UnrollImage.create(height, width, CV_8UC3);
    m_UnrollImage.setTo(0);
    m_UnpitchImage.create(height, width, CV_8UC3);
    m_UnpitchImage.setTo(0);

    // 推定のサンプルに使われた点を描画する
    for (const cv::Point3f &sample : m_PositionTracker.m_UsedPoints) {
        cv::circle(disparity_map_color, cv::Point(sample.x, sample.y), 3, cv::Scalar(255, 0, 0), 2);
    }

    // 面の法線ベクトルのカメラに対してのねじれを求める(Z軸周りの回転)
    float roll_matrix_data[9] = {
        cos(roll_angle), -sin(roll_angle), 0.0f,
        sin(roll_angle), cos(roll_angle), 0.0f,
        0.0f, 0.0f, 1.0f
    };
    cv::Mat1f roll_matrix(3, 3, roll_matrix_data);
    cv::putText(m_WhiteLineImage, cv::format("Roll = %.2f deg", roll_angle / M_PI * 180.0), cv::Point(width / 2, height - 32), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    // 面のX軸方向の傾きを求める
    double unpitch_angle = pitch_angle - M_PI * 0.5;
    float unpitch_matrix_data[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, cos(unpitch_angle), -sin(unpitch_angle),
        0.0f, sin(unpitch_angle), cos(unpitch_angle)
    };
    cv::Mat1f unpitch_matrix(3, 3, unpitch_matrix_data);
    cv::putText(m_UnrollImage, cv::format("Pitch = %.2f deg", pitch_angle * 180 / M_PI), cv::Point(width / 2, height - 32), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    // 白線を描画する
    {
        cv::Mat camera_matrix = m_Undistort.cameraMatrix(0);
        double fx = camera_matrix.at<double>(0, 0);
        double cx = camera_matrix.at<double>(0, 2);
        double fy = camera_matrix.at<double>(1, 1);
        double cy = camera_matrix.at<double>(1, 2);

        cv::Point3f zero(0.0, 0.0, 0.0);
        std::vector<cv::Point3f> world_points(2);
        std::vector<cv::Point2f> image_points(2);
        for (const cv::Vec4f &segment : edge_line_segments) {
            cv::Point2f a(segment[0], segment[1]);
            cv::Point2f b(segment[2], segment[3]);
            cv::line(m_WhiteLineImage, a, b, cv::Scalar(255, 0, 0), 2);
            cv::Point3f vec_a_world((a.x - cx) / fx, (a.y - cy) / fy, 1.0f);
            cv::Point3f vec_b_world((b.x - cx) / fx, (b.y - cy) / fy, 1.0f);
            cv::Point3f a_world = zero - field_normal.dot(zero - cv::Point3f(field_centroid)) / field_normal.dot(vec_a_world) * vec_a_world;
            cv::Point3f b_world = zero - field_normal.dot(zero - cv::Point3f(field_centroid)) / field_normal.dot(vec_b_world) * vec_b_world;
            world_points[0] = (cv::Point3f)cv::Mat(roll_matrix * cv::Mat(a_world));
            world_points[1] = (cv::Point3f)cv::Mat(roll_matrix * cv::Mat(b_world));
            m_Undistort.projectPointsTo2D(world_points, image_points);
            cv::line(m_UnrollImage, image_points[0], image_points[1], cv::Scalar(255, 0, 0), 2);

            world_points[0] = (cv::Point3f)cv::Mat(unpitch_matrix * roll_matrix * cv::Mat(a_world));
            world_points[1] = (cv::Point3f)cv::Mat(unpitch_matrix * roll_matrix * cv::Mat(b_world));
            image_points[0] = cv::Point2f(100.0 * world_points[0].x + width * 0.5, -100.0 * world_points[0].y + height);
            image_points[1] = cv::Point2f(100.0 * world_points[1].x + width * 0.5, -100.0 * world_points[1].y + height);
            cv::line(m_UnpitchImage, image_points[0], image_points[1], cv::Scalar(255, 0, 0), 2);
        }
        cv::circle(m_UnpitchImage, cv::Point(width / 2, height), 10, cv::Scalar(255, 255, 255), -1);
        cv::putText(m_UnpitchImage, cv::format("Height = %.2f m", viewpoint_height), cv::Point(width / 2 + 10, height - 16), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        cv::putText(m_UnpitchImage, cv::format("Error = %.4f", estimation_error), cv::Point(width / 2 + 10, height - 48), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    }

    // 平面を表す軸を表示する
    if (m_PositionTracker.m_PCAEigenVectors.empty() == false) {
        std::vector<cv::Point3f> points3d(4);
        std::vector<cv::Point2f> points2d;

        // 視線の中心から延びた直線が平面と交わる点x1を求める
        cv::Point3d x0(0.0f, 0.0f, 0.0f);
        cv::Point3d p(0.0f, 0.0f, 1.0f);
        cv::Point3f x1 = x0 - field_normal.dot(x0 - field_centroid) / field_normal.dot(p) * p;

        points3d[0] = x1;
        for (int axis = 0; axis <= 2; axis++) {
            points3d[axis + 1].x = points3d[0].x + m_PositionTracker.m_PCAEigenVectors.at<float>(axis, 0);
            points3d[axis + 1].y = points3d[0].y + m_PositionTracker.m_PCAEigenVectors.at<float>(axis, 1);
            points3d[axis + 1].z = points3d[0].z + m_PositionTracker.m_PCAEigenVectors.at<float>(axis, 2);
        }

        m_Undistort.projectPointsTo2D(points3d, points2d);
        cv::circle(m_WhiteLineImage, cv::Point(points2d[0].x, points2d[0].y), 5, cv::Scalar(255, 0, 0), 2);
        cv::line(m_WhiteLineImage, cv::Point(points2d[0].x, points2d[0].y), cv::Point(points2d[1].x, points2d[1].y), cv::Scalar(255, 0, 0), 4);
        cv::line(m_WhiteLineImage, cv::Point(points2d[0].x, points2d[0].y), cv::Point(points2d[2].x, points2d[2].y), cv::Scalar(0, 255, 0), 4);
        cv::line(m_WhiteLineImage, cv::Point(points2d[0].x, points2d[0].y), cv::Point(points2d[3].x, points2d[3].y), cv::Scalar(0, 128, 255), 4);
        cv::putText(m_WhiteLineImage, cv::format("%.2f m", points3d[0].z), cv::Point(points2d[0].x + 8, points2d[0].y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

        // 傾きを補正して再描画
        points3d[0] = (cv::Point3f)cv::Mat(roll_matrix * cv::Mat(points3d[0]));
        points3d[1] = (cv::Point3f)cv::Mat(roll_matrix * cv::Mat(points3d[1]));
        points3d[2] = (cv::Point3f)cv::Mat(roll_matrix * cv::Mat(points3d[2]));
        points3d[3] = (cv::Point3f)cv::Mat(roll_matrix * cv::Mat(points3d[3]));

        m_Undistort.projectPointsTo2D(points3d, points2d);

        cv::circle(m_UnrollImage, cv::Point(points2d[0].x, points2d[0].y), 5, cv::Scalar(255, 0, 0), 2);
        cv::line(m_UnrollImage, cv::Point(points2d[0].x, points2d[0].y), cv::Point(points2d[1].x, points2d[1].y), cv::Scalar(255, 0, 0), 4);
        cv::line(m_UnrollImage, cv::Point(points2d[0].x, points2d[0].y), cv::Point(points2d[2].x, points2d[2].y), cv::Scalar(0, 255, 0), 4);
        cv::line(m_UnrollImage, cv::Point(points2d[0].x, points2d[0].y), cv::Point(points2d[3].x, points2d[3].y), cv::Scalar(0, 128, 255), 4);
    }

    // 3次元線分を描画する
    {
        std::vector<cv::Point3f> world_points(2 * m_PositionTracker.m_LineSegmentCentroids.size());
        std::vector<cv::Point2f> image_points;
        for (int i = 0; i < static_cast<int>(m_PositionTracker.m_LineSegmentCentroids.size()); i++) {
            world_points[2 * i + 0] = m_PositionTracker.m_LineSegmentCentroids[i] - m_PositionTracker.m_LineSegmentVectors[i];
            world_points[2 * i + 1] = m_PositionTracker.m_LineSegmentCentroids[i] + m_PositionTracker.m_LineSegmentVectors[i];
        }
        m_Undistort.projectPointsTo2D(world_points, image_points);
        for (int i = 0; i < static_cast<int>(image_points.size() / 2); i++) {
            cv::line(m_WhiteLineImage, image_points[2 * i], image_points[2 * i + 1], cv::Scalar(0, 128, 255), 4);
        }
        for (int i = 0; i < static_cast<int>(image_points.size() / 2); i++) {
            cv::putText(m_WhiteLineImage, cv::format("%.4f", m_PositionTracker.m_LineSegmentLikelihood[i]), (image_points[2 * i] + image_points[2 * i + 1]) / 2 + cv::Point2f(8, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        }
    }

    m_Output[0]->setImage(m_WhiteLineImage);
    m_Output[1]->setImage(disparity_map_color);
    m_Output[2]->setImage(m_UnrollImage);
    m_Output[3]->setImage(m_UnpitchImage);

    //m_Output[0]->setImage(m_ColorImage[0]);
    //m_Output[1]->setImage(m_WhiteLineImage);
    //m_Output[2]->setImage(green_display);
    //m_Output[3]->setImage(white);
}

void VideoFieldDetectorThread::showColor(int x, int y) {
    m_WatchPointX = x;
    m_WatchPointY = y;
}
