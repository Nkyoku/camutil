#include "line_stereo.h"
#include "../imageviewgl.h"
#include "algorithm/geometry_2d.h"
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QBoxLayout>
#define _USE_MATH_DEFINES
#include <math.h>


VideoLineStereoThread::VideoLineStereoThread(VideoInput *video_input)
    : VideoThread(video_input)
{
    m_Lsd = cv::createLineSegmentDetector(1, 0.5);
}

VideoLineStereoThread::~VideoLineStereoThread(){
    quitThread();
}

QString VideoLineStereoThread::initializeOnce(QWidget *parent) {
    QVBoxLayout *layout = new QVBoxLayout;
    parent->setLayout(layout);
    
    m_HeightInput = new QDoubleSpinBox;
    m_HeightInput->setSingleStep(0.01);
    m_HeightInput->setMinimum(0.1);
    m_HeightInput->setMaximum(2.0);
    m_HeightInput->setValue(1.0);
    layout->addWidget(m_HeightInput);

    m_PitchAngleInput = new QDoubleSpinBox;
    m_PitchAngleInput->setSingleStep(0.1);
    m_PitchAngleInput->setMinimum(-90.0);
    m_PitchAngleInput->setMaximum(90.0);
    layout->addWidget(m_PitchAngleInput);

    m_RollAngleInput = new QDoubleSpinBox;
    m_RollAngleInput->setSingleStep(0.1);
    m_RollAngleInput->setMinimum(-90.0);
    m_RollAngleInput->setMaximum(90.0);
    layout->addWidget(m_RollAngleInput);

    return tr("LineStereo");
}

void VideoLineStereoThread::initialize(QWidget *parent) {
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
    m_Output[1]->convertBgrToRgb();
}

void VideoLineStereoThread::uninitialize(void) {
    m_Undistort.destroy();
}

void VideoLineStereoThread::restoreSettings(const QSettings &settings) {
    
}

void VideoLineStereoThread::saveSettings(QSettings &settings) const {
    
}

void VideoLineStereoThread::processImage(const cv::Mat &input_image) {
    int width = input_image.cols / 2;
    int height = input_image.rows;

    m_Undistort.undistort(cv::Mat(input_image, cv::Rect(0, 0, width, height)), m_OriginalBgrImage[0], 0);
    m_Undistort.undistort(cv::Mat(input_image, cv::Rect(width, 0, width, height)), m_OriginalBgrImage[1], 1);
    cv::cvtColor(m_OriginalBgrImage[0], m_OriginalGrayscaleImage[0], cv::COLOR_BGR2GRAY);
    cv::cvtColor(m_OriginalBgrImage[1], m_OriginalGrayscaleImage[1], cv::COLOR_BGR2GRAY);
    cv::cvtColor(m_OriginalBgrImage[0], m_OriginalLabImage[0], cv::COLOR_BGR2Lab);
    cv::cvtColor(m_OriginalBgrImage[1], m_OriginalLabImage[1], cv::COLOR_BGR2Lab);

    // 芝を検知する
    std::vector<cv::Range> field_ranges[2];
    m_FieldDetector.detectGrass(m_OriginalLabImage[0], &field_ranges[0], 1);
    m_FieldDetector.detectGrass(m_OriginalLabImage[1], &field_ranges[1], 1);

    // 芝の検知を行う
    // 強調フィルタの処理過程で出た中間値画像を利用して白線を除く
    cv::Mat green_display(height, width, CV_8UC3);
    for (int y = 0; y < height; y++) {
        int start_l_x = field_ranges[0][y].start;
        int end_l_x = field_ranges[0][y].end;
        int start_r_x = field_ranges[1][y].start;
        int end_r_x = field_ranges[1][y].end;
        for (int x = 0; x < width; x++) {
            bool inside_left = (start_l_x <= x) && (x < end_l_x);
            bool inside_right = (start_r_x <= x) && (x < end_r_x);
            if (inside_left == true) {
                if (inside_right == true) {
                    green_display.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
                } else {
                    green_display.at<cv::Vec3b>(y, x) = cv::Vec3b(128, 128, 128);
                }
            } else {
                if (inside_right == true) {
                    green_display.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 128, 0);
                } else {
                    green_display.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                }
            }
        }
    }

    // 線分を検知する
    std::vector<cv::Vec4f> raw_segments[2];
    m_Lsd->detect(m_OriginalGrayscaleImage[0], raw_segments[0]);
    m_Lsd->detect(m_OriginalGrayscaleImage[1], raw_segments[1]);
    
    // 端点、中点のいずれかがフィールド内にある線分のみを選別する
    std::vector<cv::Vec4f> raw_segments_inside[2];
    for (int side = 0; side < 2; side++) {
        raw_segments_inside[side].reserve(raw_segments[side].size());
        for (const cv::Vec4f &segment : raw_segments[side]) {
            cv::Point2d a(segment[0], segment[1]);
            cv::Point2d b(segment[2], segment[3]);
            cv::Point2d c((segment[0] + segment[2]) * 0.5, (segment[1] + segment[3]) * 0.5);
            auto is_inside = [](const cv::Point2d &p, const std::vector<cv::Range> &ranges) -> bool {
                int y = std::max(0, std::min(static_cast<int>(round(p.y)), static_cast<int>(ranges.size()) - 1));
                int x = static_cast<int>(round(p.x));
                return (ranges[y].start <= x) && (x < ranges[y].end);
            };
            if (is_inside(a, field_ranges[side]) || is_inside(b, field_ranges[side]) || is_inside(c, field_ranges[side])) {
                raw_segments_inside[side].push_back(segment);
            }
        }
    }

    // 平面を仮定する
    double roll_angle = m_RollAngleInput->value() / 180.0 * M_PI;
    double pitch_angle = m_PitchAngleInput->value() / 180.0 * M_PI;
    double unpitch_angle = pitch_angle - M_PI * 0.5;
    float roll_matrix_data[9] = {
        cos(roll_angle), -sin(roll_angle), 0.0f,
        sin(roll_angle), cos(roll_angle), 0.0f,
        0.0f, 0.0f, 1.0f
    };
    float unpitch_matrix_data[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, cos(unpitch_angle), -sin(unpitch_angle),
        0.0f, sin(unpitch_angle), cos(unpitch_angle)
    };
    cv::Mat1f unpitch_matrix(3, 3, unpitch_matrix_data);
    cv::Mat1f roll_matrix(3, 3, roll_matrix_data);
    cv::Point3d field_normal;
    field_normal.x = sin(roll_angle) * cos(pitch_angle);
    field_normal.y = cos(roll_angle) * cos(pitch_angle);
    field_normal.z = sin(pitch_angle);

    // 平面に投影された線分を法線方向から見た図を描画する
    cv::Mat overlook_image[2];
    for(int side = 0; side < 2; side++) {
        overlook_image[side].create(height, width, CV_8UC3);
        overlook_image[side].setTo(0);

        cv::Point3d field_centroid(0.0, (side == 1) ? m_HeightInput->value() : 1.0, 0.0);

        cv::Mat projection_matrix = m_Undistort.projectionMatrix(side);
        double fx = projection_matrix.at<double>(0, 0);
        double cx = projection_matrix.at<double>(0, 2);
        double fy = projection_matrix.at<double>(1, 1);
        double cy = projection_matrix.at<double>(1, 2);

        cv::Point3f zero(0.0, 0.0, 0.0);
        std::vector<cv::Point3f> world_points(2);
        std::vector<cv::Point2f> image_points(2);
        for (const cv::Vec4f &segment : raw_segments_inside[side]) {
            cv::Point2f a(segment[0], segment[1]);
            cv::Point2f b(segment[2], segment[3]);
            cv::Point3f vec_a_world((a.x - cx) / fx, (a.y - cy) / fy, 1.0f);
            cv::Point3f vec_b_world((b.x - cx) / fx, (b.y - cy) / fy, 1.0f);
            cv::Point3f a_world = zero - field_normal.dot(zero - cv::Point3f(field_centroid)) / field_normal.dot(vec_a_world) * vec_a_world;
            cv::Point3f b_world = zero - field_normal.dot(zero - cv::Point3f(field_centroid)) / field_normal.dot(vec_b_world) * vec_b_world;
            world_points[0] = (cv::Point3f)cv::Mat(unpitch_matrix * roll_matrix * cv::Mat(a_world));
            world_points[1] = (cv::Point3f)cv::Mat(unpitch_matrix * roll_matrix * cv::Mat(b_world));
            image_points[0] = cv::Point2f(100.0 * world_points[0].x + width * 0.5, -100.0 * world_points[0].y + height);
            image_points[1] = cv::Point2f(100.0 * world_points[1].x + width * 0.5, -100.0 * world_points[1].y + height);
            cv::line(overlook_image[0], image_points[0], image_points[1], (side == 0) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 128, 255), 2);
        }
    }

    std::vector<cv::Vec4f> combined_segments[2];
    //combineParallelSegments(raw_segments_inside[0], combined_segments[0], 16.0);
    //combineParallelSegments(raw_segments_inside[1], combined_segments[1], 16.0);

    drawSegments(m_OriginalBgrImage[0], raw_segments_inside[0], cv::Scalar(0, 0, 255));
    drawSegments(m_OriginalBgrImage[1], raw_segments_inside[1], cv::Scalar(0, 0, 255));
    drawSegments(m_OriginalBgrImage[0], combined_segments[0], cv::Scalar(255, 128, 0), 3);
    drawSegments(m_OriginalBgrImage[1], combined_segments[1], cv::Scalar(255, 128, 0), 3);

    m_Output[0]->setImage(m_OriginalBgrImage[0]);
    m_Output[1]->setImage(m_OriginalBgrImage[1]);
    m_Output[2]->setImage(overlook_image[0]);
    m_Output[3]->setImage(overlook_image[1]);
}

void VideoLineStereoThread::combineParallelSegments(const std::vector<cv::Vec4f> &input_segments, std::vector<cv::Vec4f> &output_segments, double threshold) {
    output_segments.reserve(input_segments.size());
    for (int i = 0; i < static_cast<int>(input_segments.size()); i++) {
        // 線分A
        cv::Point2d a1(input_segments[i][0], input_segments[i][1]);
        cv::Point2d a2(input_segments[i][2], input_segments[i][3]);
        cv::Point2d vector_a = a2 - a1;
        double length_a = normalizeAndLength(vector_a);
        for (int j = i + 1; j < static_cast<int>(input_segments.size()); j++) {
            // 線分B
            cv::Point2d b1(input_segments[j][0], input_segments[j][1]);
            cv::Point2d b2(input_segments[j][2], input_segments[j][3]);
            cv::Point2d vector_b = b2 - b1;
            double length_b = normalizeAndLength(vector_b);

            // 線分A,Bの成す角がほぼ平行であるか調べる
            double cos_angle = vector_a.dot(vector_b);
            if (abs(cos_angle) < kParallelAngle) {
                continue;
            }
            if (cos_angle < 0) {
                std::swap(b1, b2);
                vector_b *= -1;
            }

            // 線分同士が近いか調べる
            if (threshold < distanceBetweenSegmentsSimple(a1, a2, b1, b2)) {
                continue;
            }
            if (threshold < distanceBetweenSegments(a1, a2, b1, b2)) {
                continue;
            }

            // 線分Cは線分A,Bの平均ベクトル
            cv::Point2d c1((a1 + b1) * 0.5);
            cv::Point2d c2((a2 + b2) * 0.5);

            cv::Vec4f result;
            result[0] = static_cast<float>(c1.x);
            result[1] = static_cast<float>(c1.y);
            result[2] = static_cast<float>(c2.x);
            result[3] = static_cast<float>(c2.y);
            output_segments.push_back(result);
        }
    }
}

void VideoLineStereoThread::drawSegments(cv::Mat &image, const std::vector<cv::Vec4f> &line_segments, const cv::Scalar &color, int thickness) {
    for (const cv::Vec4f &segment : line_segments) {
        cv::line(image, cv::Point(segment[0], segment[1]), cv::Point(segment[2], segment[3]), color, thickness);
    }
}
