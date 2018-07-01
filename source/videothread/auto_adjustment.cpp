#include "auto_adjustment.h"
#include "ui_auto_adjustment.h"
#include "imageviewgl.h"
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QBoxLayout>
#define _USE_MATH_DEFINES
#include <math.h>

VideoAutoAdjustmentThread::VideoAutoAdjustmentThread(VideoInput *video_input)
    : VideoThread(video_input)
{
    m_Akaze = cv::AKAZE::create();
    m_Matcher = cv::DescriptorMatcher::create("BruteForce");
    m_StereoSgbm = cv::StereoSGBM::create(0, 32, 5);
}

VideoAutoAdjustmentThread::~VideoAutoAdjustmentThread(){
    quitThread();
}

QString VideoAutoAdjustmentThread::initializeOnce(QWidget *parent) {
    m_ui = new Ui_AutoAdjustment;
    m_ui->setupUi(parent);
    
    connect(m_ui->Roll, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &VideoAutoAdjustmentThread::setRollAngle, Qt::QueuedConnection);
    connect(m_ui->Pitch, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &VideoAutoAdjustmentThread::setPitchAngle, Qt::QueuedConnection);
    connect(m_ui->StartCapture, &QPushButton::clicked, this, &VideoAutoAdjustmentThread::startCapture, Qt::QueuedConnection);
    connect(m_ui->FinishCapture, &QPushButton::clicked, this, &VideoAutoAdjustmentThread::finishCapture, Qt::QueuedConnection);
    connect(m_ui->Apply, &QPushButton::clicked, this, &VideoAutoAdjustmentThread::applyAdjustment, Qt::QueuedConnection);
    connect(m_ui->Save, &QPushButton::clicked, this, &VideoAutoAdjustmentThread::saveAdjustment, Qt::QueuedConnection);
    connect(m_ui->ManualUp, &QPushButton::clicked, m_ui->Pitch, &QDoubleSpinBox::stepUp);
    connect(m_ui->ManualDown, &QPushButton::clicked, m_ui->Pitch, &QDoubleSpinBox::stepDown);
    connect(m_ui->ManualLeft, &QPushButton::clicked, m_ui->Roll, &QDoubleSpinBox::stepDown);
    connect(m_ui->ManualRight, &QPushButton::clicked, m_ui->Roll, &QDoubleSpinBox::stepUp);
    connect(m_ui->ManualUp, &QPushButton::clicked, this, &VideoAutoAdjustmentThread::applyAdjustment, Qt::QueuedConnection);
    connect(m_ui->ManualDown, &QPushButton::clicked, this, &VideoAutoAdjustmentThread::applyAdjustment, Qt::QueuedConnection);
    connect(m_ui->ManualLeft, &QPushButton::clicked, this, &VideoAutoAdjustmentThread::applyAdjustment, Qt::QueuedConnection);
    connect(m_ui->ManualRight, &QPushButton::clicked, this, &VideoAutoAdjustmentThread::applyAdjustment, Qt::QueuedConnection);

    return tr("AutoAdjust");
}

void VideoAutoAdjustmentThread::initialize(QWidget *parent) {
    uninitialize();

    QSize size = m_VideoInput->sourceResolution();
    m_Undistort.load(size.width() / 2, size.height());

    double roll_radian, pitch_radian;
    m_Undistort.adjustmentParameters(&roll_radian, &pitch_radian);
    setRollAngle(roll_radian * 180.0 / M_PI);
    setPitchAngle(pitch_radian * 180.0 / M_PI);

    // ImageViewGlを生成
    QGridLayout *grid_layout = new QGridLayout;
    parent->setLayout(grid_layout);
    for (int side = 0; side < 2; side++) {
        m_OriginalOutput[side] = new ImageViewGl;
        m_OriginalOutput[side]->convertBgrToRgb();
        grid_layout->addWidget(m_OriginalOutput[side], 0, side);
        connect(this, &VideoThread::update, m_OriginalOutput[side], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    }
    m_StereoOutput = new ImageViewGl;
    grid_layout->addWidget(m_StereoOutput, 1, 0, 1, 2);
    connect(this, &VideoThread::update, m_StereoOutput, QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
}

void VideoAutoAdjustmentThread::uninitialize(void) {
    m_Undistort.destroy();
    m_IsCapturing = false;
}

void VideoAutoAdjustmentThread::restoreSettings(const QSettings &settings) {
    
}

void VideoAutoAdjustmentThread::saveSettings(QSettings &settings) const {
    
}

void VideoAutoAdjustmentThread::processImage(const cv::Mat &input_image) {
    int width = input_image.cols / 2;
    int height = input_image.rows;
    
    m_Undistort.undistort(cv::Mat(input_image, cv::Rect(0, 0, width, height)), m_OriginalImage[0], 0);
    m_Undistort.undistort(cv::Mat(input_image, cv::Rect(width, 0, width, height)), m_OriginalImage[1], 1);

    // ステレオマッチングを行う
    cv::cvtColor(m_OriginalImage[0], m_GrayscaleImage[0], cv::COLOR_BGR2GRAY);
    cv::cvtColor(m_OriginalImage[1], m_GrayscaleImage[1], cv::COLOR_BGR2GRAY);
    m_StereoSgbm->compute(m_GrayscaleImage[0], m_GrayscaleImage[1], m_StereoFixedPoint);
    m_StereoFixedPoint.convertTo(m_Stereo8Bit, CV_8U, 0.5);

    if (m_IsCapturing == true) {
        // 特徴点・特徴量を算出する
        std::vector<cv::KeyPoint> keypoints_left, keypoints_right;
        cv::Mat descriptors_left, descriptors_right;
        m_Akaze->detect(m_OriginalImage[0], keypoints_left);
        m_Akaze->compute(m_OriginalImage[0], keypoints_left, descriptors_left);
        m_Akaze->detect(m_OriginalImage[1], keypoints_right);
        m_Akaze->compute(m_OriginalImage[1], keypoints_right, descriptors_right);

        // 特徴点のマッチングを行う
        std::vector<cv::DMatch> match_l_r, match_r_l, mutual_match;
        m_Matcher->match(descriptors_left, descriptors_right, match_l_r);
        m_Matcher->match(descriptors_right, descriptors_left, match_r_l);
        for (size_t i = 0; i < match_l_r.size(); i++) {
            cv::DMatch forward = match_l_r[i];
            cv::DMatch backward = match_r_l[forward.trainIdx];
            if (backward.trainIdx == forward.queryIdx) {
                mutual_match.push_back(forward);
            }
        }

        // 特徴点の位置のずれを記録する
        // ずれの大きすぎるものは除外する
        double reject_threshold = kMaximumAngle / 180.0 * M_PI * width * 0.5;
        for (const cv::DMatch &match : mutual_match) {
            const cv::Point2f &left = keypoints_left[match.queryIdx].pt;
            const cv::Point2f &right = keypoints_right[match.trainIdx].pt;
            double disparity = right.y - left.y;
            if (abs(disparity) < reject_threshold) {
                m_CaptureData.push_back(cv::Vec2f(right.x, static_cast<float>(disparity)));
            }
        }

        cv::String text = cv::format("Measuring... (%d samples)", static_cast<int>(m_CaptureData.size()));
        cv::putText(m_OriginalImage[0], text, cv::Point(8, 32), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }

    m_OriginalOutput[0]->setImage(m_OriginalImage[0]);
    m_OriginalOutput[1]->setImage(m_OriginalImage[1]);
    m_StereoOutput->setImage(m_Stereo8Bit);
}

void VideoAutoAdjustmentThread::setRollAngle(double roll) {
    m_RollAngle = roll;
    m_ui->Roll->blockSignals(true);
    m_ui->Roll->setValue(m_RollAngle);
    m_ui->Roll->blockSignals(false);
}

void VideoAutoAdjustmentThread::setPitchAngle(double pitch) {
    m_PitchAngle = pitch;
    m_ui->Pitch->blockSignals(true);
    m_ui->Pitch->setValue(m_PitchAngle);
    m_ui->Pitch->blockSignals(false);
}

void VideoAutoAdjustmentThread::startCapture(void) {
    if (m_Undistort.isStereoRectified() == false) {
        return;
    }
    m_IsCapturing = true;
    m_ui->StartCapture->setEnabled(false);
    m_ui->FinishCapture->setEnabled(true);
    m_ui->Apply->setEnabled(false);
    m_ui->Save->setEnabled(false);
    m_ui->ManualGroup->setEnabled(false);
    setRollAngle(0.0);
    setPitchAngle(0.0);
    applyAdjustment();
    m_CaptureData.clear();
}

void VideoAutoAdjustmentThread::finishCapture(bool discard) {
    m_IsCapturing = false;
    m_ui->StartCapture->setEnabled(true);
    m_ui->FinishCapture->setEnabled(false);
    m_ui->Apply->setEnabled(true);
    m_ui->Save->setEnabled(true);
    m_ui->ManualGroup->setEnabled(true);
    if (discard || !m_Undistort.isStereoRectified()) {
        m_CaptureData.clear();
        return;
    }

    // 回転補正を加えて分散が最小になる角度を探す
    int width = m_Undistort.width();
    int height = m_Undistort.height();
    double reject_threshold = kMaximumAngle / 180.0 * M_PI * width * 0.5;
    double min_variance = std::numeric_limits<double>::max();
    double min_angle = 0.0;
    double min_mean = 0.0;

    int max_trial = static_cast<int>(2 * kMaximumAngle / kDeltaAngle);
    for (int trial = 1; trial <= max_trial; trial++) {
        double angle = kDeltaAngle * (trial / 2) * ((trial % 2) ? 1 : -1);
        angle *= M_PI / 180.0;

        double mean, variance;
        calculateVariance(angle, &mean, &variance);

        if (variance < min_variance) {
            min_variance = variance;
            min_angle = angle;
            min_mean = mean;
        }
    }

    // 調整値に適用
    setRollAngle(-min_angle * 180.0 / M_PI);
    double focal_y = m_Undistort.projectionMatrix(1).at<double>(1, 1);
    double fov_y = 2 * atan(height / (2 * focal_y));
    double pitch = min_mean / height * fov_y;
    setPitchAngle(pitch * 180.0 / M_PI);
}

void VideoAutoAdjustmentThread::calculateVariance(double angle, double *mean_, double *variance_) {
    int width = m_Undistort.width();
    double reject_threshold = kMaximumAngle / 180.0 * M_PI * width * 0.5;

    double mean = 0.0;
    for (const cv::Vec2f &data : m_CaptureData) {
        double disparity = data[1] - (data[0] - width * 0.5) * angle;
        mean += std::min(std::max(disparity, -reject_threshold), reject_threshold);
    }
    mean /= m_CaptureData.size();
    double variance = 0.0;
    for (const cv::Vec2f &data : m_CaptureData) {
        double disparity = data[1] - (data[0] - width * 0.5) * angle;
        variance += pow(std::min(abs(disparity - mean), reject_threshold), 2);
    }
    variance /= m_CaptureData.size();

    *mean_ = mean;
    *variance_ = variance;
}

void VideoAutoAdjustmentThread::applyAdjustment(void) {
    m_Undistort.adjustRectification(m_RollAngle / 180.0 * M_PI, m_PitchAngle / 180.0 * M_PI);
}

void VideoAutoAdjustmentThread::saveAdjustment(void) {
    m_Undistort.save();
}
