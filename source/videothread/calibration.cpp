#include "calibration.h"
#include "../imageviewgl.h"
#include "ui_calibration.h"
#include <QtWidgets/QBoxLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

VideoCalibrationThread::~VideoCalibrationThread(){
    quitThread();
}

QString VideoCalibrationThread::initializeOnce(QWidget *parent) {
    m_ui = new Ui_Calibration;
    m_ui->setupUi(parent);
    m_ui->CaptureList->setIconSize(QSize(kPreviewSize, kPreviewSize));

    connect(m_ui->Capture, &QPushButton::clicked, this, [&]() {
        m_CaptureCounter = kCaptureDeadline;
    });
    connect(m_ui->Calibrate, &QPushButton::clicked, this, &VideoCalibrationThread::applyCalibration, Qt::QueuedConnection);
    connect(m_ui->Clear, &QPushButton::clicked, this, &VideoCalibrationThread::clearAllPoints, Qt::QueuedConnection);
    connect(m_ui->SetAsDefault, &QPushButton::clicked, this, &VideoCalibrationThread::saveCalibration, Qt::QueuedConnection);

    return tr("Calibration");
}

void VideoCalibrationThread::initialize(QWidget *parent) {
    uninitialize();

    QSize size = m_VideoInput->sourceResolution();
    m_Undistort.load(size.width() / 2, size.height());

    // 補正前と補正後の画像を表示するImageViewGlを生成
    QGridLayout *grid_layout = new QGridLayout;
    parent->setLayout(grid_layout);
    for (int side = 0; side < 2; side++) {
        m_Original[side] = new ImageViewGl;
        grid_layout->addWidget(m_Original[side], 0, side);
        m_Undistorted[side] = new ImageViewGl;
        grid_layout->addWidget(m_Undistorted[side], 1, side);
        connect(this, &VideoThread::update, m_Original[side], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
        connect(this, &VideoThread::update, m_Undistorted[side], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    }
}

void VideoCalibrationThread::uninitialize(void) {
    m_CaptureCounter = 0;
    clearAllPoints();
    m_Undistort.destroy();
}

void VideoCalibrationThread::restoreSettings(const QSettings &settings) {
    m_ui->PatternColumns->setValue(settings.value(tr("CalPatternColumns"), m_ui->PatternColumns->value()).toInt());
    m_ui->PatternRows->setValue(settings.value(tr("CalPatternRows"), m_ui->PatternRows->value()).toInt());
    m_ui->PatternDimmension->setValue(settings.value(tr("CalPatternDimmension"), m_ui->PatternDimmension->value()).toDouble());
}

void VideoCalibrationThread::saveSettings(QSettings &settings) const {
    settings.setValue(tr("CalPatternColumns"), m_ui->PatternColumns->value());
    settings.setValue(tr("CalPatternRows"), m_ui->PatternRows->value());
    settings.setValue(tr("CalPatternDimmension"), m_ui->PatternDimmension->value());
}

void VideoCalibrationThread::processImage(const cv::Mat &input_image) {
    int width = input_image.cols / 2;
    int height = input_image.rows;
    int pattern_cols = m_ui->PatternColumns->value();
    int pattern_rows = m_ui->PatternRows->value();
    cv::Size pattern_size(pattern_cols, pattern_rows);

    // 左右の画像についてチェスボードの交点を検出する
    cv::Mat original[2];
    std::vector<cv::Point2f> image_points[2];
    for (int side = 0; side < 2; side++) {
        cv::cvtColor(cv::Mat(input_image, cv::Rect(side * width, 0, width, height)), original[side], cv::COLOR_BGR2RGB);
    }
    for (int side = 0; side < 2; side++) {
        bool found = cv::findChessboardCorners(original[side], pattern_size, image_points[side], cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
        cv::drawChessboardCorners(original[side], pattern_size, image_points[side], found);
        if (found == false) {
            // 左カメラで正しいパターンが検知できなかったときは右カメラの処理を行わない
            break;
        }
    }

    // Captureボタンが押されたとき検出された点群情報を保存してプレビューをCaptureListに表示する
    if (0 < m_CaptureCounter) {
        m_CaptureCounter--;
        if ((image_points[0].size() == (pattern_cols * pattern_rows)) && (image_points[1].size() == (pattern_cols * pattern_rows))) {
            m_CaptureCounter = 0;

            // 撮影された画像をプレビュー用のアイコンに変換する
            int preview_width, preview_height;
            if (width < height) {
                preview_width = kPreviewSize * width / height;
                preview_height = kPreviewSize;
            } else {
                preview_width = kPreviewSize;
                preview_height = kPreviewSize * height / width;
            }
            cv::Mat preview;
            cv::resize(original[0], preview, cv::Size(preview_width, preview_height));
            QIcon icon(QPixmap::fromImage(QImage(preview.data, preview.cols, preview.rows, QImage::Format_RGB888)));
            
            // チェスボードの点群情報を生成する
            float dimmension = static_cast<float>(m_ui->PatternDimmension->value());
            std::vector<cv::Point3f> object_points;
            object_points.reserve(pattern_cols * pattern_rows);
            for (int row = 0; row < pattern_rows; row++) {
                for (int col = 0; col < pattern_cols; col++) {
                    object_points.push_back(cv::Point3f(col * dimmension, row * dimmension, 0.0));
                }
            }

            // 点群情報を保存する
            m_ObjectPoints.push_back(object_points);
            m_ImagePoints[0].push_back(image_points[0]);
            m_ImagePoints[1].push_back(image_points[1]);
            m_ui->CaptureList->addItem(new QListWidgetItem(icon, tr("%1x%2, %3 mm").arg(pattern_cols).arg(pattern_rows).arg(dimmension)));
        }
    }

    // 歪みを補正して表示する
    for (int side = 0; side < 2; side++) {
        m_Undistorted[side]->setImage(m_Undistort.undistort(original[side], side, true));
        m_Original[side]->setImage(original[side]);
    }
}

void VideoCalibrationThread::clearAllPoints(void) {
    m_ObjectPoints.clear();
    m_ImagePoints[0].clear();
    m_ImagePoints[1].clear();
    m_ui->CaptureList->clear();
}

void VideoCalibrationThread::applyCalibration(void) {
    if (m_ObjectPoints.empty() || m_ImagePoints[0].empty() || m_ImagePoints[1].empty()) {
        return;
    }

    QSize size = m_VideoInput->sourceResolution();
    m_Undistort.calibrate(size.width() / 2, size.height(), m_ObjectPoints, m_ImagePoints[0], m_ImagePoints[1]);
}

void VideoCalibrationThread::saveCalibration(void) {
    m_Undistort.save();
}
