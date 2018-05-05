#include "lenscalibration.h"
#include "../imageviewgl.h"
#include "../ui_lenscalibration.h"
#include <QtWidgets/QBoxLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

VideoLensCalibrationThread::~VideoLensCalibrationThread(){
    quitThread();
}

QString VideoLensCalibrationThread::initializeOnce(QWidget *parent) {
    m_ui = new Ui_LensCalibration;
    m_ui->setupUi(parent);
    m_ui->CaptureList->setIconSize(QSize(kPreviewSize, kPreviewSize));

    connect(m_ui->Capture, &QPushButton::clicked, this, [&]() {
        m_IsCaptuedPushed = true;
    });
    connect(m_ui->Calibrate, &QPushButton::clicked, this, [&]() {
        m_IsCalibratePushed = true;
    });
    connect(m_ui->Clear, &QPushButton::clicked, this, [&]() {
        m_IsClearPushed = true;
    });

    return tr("Lens Cal");
}

void VideoLensCalibrationThread::initialize(QWidget *parent) {
    // 補正前と補正後の画像を表示するImageViewGlを生成
    QGridLayout *grid_layout = new QGridLayout;
    parent->setLayout(grid_layout);
    m_Original = new ImageViewGl;
    grid_layout->addWidget(m_Original);
    m_Undistort = new ImageViewGl;
    grid_layout->addWidget(m_Undistort);
    connect(this, &VideoThread::update, m_Original, QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    connect(this, &VideoThread::update, m_Undistort, QOverload<>::of(&QWidget::update), Qt::QueuedConnection);

    // 変数とUIを初期化
    m_IsCaptuedPushed = false;
    m_IsCalibratePushed = false;
    m_ObjectPoints.clear();
    m_ImagePoints.clear();
    m_CameraMatrix[0] = cv::Mat();
    m_CameraMatrix[1] = cv::Mat();
    m_DistortionCoefficients[0] = cv::Mat();
    m_DistortionCoefficients[1] = cv::Mat();
    m_ui->CaptureList->clear();
}

void VideoLensCalibrationThread::restoreSettings(const QSettings &settings) {
    m_ui->PatternColumns->setValue(settings.value(tr("LensCalPatternColumns"), m_ui->PatternColumns->value()).toInt());
    m_ui->PatternRows->setValue(settings.value(tr("LensCalPatternRows"), m_ui->PatternRows->value()).toInt());
    m_ui->PatternDimmension->setValue(settings.value(tr("LensCalPatternDimmension"), m_ui->PatternDimmension->value()).toDouble());
}

void VideoLensCalibrationThread::saveSettings(QSettings &settings) const {
    settings.setValue(tr("LensCalPatternColumns"), m_ui->PatternColumns->value());
    settings.setValue(tr("LensCalPatternRows"), m_ui->PatternRows->value());
    settings.setValue(tr("LensCalPatternDimmension"), m_ui->PatternDimmension->value());
}

void VideoLensCalibrationThread::processImage(const cv::Mat &input_image) {
    int width = input_image.cols / 2;
    int height = input_image.rows;
    int side = m_ui->LeftEye->isChecked() ? 0 : 1;
    int pattern_cols = m_ui->PatternColumns->value();
    int pattern_rows = m_ui->PatternRows->value();
    
    cv::Mat original;
    cv::cvtColor(cv::Mat(input_image, cv::Rect(side * width, 0, width, height)), original, cv::COLOR_BGR2RGB);

    // Clearボタンが押されたら点群情報を破棄する
    if (m_IsClearPushed == true) {
        m_IsClearPushed = false;
        m_ObjectPoints.clear();
        m_ImagePoints.clear();
        m_ui->CaptureList->clear();
    }

    // Calibrateボタンが押されたら今まで撮影した点群情報からカメラの歪みを推定する
    if (m_IsCalibratePushed == true) {
        m_IsCalibratePushed = false;
        if (!m_ObjectPoints.empty() && !m_ImagePoints.empty()) {
            std::vector<cv::Mat> rvecs, tvecs;
            //cv::Mat camera_matrix;
            double error = cv::calibrateCamera(m_ObjectPoints, m_ImagePoints, cv::Size(width, height), m_CameraMatrix[side], m_DistortionCoefficients[side], rvecs, tvecs);
            qDebug("calibrateCamera returns %g", error);

            //m_CameraMatrix[side] = cv::getOptimalNewCameraMatrix(camera_matrix, m_DistortionCoefficients[side], cv::Size(width, height), 1.0, cv::Size(width, height));
                

        }
    }

    // チェスボードの交点を検出する
    std::vector<cv::Point2f> image_points;
    cv::Size pattern_size(pattern_cols, pattern_rows);
    bool found = cv::findChessboardCorners(original, pattern_size, image_points, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
    cv::drawChessboardCorners(original, pattern_size, image_points, found);

    // Captureボタンが押されたとき検出された点群情報を保存してプレビューをCaptureListに表示する
    if (m_IsCaptuedPushed == true) {
        m_IsCaptuedPushed = false;
        if (static_cast<int>(image_points.size()) == (pattern_cols * pattern_rows)) {
            // 撮影された画像をプレビュー用のアイコンに変換
            int preview_width, preview_height;
            if (width < height) {
                preview_width = kPreviewSize * width / height;
                preview_height = kPreviewSize;
            } else {
                preview_width = kPreviewSize;
                preview_height = kPreviewSize * height / width;
            }
            cv::Mat preview;
            cv::resize(original, preview, cv::Size(preview_width, preview_height));
            QIcon icon(QPixmap::fromImage(QImage(preview.data, preview.cols, preview.rows, QImage::Format_RGB888)));
            
            // チェスボードの点群情報を生成
            float dimmension = static_cast<float>(m_ui->PatternDimmension->value());
            std::vector<cv::Point3f> object_points;
            object_points.reserve(pattern_cols * pattern_rows);
            for (int row = 0; row < pattern_rows; row++) {
                for (int col = 0; col < pattern_cols; col++) {
                    object_points.push_back(cv::Point3f(col * dimmension, row * dimmension, 0.0));
                }
            }

            // リストに保存
            m_ObjectPoints.push_back(object_points);
            m_ImagePoints.push_back(image_points);
            m_ui->CaptureList->addItem(new QListWidgetItem(icon, tr("%1x%2, %3 mm").arg(pattern_cols).arg(pattern_rows).arg(dimmension)));
        }
    }

    // 歪みを補正して表示する
    if (!m_CameraMatrix[side].empty() && !m_DistortionCoefficients[side].empty()) {
        cv::Mat undistort;
        cv::undistort(original, undistort, m_CameraMatrix[side], m_DistortionCoefficients[side]);
        m_Undistort->setImage(undistort);
    } else {
        m_Undistort->setImage(cv::Mat());
    }

    m_Original->setImage(original);
}
