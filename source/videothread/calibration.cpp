#include "calibration.h"
#include "../imageviewgl.h"
#include "ui_calibration.h"
#include "ui_calibration_convert.h"
//#include <QtWidgets/QBoxLayout>
//#include <QtWidgets/QGridLayout>
//#include <QtWidgets/QLabel>
#include <QtWidgets/QMessageBox>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

VideoCalibrationThread::~VideoCalibrationThread(){
    quitThread();
}

QString VideoCalibrationThread::initializeOnce(QWidget *parent) {
    m_ui = new Ui_Calibration;
    m_ui->setupUi(parent);
    m_ui->CaptureList->setIconSize(QSize(kPreviewSize, kPreviewSize));
    m_ConvertDialog = new CalibrationConvertDialog(parent);
    m_ConvertDialog->setModal(true);

    connect(m_ui->Step1Selection, &QRadioButton::clicked, this, [&](bool checked) {
        if (checked == true) {
            changeCalibrationStep(1);
        }
    });
    connect(m_ui->Step2Selection, &QRadioButton::clicked, this, [&](bool checked) {
        if (checked == true) {
            changeCalibrationStep(2);
        }
    });
    connect(m_ui->Step3Selection, &QRadioButton::clicked, this, [&](bool checked) {
        if (checked == true) {
            changeCalibrationStep(3);
        }
    });
    connect(m_ui->Capture, &QPushButton::clicked, this, [&]() {
        m_CaptureCounter = kCaptureDeadline;
    });
    connect(m_ui->Calibrate, &QPushButton::clicked, this, &VideoCalibrationThread::applyCalibration, Qt::QueuedConnection);
    connect(m_ui->SaveAsDefault, &QPushButton::clicked, this, QOverload<>::of(&VideoCalibrationThread::saveCalibration), Qt::QueuedConnection);
    connect(m_ui->ConvertResolution, &QPushButton::clicked, m_ConvertDialog, [&](void) {
        if (m_Undistort.isCalibrated(0) || m_Undistort.isCalibrated(1) || m_Undistort.isStereoRectified()) {
            m_ConvertDialog->setCurrentResolution(m_Undistort.width(), m_Undistort.height());
            m_ConvertDialog->show();
        }
    });
    connect(m_ConvertDialog, &CalibrationConvertDialog::savePushed, this, QOverload<int, int>::of(&VideoCalibrationThread::saveCalibration), Qt::QueuedConnection);

    return tr("Calibration");
}

void VideoCalibrationThread::initialize(QWidget *parent) {
    uninitialize();

    QSize size = m_VideoInput->sourceResolution();
    m_Undistort.load(size.width() / 2, size.height());
    if (m_Undistort.isCalibrated(0) == true) {
        m_ui->Step1Status->setText(tr("Loaded"));
    }
    if (m_Undistort.isCalibrated(1) == true) {
        m_ui->Step2Status->setText(tr("Loaded"));
    }
    if (m_Undistort.isStereoRectified() == true) {
        m_ui->Step3Status->setText(tr("Loaded"));
    }

    // 補正前と補正後の画像を表示するImageViewGlを生成
    QGridLayout *grid_layout = new QGridLayout;
    parent->setLayout(grid_layout);
    for (int side = 0; side < 2; side++) {
        m_Original[side] = new ImageViewGl;
        m_Original[side]->convertBgrToRgb();
        grid_layout->addWidget(m_Original[side], 0, side);
        m_Undistorted[side] = new ImageViewGl;
        m_Undistorted[side]->convertBgrToRgb();
        grid_layout->addWidget(m_Undistorted[side], 1, side);
        connect(this, &VideoThread::update, m_Original[side], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
        connect(this, &VideoThread::update, m_Undistorted[side], QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    }
}

void VideoCalibrationThread::uninitialize(void) {
    m_Undistort.destroy();
    m_CaptureCounter = 0;
    clearAllPoints();
    m_ui->Step1Selection->click();
    m_ui->Step1Status->setText(tr("Not Calibrated"));
    m_ui->Step2Status->setText(tr("Not Calibrated"));
    m_ui->Step3Status->setText(tr("Not Calibrated"));
    m_ConvertDialog->close();
}

void VideoCalibrationThread::restoreSettings(const QSettings &settings) {
    int type = settings.value(tr("CalPatternType"), 0).toInt();
    if (type == 1) {
        m_ui->PatternCircle->setChecked(true);
    } else {
        m_ui->PatternChessboard->setChecked(true);
    }
    m_ui->PatternColumns->setValue(settings.value(tr("CalPatternColumns"), m_ui->PatternColumns->value()).toInt());
    m_ui->PatternRows->setValue(settings.value(tr("CalPatternRows"), m_ui->PatternRows->value()).toInt());
    m_ui->PatternDimmension->setValue(settings.value(tr("CalPatternDimmension"), m_ui->PatternDimmension->value()).toDouble());
}

void VideoCalibrationThread::saveSettings(QSettings &settings) const {
    settings.setValue(tr("CalPatternType"), m_ui->PatternCircle->isChecked() ? 1 : 0);
    settings.setValue(tr("CalPatternColumns"), m_ui->PatternColumns->value());
    settings.setValue(tr("CalPatternRows"), m_ui->PatternRows->value());
    settings.setValue(tr("CalPatternDimmension"), m_ui->PatternDimmension->value());
}

void VideoCalibrationThread::processImage(const cv::Mat &input_image) {
    int pattern_type = m_ui->PatternCircle->isChecked() ? 1 : 0;
    int pattern_cols = m_ui->PatternColumns->value();
    int pattern_rows = m_ui->PatternRows->value();
    cv::Size pattern_size(pattern_cols, pattern_rows);

    bool calibration_enabled[2];
    calibration_enabled[0] = (m_CalibrationStep == 1) || (m_CalibrationStep == 3);
    calibration_enabled[1] = (m_CalibrationStep == 2) || (m_CalibrationStep == 3);

    // 画像を取得する
    int width = input_image.cols / 2;
    int height = input_image.rows;
    cv::Mat original[2];
    original[0] = cv::Mat(input_image, cv::Rect(0, 0, width, height));
    original[1] = cv::Mat(input_image, cv::Rect(width, 0, width, height));
    cv::Mat original_clone[2];
    original[0].copyTo(original_clone[0]);
    original[1].copyTo(original_clone[1]);

    // チェスボードの交点を検出する
    std::vector<cv::Point2f> image_points[2];
    bool pattern_found = true;
    for (int side = 0; side < 2; side++) {
        if (calibration_enabled[side] == false) {
            continue;
        }
        if (pattern_type == 1) {
            // 円形パターンを探す
            pattern_found &= cv::findCirclesGrid(original[side], pattern_size, image_points[side], cv::CALIB_CB_SYMMETRIC_GRID);
        } else {
            // チェスボードパターンを探す
            pattern_found &= cv::findChessboardCorners(original[side], pattern_size, image_points[side], cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
        }
        if (pattern_found == false) {
            // 片方のカメラで完全なパターンが検知できなかったときはもう片方の処理を行わない
            break;
        }
        cv::drawChessboardCorners(original_clone[side], pattern_size, image_points[side], pattern_found);
    }

    // Captureボタンが押されたとき検出された点群情報を保存してプレビューをCaptureListに表示する
    if (0 < m_CaptureCounter) {
        m_CaptureCounter--;
        if ((pattern_found == true) && (calibration_enabled[0] || calibration_enabled[1])){
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
            cv::Mat preview_bgr, preview_rgb;
            int side = calibration_enabled[0] ? 0 : 1;
            cv::resize(original[side], preview_bgr, cv::Size(preview_width, preview_height));
            cv::cvtColor(preview_bgr, preview_rgb, cv::COLOR_BGR2RGB);
            QIcon icon(QPixmap::fromImage(QImage(preview_rgb.data, preview_rgb.cols, preview_rgb.rows, QImage::Format_RGB888)));
            
            // チェスボードの点群情報を生成する
            double dimmension = m_ui->PatternDimmension->value() * 0.001;
            std::vector<cv::Point3f> object_points;
            object_points.reserve(pattern_cols * pattern_rows);
            for (int row = 0; row < pattern_rows; row++) {
                for (int col = 0; col < pattern_cols; col++) {
                    object_points.push_back(cv::Point3f(static_cast<float>(col * dimmension), static_cast<float>(row * dimmension), 0.0f));
                }
            }

            // 点群情報を保存する
            m_ObjectPoints.push_back(object_points);
            m_ImagePoints[0].push_back(image_points[0]);
            m_ImagePoints[1].push_back(image_points[1]);
            m_ui->CaptureList->addItem(new QListWidgetItem(icon, tr("%1x%2, %3 mm").arg(pattern_cols).arg(pattern_rows).arg(dimmension * 1000)));
        }
    }

    // 歪みを補正して表示する
    for (int side = 0; side < 2; side++) {
        m_Original[side]->setImage(original_clone[side]);
        m_Undistorted[side]->setImage(m_Undistort.undistort(original[side], side, true));
    }
}

void VideoCalibrationThread::changeCalibrationStep(int step) {
    m_CalibrationStep = step;
    clearAllPoints();
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
    int width = size.width() / 2;
    int height = size.height();

    if ((m_CalibrationStep == 1) || (m_CalibrationStep == 2)) {
        // カメラ単独のキャリブレーションを行う
        int side = (m_CalibrationStep == 1) ? 0 : 1;
        if (m_Undistort.calibrate(side, width, height, m_ObjectPoints, m_ImagePoints[side]) == true) {
            ((side == 0) ? m_ui->Step1Status : m_ui->Step2Status)->setText(tr("OK"));
            m_ui->Step3Status->setText(tr("Not Calibrated"));
        } else {
            QMessageBox::critical(nullptr, tr("Calibration"), tr("Failed to calibrate the camera"));
        }
    } else if (m_CalibrationStep == 3) {
        // ステレオ平行化を行う
        if (m_Undistort.stereoRectify(width, height, m_ObjectPoints, m_ImagePoints[0], m_ImagePoints[1]) == true) {
            m_ui->Step3Status->setText(tr("OK"));
        } else {
            QMessageBox::critical(nullptr, tr("Calibration"), tr("Failed to do stereo rectification"));
        }
    }
}

void VideoCalibrationThread::saveCalibration(void) {
    m_Undistort.save();
}

void VideoCalibrationThread::saveCalibration(int width, int height) {
    m_Undistort.save(width, height);
}
