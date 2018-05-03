#include "camutilwindow.h"
#include "ui_camutilwindow.h"
#include "imageviewgl.h"
#include <QEventLoop>
#include <QFileInfo>
#include <QSettings>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QBoxLayout>
#include <QtWidgets/QMessageBox>

#include <opencv2/core.hpp>

#include <QDebug>

#include <QTimer>


const char CamUtilWindow::kConfigPath[] = "config.ini";

CamUtilWindow::CamUtilWindow(QWidget *parent)
	: QMainWindow(parent)
	, m_ui(), m_SourceDialog(this)
{
    // UIを生成する
	m_ui = new Ui_CamUtilWindow;
	m_ui->setupUi(this);

    // 映像出力のためのレイアウトをVideoOutput内に生成する
    m_OutputViewLayout = new QGridLayout;
    m_OutputViewLayout->setContentsMargins(0, 0, 0, 0);
    m_ui->OutputView->setLayout(m_OutputViewLayout);


    connect(m_ui->SourceSelect, &QPushButton::clicked, &m_SourceDialog, &QDialog::exec);
	connect(m_ui->SourceOpen, &QPushButton::clicked, this, &CamUtilWindow::openSource);
	connect(m_ui->SourceClose, &QPushButton::clicked, this, &CamUtilWindow::closeSource);
    connect(m_ui->SourcePlayPause, &QPushButton::clicked, &m_VideoInput, &VideoInput::startOrPausePlaying);
    connect(m_ui->SourceStop, &QPushButton::clicked, &m_VideoInput, &VideoInput::stopPlaying);




    

    connect(m_ui->SourceTime1, QOverload<int>::of(&QSpinBox::valueChanged), [&](int frame_number) {
        m_VideoInput.setFrameNumber(frame_number);
    });
    connect(m_ui->SourceTime2, QOverload<int>::of(&QSpinBox::valueChanged), [&](int frame_number) {
        m_VideoInput.setFrameNumber(m_VideoInput.sourceFrameCount() - 1 - frame_number);
    });
    connect(m_ui->SourceSlider, &QSlider::valueChanged, [&](int frame_number) {
        m_VideoInput.setFrameNumber(frame_number);
    });
    connect(&m_VideoInput, &VideoInput::frameNumberChanged, this, [&](int frame_number) {
        m_ui->SourceTime1->blockSignals(true);
        m_ui->SourceTime1->setValue(frame_number);
        m_ui->SourceTime1->blockSignals(false);
        m_ui->SourceTime2->blockSignals(true);
        m_ui->SourceTime2->setValue(m_VideoInput.sourceFrameCount() - 1 - frame_number);
        m_ui->SourceTime2->blockSignals(false);
        m_ui->SourceSlider->blockSignals(true);
        m_ui->SourceSlider->setValue(frame_number);
        m_ui->SourceSlider->blockSignals(false);
    }, Qt::QueuedConnection);
    connect(&m_VideoInput, &VideoInput::frameCountChanged, this, [&](int frame_count) {
        m_ui->SourceTime1->blockSignals(true);
        m_ui->SourceTime1->setMaximum(frame_count - 1);
        m_ui->SourceTime1->blockSignals(false);
        m_ui->SourceTime2->blockSignals(true);
        m_ui->SourceTime2->setMaximum(frame_count - 1);
        m_ui->SourceTime2->blockSignals(false);
        m_ui->SourceSlider->blockSignals(true);
        m_ui->SourceSlider->setMaximum(frame_count - 1);
        m_ui->SourceSlider->blockSignals(false);
    });

	//connect(&m_CaptureDevice, &PS4Eye::captured, this, &TheSight::UpdateImage, Qt::QueuedConnection);



	/*connect(m_ui.VerticalOffsetInput, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), [&](int value){
		if (m_Vidsenso != nullptr){
			m_Vidsenso->setVerticalOffset(value);
		}
	});*/

	/*connect(m_ui.BrightnessSlider, &QSlider::valueChanged, [&](int value){
		m_ui.BrightnessLabel->setText(QString::number(value));
		//m_Camera.setBrightness(value);
		m_Camera.set(cv::CAP_PROP_BRIGHTNESS, value);
	});
	connect(m_ui.GainSlider, &QSlider::valueChanged, [&](int value){
		m_ui.GainLabel->setText(QString::number(value));
		//m_Camera.setGain(value);
		m_Camera.set(cv::CAP_PROP_GAIN, value);
	});
	connect(m_ui.ExposureSlider, &QSlider::valueChanged, [&](int value){
		m_ui.ExposureLabel->setText(QString::number(value));
		//m_Camera.setExposure(false, value);
		m_Camera.set(cv::CAP_PROP_EXPOSURE, value);
	});*/
	

    QTimer *timer = new QTimer(this);
    timer->setInterval(100);
    connect(timer, &QTimer::timeout, [&]() {
        if (m_VideoThread != nullptr) {
            double framerate = m_VideoThread->processingFramerate();
            this->setWindowTitle(tr("FPS : %1").arg(framerate));
        }
    });
    timer->start();
	
    restoreSettings();
    setObjectsState(false, false);
}

CamUtilWindow::~CamUtilWindow(){
	closeSource();

    saveSettings();
}

void CamUtilWindow::restoreSettings(void) {
    QSettings settings(kConfigPath, QSettings::IniFormat);
    restoreState(settings.value("WindowState").toByteArray());
    m_ui->Splitter->restoreState(settings.value(tr("SplitterState")).toByteArray());


    m_SourceDialog.restoreSettings(settings);
}

void CamUtilWindow::saveSettings(void) const {
    QSettings settings(kConfigPath, QSettings::IniFormat);
    settings.setValue("WindowState", saveState());
    settings.setValue("SplitterState", m_ui->Splitter->saveState());


    m_SourceDialog.saveSettings(settings);
}

QWidget* CamUtilWindow::findOutputWidget(const QString &name) const {
    if (name.isEmpty() == true) {
        return nullptr;
    }
    int count = m_OutputViewLayout->count();
    for (int index = 0; index < count; index++) {
        QGroupBox *groupbox = reinterpret_cast<QGroupBox*>(m_OutputViewLayout->itemAt(index)->widget());
        if ((groupbox != nullptr) && (name == groupbox->title())) {
            QLayoutItem *item = groupbox->layout()->itemAt(0);
            if (item != nullptr) {
                return item->widget();
            }
        }
    }
    return nullptr;
}

QWidget* CamUtilWindow::addOutputWidget(QWidget *widget, const QString &name, bool wide) {
    QWidget *existing_widget;
    existing_widget = findOutputWidget(name);
    if (existing_widget != nullptr) {
        return existing_widget;
    } else {
        QGroupBox *new_groupbox = new QGroupBox;
        QVBoxLayout *new_layout = new QVBoxLayout;
        new_groupbox->setTitle(name);
        new_groupbox->setLayout(new_layout);
        new_layout->addWidget(widget);
        widget->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

        // 空いているセルを探す
        int row_count = m_OutputViewLayout->rowCount();
        if (wide == false) {
            int row, col = 0;
            for (row = 0; row < row_count; row++) {
                for (; col < kVideoOutputColumns; col++) {
                    if (m_OutputViewLayout->itemAtPosition(row, col) == nullptr) {
                        break;
                    }
                }
                if (col != kVideoOutputColumns) {
                    break;
                }
                col = 0;
            }
            m_OutputViewLayout->addWidget(new_groupbox, row, col);
        } else {
            int row;
            for (row = 0; row < row_count; row++) {
                int col;
                for (col = 0; col < kVideoOutputColumns; col++) {
                    if (m_OutputViewLayout->itemAtPosition(row, col) != nullptr) {
                        break;
                    }
                }
                if (col == kVideoOutputColumns) {
                    break;
                }
            }
            m_OutputViewLayout->addWidget(new_groupbox, row, 0, 1, kVideoOutputColumns);
        }
        return widget;
    }
}

void CamUtilWindow::destroyOutputWidget(const QString &name) {
    if (name.isEmpty() == true) {
        return;
    }
    int count = m_OutputViewLayout->count();
    for (int index = 0; index < count; index++) {
        QGroupBox *groupbox = reinterpret_cast<QGroupBox*>(m_OutputViewLayout->itemAt(index)->widget());
        if ((groupbox != nullptr) && (name == groupbox->title())) {
            QLayoutItem *item = m_OutputViewLayout->takeAt(index);
            delete item->widget();
            delete item;

            // ウィジェットによっては連続削除時にイベントループを挟む必要がある
            QEventLoop event_loop;
            event_loop.processEvents(QEventLoop::ExcludeUserInputEvents);
        }
    }
}

void CamUtilWindow::destroyAllOutputWidgets(void) {
    QEventLoop event_loop;
    QLayoutItem *item;
    while ((item = m_OutputViewLayout->takeAt(0)) != nullptr) {
        delete item->widget();
        delete item;

        // ウィジェットによっては連続削除時にイベントループを挟む必要がある
        event_loop.processEvents(QEventLoop::ExcludeUserInputEvents);
    }
}

bool CamUtilWindow::openSource(void){
    closeSource();
    do {
        // ソースをVideoInputで開く
        QString status_message;
        auto type = m_SourceDialog.sourceType();
        if (type == VideoInput::kUvc) {
            int id = m_SourceDialog.sourceId();
            QSize resolution = m_SourceDialog.sourceResolution();
            double framerate = m_SourceDialog.sourceFramerate();
            if (m_VideoInput.openUvcCamera(id, resolution, framerate) == false) {
                QMessageBox::warning(this, tr("Error"), tr("Unable to open UVC device ID %1").arg(id));
                break;
            }
            status_message = tr("UVC ID %1").arg(id);
        } else if (type == VideoInput::kMovie) {
            QString path = m_SourceDialog.sourcePath();
            if (m_VideoInput.openMovie(path) == false) {
                QMessageBox::warning(this, tr("Error"), tr("Unable to open movie file %1").arg(path));
                break;
            }
            QFileInfo fileinfo(path);
            status_message = tr("Movie '%1'").arg(fileinfo.fileName());
        } else if (type == VideoInput::kSequence) {
            QString path = m_SourceDialog.sourcePath();
            double framerate = m_SourceDialog.sourceFramerate();
            if (m_VideoInput.openSequence(path, framerate) == false) {
                QMessageBox::warning(this, tr("Error"), tr("Unable to open sequence file %1").arg(path));
                break;
            }
            QFileInfo fileinfo(path);
            status_message = tr("Sequence '%1'").arg(fileinfo.fileName());
        } else {
            Q_ASSERT(false);
        }

        // ステータスバーにソースの情報を表示する
        QSize resolution = m_VideoInput.sourceResolution();
        double framerate = m_VideoInput.sourceFramerate();
        m_ui->StatusBar->showMessage(tr("Source : %1, %2x%3 %4 fps").arg(status_message).arg(resolution.width()).arg(resolution.height()).arg(framerate));

        // 映像処理スレッドを起動
        m_VideoThread = new VideoThread(&m_VideoInput);
        m_VideoThread->initialize(this);
        m_VideoThread->start();

        setObjectsState(true, m_VideoInput.isSeekable());
        return true;
    } while (false);
    closeSource();
	return false;
}

void CamUtilWindow::closeSource(void) {
    if (m_VideoThread != nullptr) {
        delete m_VideoThread;
        m_VideoThread = nullptr;
    }
    if (m_VideoInput.isOpened() == true) {
        m_VideoInput.close();
        m_ui->StatusBar->showMessage(QString());
        setObjectsState(false, false);
    }
    destroyAllOutputWidgets();
}

void CamUtilWindow::setObjectsState(bool is_opened, bool is_seekable) {
	m_ui->SourceSelect->setEnabled(!is_opened);
    m_ui->SourceOpen->setEnabled(!is_opened);
    m_ui->SourceClose->setEnabled(is_opened);
    m_ui->SourcePlayPause->setEnabled(is_opened);
    m_ui->SourceStop->setEnabled(is_opened);
    m_ui->SourceTime1->setEnabled(is_opened && is_seekable);
    m_ui->SourceTime2->setEnabled(is_opened && is_seekable);
    m_ui->SourceSlider->setEnabled(is_opened && is_seekable);
}

