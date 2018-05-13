#include "camutilwindow.h"
#include "ui_camutilwindow.h"
#include "tabpage.h"
#include "videothread/preview.h"
#include "videothread/calibration.h"
#include "videothread/sgbm.h"
#include <QEventLoop>
#include <QFileInfo>
#include <QSettings>
#include <QTimer>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QBoxLayout>
#include <QtWidgets/QMessageBox>

const char CamUtilWindow::kConfigFilePath[] = "config.ini";

CamUtilWindow::CamUtilWindow(QWidget *parent)
	: QMainWindow(parent)
	, m_ui(), m_SourceDialog(this)
{
    // UIを生成する
	m_ui = new Ui_CamUtilWindow;
	m_ui->setupUi(this);

    // タブに各VideoThreadのGUIのページを作成する
    addTabPage(new VideoPreviewThread(&m_VideoInput));
    addTabPage(new VideoCalibrationThread(&m_VideoInput));
    addTabPage(new VideoSgbmThread(&m_VideoInput));

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

    // FPSと処理時間を描画する
    QTimer *timer = new QTimer(this);
    timer->setInterval(100);
    connect(timer, &QTimer::timeout, [&]() {
        if (m_CurrentVideoThread != nullptr) {
            double framerate = m_CurrentVideoThread->processingFramerate();
            double time = m_CurrentVideoThread->proccessingTime();
            this->setWindowTitle(tr("FPS : %1, Time : %2 ms").arg(framerate).arg(time * 1000, 0, 'f', 3));
        }
    });
    timer->start();
	
    restoreSettings();
    setObjectsState(false, false);
}

CamUtilWindow::~CamUtilWindow(){
    closeSource();
    
    saveSettings();

    // すべてのVideoThreadを削除する
    for (int index = m_ui->Tab->count(); 0 <= --index;) {
        TabPage *page = reinterpret_cast<TabPage*>(m_ui->Tab->widget(index));
        delete page->videoThread();
        m_ui->Tab->removeTab(index);
        delete page;
    }
}

void CamUtilWindow::restoreSettings(void) {
    QSettings settings(kConfigFilePath, QSettings::IniFormat);
    restoreState(settings.value("WindowState").toByteArray());
    m_ui->Splitter->restoreState(settings.value(tr("SplitterState")).toByteArray());
    m_SourceDialog.restoreSettings(settings);
    for (int index = 0, count = m_ui->Tab->count(); index < count; index++) {
        reinterpret_cast<TabPage*>(m_ui->Tab->widget(index))->videoThread()->restoreSettings(settings);
    }
}

void CamUtilWindow::saveSettings(void) const {
    QSettings settings(kConfigFilePath, QSettings::IniFormat);
    settings.setValue("WindowState", saveState());
    settings.setValue("SplitterState", m_ui->Splitter->saveState());
    m_SourceDialog.saveSettings(settings);
    for (int index = 0, count = m_ui->Tab->count(); index < count; index++) {
        reinterpret_cast<TabPage*>(m_ui->Tab->widget(index))->videoThread()->saveSettings(settings);
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

        // 選択されていないタブページを無効化
        for (int index = 0, count = m_ui->Tab->count(); index < count; index++) {
            if (m_ui->Tab->currentIndex() != index) {
                m_ui->Tab->setTabEnabled(index, false);
            }
        }

        // 選択されているVideoThreadを起動
        VideoThread *video_thread = reinterpret_cast<TabPage*>(m_ui->Tab->currentWidget())->videoThread();
        m_CurrentVideoThread = video_thread;
        video_thread->initialize(m_ui->OutputView);
        video_thread->startThread();

        setObjectsState(true, m_VideoInput.isSeekable());
        return true;
    } while (false);
    closeSource();
	return false;
}

void CamUtilWindow::closeSource(void) {
    if (m_CurrentVideoThread != nullptr) {
        m_CurrentVideoThread->quitThread();
        m_CurrentVideoThread->uninitialize();
        m_CurrentVideoThread = nullptr;
        destroyAllWidgets(m_ui->OutputView);
        delete m_ui->OutputView->layout();

        // すべてのタブページを有効化
        for (int index = 0, count = m_ui->Tab->count(); index < count; index++) {
            m_ui->Tab->setTabEnabled(index, true);
        }
    }
    if (m_VideoInput.isOpened() == true) {
        m_VideoInput.close();
        m_ui->StatusBar->showMessage(QString());
        setObjectsState(false, false);
    }
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

void CamUtilWindow::addTabPage(VideoThread *video_thread) {
    TabPage *page = new TabPage(video_thread);
    m_ui->Tab->addTab(page, video_thread->initializeOnce(page));
}

void CamUtilWindow::destroyAllWidgets(QWidget *parent) {
    QEventLoop event_loop;
    auto children = parent->findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (auto widget : children) {
        delete widget;

        // ウィジェットによっては連続削除時にイベントループを挟む必要がある
        event_loop.processEvents(QEventLoop::ExcludeUserInputEvents);
    }
    delete parent->layout();
}
