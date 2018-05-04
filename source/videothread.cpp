#include "videothread.h"
#include "camutilwindow.h"
#include "imageviewgl.h"
#include <QApplication>
#include <QElapsedTimer>
#include <QEventLoop>

VideoThread::VideoThread(VideoInput *video_input, QObject *parent)
    : QObject(parent), m_Thread(nullptr), m_VideoInput(video_input)
{
	
}

VideoThread::~VideoThread(){
	quitThread();
}

void VideoThread::startThread(QThread::Priority priority){
    if (m_Thread == nullptr) {
        m_Thread = new QThread;
    }
	if ((m_Thread->isRunning() == false) && (m_Thread->isFinished() == false)){
        moveToThread(m_Thread);
        connect(m_Thread, &QThread::started, this, &VideoThread::doWork);
		m_Thread->start(priority);
	}
}

void VideoThread::quitThread(void){
	if (m_Thread != nullptr) {
        if (m_Thread->isRunning() == true) {
            m_ExitFlag = true;
            m_Thread->quit();
            QThread::msleep(100);
            m_Thread->wait();
        }
        moveToThread(QApplication::instance()->thread());
        delete m_Thread;
        m_Thread = nullptr;
        m_ExitFlag = false;
	}
}

void VideoThread::doWork(void){
    QEventLoop event_loop;
    QElapsedTimer elapsed_timer;
    bool use_fps_limitter = m_VideoInput->isSeekable();
    double adjuster_setting = 1000.0 / m_VideoInput->sourceFramerate();
    cv::Mat input_image;

    qDebug("thread started");
    elapsed_timer.start();
	while (m_ExitFlag == false){
        qint64 interval = elapsed_timer.restart();
        event_loop.processEvents();

		// フレームを読み込む
        if (m_VideoInput->readFrame(input_image) == false) {
            QThread::msleep(TIMEOUT);
            continue;
        }

        // 処理を行う
        // 前後の時刻を参照して計算時間を求める
        qint64 start_time = elapsed_timer.elapsed();
        processImage(input_image);
        emit update();
        qint64 end_time = elapsed_timer.elapsed();
        m_ProcessingTime = (end_time - start_time) * 0.001;

        // シーク可能な映像入力はウェイト無しでreadFrame()が完了するため
        // フレームレートを調整するためにウェイトを挿入する
        if (use_fps_limitter == true) {
            double elapsed_time = end_time;
            int wait_time = static_cast<int>(round(adjuster_setting - elapsed_time));
            if (0 < wait_time) {
                QThread::msleep(wait_time);
            }
        }

        // フレームレートを計算する
        // IIRフィルタによる簡単な平滑化を行っている
        if (interval != 0) {
            m_ProcessingFramerate = ((m_ProcessingFramerate * 7.0) + 1000.0 / interval) * 0.125;
        }
	}
    qDebug("thread finished");
}
