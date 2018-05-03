#pragma once

#include <QThread>
#include <opencv2/core.hpp>
#include "videoinput.h"

QT_FORWARD_DECLARE_CLASS(CamUtilWindow);
QT_FORWARD_DECLARE_CLASS(ImageViewGl);

class VideoThread : public QObject{
	Q_OBJECT

public:
    // コンストラクタ
    VideoThread(VideoInput *video_input, QObject *parent = nullptr);

    // デストラクタ
	~VideoThread();

    // 初期化する
    void initialize(CamUtilWindow *camutil);

	// スレッドを開始する
	Q_SLOT void start(QThread::Priority priority = QThread::InheritPriority);

	// スレッドを終了する
	Q_SLOT void quit(void);

    // 1フレームあたりの処理時間を取得する [s]
    // 画像処理にかかる時間のみを返す
    double proccessingTime(void) const {
        return m_ProcessingTime;
    }

    // 処理フレームレートを取得する [Hz]
    double processingFramerate(void) const {
        return m_ProcessingFramerate;
    }

    // 処理が終わったときに呼ばれるシグナル
    Q_SIGNAL void update(void);

private:
	// フレームの読み込みに失敗したときのタイムアウト[ms]
	static const int TIMEOUT = 100;

	// スレッドオブジェクト
	QThread m_Thread;

	// スレッドを終了させるフラグ
	volatile bool m_ExitFlag = false;

    // 映像入力
    VideoInput *m_VideoInput = nullptr;

    // 処理時間 [s]
    double m_ProcessingTime = 0;

    // 処理フレームレート [Hz]
    double m_ProcessingFramerate = 0.0;

    ImageViewGl *m_Left, *m_Right;
    ImageViewGl *m_Left2, *m_Right2;


	// 画像処理を行うスレッドで実行される
	Q_SLOT void doWork(void);

    // 画像処理の本体
    void processImage(const cv::Mat &input_image);

	/*// Census変換を5x5のウィンドウを用いて行う
	static void doCensus5x5Transform(cv::Mat &src, cv::Mat &dst);
	
	// Census変換を9x1のウィンドウを用いて行う
	static void doCensus9x1Transform(cv::Mat &src, cv::Mat &dst);

	// Census変換を33x1のウィンドウを用いて行う
	static void doCensus33x1Transform(cv::Mat &src, cv::Mat &dst);*/



};
