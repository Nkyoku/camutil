#pragma once

#include <QThread>
#include <QSettings>
#include <QtWidgets/QWidget>
#include <opencv2/core.hpp>
#include "videoinput.h"

class VideoThread : public QObject{
	Q_OBJECT

public:
    // コンストラクタ
    VideoThread(VideoInput *video_input);

    // デストラクタ
	virtual ~VideoThread();

    // 初期化のために一度だけ実行される
    // parentにGUIを配置するためのタブのページへのポインタが格納される
    // 返値としてタブのタイトルを返す必要がある
    virtual QString initializeOnce(QWidget *parent) = 0;

    // スレッド開始前に初期化する
    // 画像などを表示するためのウィジェットへのポインタが格納される
    virtual void initialize(QWidget *parent) = 0;

	// スレッドを開始する
	Q_SLOT void startThread(QThread::Priority priority = QThread::InheritPriority);

	// スレッドを終了する
	Q_SLOT void quitThread(void);

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

    // 設定ファイルから状態を復元する
    virtual void restoreSettings(const QSettings &settings) {};

    // 設定ファイルに状態を保存する
    virtual void saveSettings(QSettings &settings) const {};

protected:
    // 画像処理の本体
    virtual void processImage(const cv::Mat &input_image) = 0;

private:
	// フレームの読み込みに失敗したときのタイムアウト[ms]
	static const int TIMEOUT = 100;

	// スレッドオブジェクト
	QThread *m_Thread;

	// スレッドを終了させるフラグ
	volatile bool m_ExitFlag = false;

    // 映像入力
    VideoInput *m_VideoInput = nullptr;

    // 処理時間 [s]
    double m_ProcessingTime = 0;

    // 処理フレームレート [Hz]
    double m_ProcessingFramerate = 0.0;

	// 画像処理を行うスレッドで実行される
	Q_SLOT void doWork(void);
};
