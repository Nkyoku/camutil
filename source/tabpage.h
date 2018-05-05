#pragma once

#include <QtWidgets/QWidget>
#include "videothread.h"

class TabPage : public QWidget{
	Q_OBJECT

public:
    // コンストラクタ
    explicit TabPage(VideoThread *video_thread, QWidget *parent = nullptr) : QWidget(parent), m_VideoThread(video_thread) {}

    // 紐づけられたVideoThreadを取得する
    VideoThread* videoThread(void) {
        return m_VideoThread;
    }

private:
    // 紐づけられたVideoThread
    VideoThread *m_VideoThread;
};
