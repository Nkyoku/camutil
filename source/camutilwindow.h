#pragma once

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QLabel>
#include <QtWidgets/QGridLayout>
#include "sourcedialog.h"
#include "videoinput.h"
#include "videothread.h"
#include "imageviewgl.h"

class Ui_CamUtilWindow;

class CamUtilWindow : public QMainWindow {
    Q_OBJECT

public:
    // コンストラクタ
    explicit CamUtilWindow(QWidget *parent = nullptr);

    // デストラクタ
    ~CamUtilWindow();

    // 設定ファイルから状態を復元する
    void restoreSettings(void);

    // 設定ファイルに状態を保存する
    void saveSettings(void) const;

private:
    // 設定ファイル名
    static const char kConfigPath[];

    // Qt Designerで作成したUI
    Ui_CamUtilWindow *m_ui;

    // ソース選択ダイアログ
    SourceDialog m_SourceDialog;

    // 映像入力
    VideoInput m_VideoInput;
    
    // 実行中のVideoThread
    VideoThread *m_CurrentVideoThread = nullptr;

    // ソースを開く
    Q_SLOT bool openSource(void);

    // ソースを閉じる
    Q_SLOT void closeSource(void);

    // カメラの接続状態に応じてUIオブジェクトの有効・無効を切り替える
    void setObjectsState(bool is_opened, bool is_seekable);

    // タブにページとVideoThreadを追加する
    void addTabPage(VideoThread *video_thread);

    // ウィジェットの全ての子ウィジェットとレイアウトを削除する
    static void destroyAllWidgets(QWidget *parent);
};
