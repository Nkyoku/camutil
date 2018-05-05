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

    /*// OutputViewにあるウィジェットを検索する
    // 無い場合はnullptrを返す
    //QWidget* findOutputWidget(const QString &name) const;

    // ウィジェットをOutputViewに追加する
    // すでに同名のウィジェットが存在する場合はそれを返して与えられたウィジェットを削除する
    //QWidget* addOutputWidget(QWidget *widget, const QString &name, bool wide = false);

    // ウィジェットをOutputViewに追加する
    // すでに同名のウィジェットが存在する場合はそれを返して与えられたウィジェットを削除する
    template<class T>
    T* addOutputWidgetGeneric(T *widget, const QString &name, bool wide = false) {
        return reinterpret_cast<T*>(addOutputWidget(widget, name, wide));
    }

    // ウィジェットをOutputViewから削除する
    void destroyOutputWidget(const QString &name);

    // OutputViewからすべてのウィジェットを削除する
    void destroyAllOutputWidgets(void);*/





private:
    // 設定ファイル名
    static const char kConfigPath[];

    // 映像出力を横に並べる最大数
    static const int kVideoOutputColumns = 2;

    // Qt Designerで作成したUI
    Ui_CamUtilWindow *m_ui;

    // ソース選択ダイアログ
    SourceDialog m_SourceDialog;

    // 映像入力
    VideoInput m_VideoInput;
    
    // OutputViewのレイアウト
    QGridLayout *m_OutputViewLayout = nullptr;

    // 映像処理スレッド
    VideoThread *m_VideoThread = nullptr;

    // ソースを開く
    Q_SLOT bool openSource(void);

    // ソースを閉じる
    Q_SLOT void closeSource(void);

    // カメラの接続状態に応じてUIオブジェクトの有効・無効を切り替える
    void setObjectsState(bool is_opened, bool is_seekable);

    // ウィジェットの全ての子ウィジェットとレイアウトを削除する
    static void destroyAllWidgets(QWidget *parent);
};
