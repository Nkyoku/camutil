#pragma once

#include <QtWidgets/QDialog>
#include <QSettings>
#include "videoinput.h"

class Ui_SourceDialog;

class SourceDialog : public QDialog {
    Q_OBJECT

public:
    // コンストラクタ
    SourceDialog(QWidget *parent = nullptr);

    // 設定ファイルから状態を復元する
    void restoreSettings(const QSettings &settings);

    // 設定ファイルに状態を保存する
    void saveSettings(QSettings &settings) const;

    // 選択されているソースの種類を取得する
    VideoInput::SourceType sourceType(void) const;

    // UVCのIDを取得する
    int sourceId(void) const;

    // ソースの解像度を取得する
    QSize sourceResolution(void) const;

    // ソースのフレームレートを取得する
    double sourceFramerate(void) const;

    // 動画あるいは連番画像のパスを取得する
    QString sourcePath(void) const;

    // ダイアログを閉じる
    Q_SLOT virtual void done(int r) override;

private:
    // ページ番号
    enum PageNumber {
        kUvcPageNumber = 0,
        kMoviePageNumber = 1,
        kSequencePageNumber = 2
    };

    // UVCで選択できる解像度
    enum UvcResolution {
        kResolution2K2,
        kResolution1080p,
        kResolution720p,
        kResolutionWvga
    };

    // UVCで選択できるフレームレート
    enum UvcFramerate {
        kFramerate15fps = 15,
        kFramerate30fps = 30,
        kFramerate60fps = 60,
        kFramerate100fps = 100
    };

    // Qt Designerで作成したUI
    Ui_SourceDialog *m_ui;

    // 保存された設定
    struct State {
        int Type;
        int UvcId;
        int UvcResolution;
        int UvcFramerate;
        QString MoviePath;
        QString SequencePath;
        double SequenceFramerate;
    } m_State;

    // 動画ファイルを選択する
    Q_SLOT void selectMovieFile(void);

    // 連番画像の始めの画像ファイルを選択する
    Q_SLOT void selectFirstSequenceFile(void);

    // 現在の状態をm_Stateに保存する
    void saveState(void);

    // 現在の状態を破棄してm_Stateに戻す
    void discardState(void);

};
