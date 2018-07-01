#pragma once

#include <QtWidgets/QDialog>

QT_FORWARD_DECLARE_CLASS(Ui_CalibrationConvert);

class CalibrationConvertDialog : public QDialog {
	Q_OBJECT

public:
    explicit CalibrationConvertDialog(QWidget *parent = nullptr);

    // 現在の解像度を設定する
    void setCurrentResolution(int width, int height);
    
    // 変換先の解像度を取得する
    QSize targetResolution(void) const;

    // 保存ボタンが押された
    Q_SIGNAL void savePushed(int width, int height);

private:
    // ダイアログのUI
    Ui_CalibrationConvert *m_ui;
    
    // アスペクト比を表示する
    static QString calculateAspectRatio(int width, int height);
};
