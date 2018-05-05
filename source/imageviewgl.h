#pragma once

#include <QtWidgets/QOpenGLWidget>
#include <QMutex>
#include <QOpenGLFunctions>
#include <opencv2/core.hpp>
#include "qtmat.h"

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram);
QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)
QT_FORWARD_DECLARE_CLASS(QOpenGLBuffer)

class ImageViewGl : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT

public:
    // コンストラクタ
    explicit ImageViewGl(QWidget *parent = nullptr);

    // デストラクタ
    ~ImageViewGl();

    // 表示画像を設定する
    // 画像は次の再描画時に更新される
    Q_SLOT void setImage(const cv::Mat &image);


protected:
    // OpenGLの初期化時に呼ばれる
    void initializeGL(void) override;

    // リサイズ時に呼ばれる
    void resizeGL(int width, int height) override;

    // 描画時に呼ばれる
    void paintGL(void) override;





private:
    // m_Imageへのアクセスを調停するミューテックス
    QMutex m_Mutex;

    // 表示画像を保持するテクスチャ
    QOpenGLTexture *m_Texture = nullptr;

    // 表示画像を保持するcv::Mat
    cv::Mat m_Image;

    // 表示画像が更新されたことを示すフラグ
    bool m_UpdateFlag = false;
};
