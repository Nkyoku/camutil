﻿#include "imageviewgl.h"
#include <QMetaObject>
#include <QMouseEvent>
#include <QOpenGLTexture>
#include <QOpenGLPixelTransferOptions>

ImageViewGl::ImageViewGl(QWidget *parent)
    : QOpenGLWidget(parent), QOpenGLFunctions()
{
    m_PixelTransferOptions = new QOpenGLPixelTransferOptions();
    m_PixelTransferOptions->setAlignment(1);
}

ImageViewGl::~ImageViewGl() {
    delete m_Texture;
}

void ImageViewGl::setMirror(bool vertical) {
    m_MirrorVertical = vertical;
}

void ImageViewGl::convertBgrToRgb(bool enable) {
    m_BgrFlag = enable;
}

void ImageViewGl::useMouse(bool enable) {
    m_MouseNotify = enable;
    setMouseTracking(enable);
}

cv::Size ImageViewGl::optimumImageSize(int original_width, int original_height) {
    double ratio_image = static_cast<double>(original_width) / static_cast<double>(original_height);
    double ratio_widget = static_cast<double>(width()) / static_cast<double>(height());
    double optimum_width, optimum_height;
    if (ratio_image < ratio_widget) {
        // 表示画像の縦幅をウィジェットに合わせる
        optimum_width = ratio_image / ratio_widget * width();
        optimum_height = height();
    } else {
        // 表示画像の横幅をウィジェットに合わせる
        optimum_width = width();
        optimum_height = ratio_widget / ratio_image * height();
    }
    return cv::Size(static_cast<int>(round(optimum_width)), static_cast<int>(round(optimum_height)));
}

void ImageViewGl::initializeGL(void) {
    initializeOpenGLFunctions();
    glClearColor(0.0f, 0.5f, 0.5f, 1.0f);
}

void ImageViewGl::resizeGL(int width, int height) {
    glViewport(0, 0, width, height);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
}

void ImageViewGl::paintGL(void) {
    glClear(GL_COLOR_BUFFER_BIT);

    // 表示画像が更新されたか確認する
    if (m_UpdateFlag == true) {
        m_Mutex.lock();
        m_UpdateFlag = false;

        // m_Imageをm_Textureに変換する
        if (m_Image.empty() == true) {
            delete m_Texture;
            m_Texture = nullptr;
        } else {
            if (m_Image.isContinuous() == false) {
                Q_ASSERT(false);
            }
            if ((m_Texture == nullptr) ||
                (m_Texture->width() != m_Image.cols) || (m_Texture->height() != m_Image.rows)) {
                delete m_Texture;
                m_Texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
                m_Texture->setSize(m_Image.cols, m_Image.rows);
                m_Texture->setFormat(QOpenGLTexture::RGBA8_UNorm);
                m_Texture->allocateStorage(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8);
                m_Texture->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
                m_Texture->setWrapMode(QOpenGLTexture::ClampToEdge);
            }
            QOpenGLTexture::PixelFormat pixel_format;
            switch (m_Image.channels()) {
            case 1:
                pixel_format = QOpenGLTexture::Luminance;
                break;
            case 2:
                pixel_format = QOpenGLTexture::RG;
                break;
            case 3:
                pixel_format = (m_BgrFlag == true) ? QOpenGLTexture::BGR : QOpenGLTexture::RGB;
                break;
            case 4:
                pixel_format = (m_BgrFlag == true) ? QOpenGLTexture::BGRA : QOpenGLTexture::RGBA;
                break;
            default:
                Q_ASSERT(false);
            }
            QOpenGLTexture::PixelType pixel_type;
            switch (m_Image.depth()) {
            case CV_8U:
                pixel_type = QOpenGLTexture::UInt8;
                break;
            case CV_8S:
                pixel_type = QOpenGLTexture::Int8;
                break;
            case CV_16U:
                pixel_type = QOpenGLTexture::UInt16;
                break;
            case CV_16S:
                pixel_type = QOpenGLTexture::Int16;
                break;
            case CV_32S:
                pixel_type = QOpenGLTexture::Int32;
                break;
            case CV_32F:
                pixel_type = QOpenGLTexture::Float32;
                break;
            default:
                Q_ASSERT(false);
            }
            m_Texture->setData(0, 0, pixel_format, pixel_type, m_Image.data, m_PixelTransferOptions);
        }

        m_Mutex.unlock();
    }
    
    // 表示画像があれば描画する
    if (m_Texture != nullptr) {
        // アスペクト比を調整する
        double normalized_width, normalized_height;
        double ratio_image = static_cast<double>(m_Image.cols) / static_cast<double>(m_Image.rows);
        double ratio_widget = static_cast<double>(width()) / static_cast<double>(height());
        if (ratio_image < ratio_widget) {
            // 表示画像の縦幅をウィジェットに合わせる
            normalized_width = ratio_image / ratio_widget;
            normalized_height = 1.0;
        } else {
            // 表示画像の横幅をウィジェットに合わせる
            normalized_width = 1.0;
            normalized_height = ratio_widget / ratio_image;
        }
        if (m_MirrorVertical == true) {
            normalized_height = -normalized_height;
        }

        // 描画する
        glEnable(GL_TEXTURE_2D);
        m_Texture->bind();
        glBegin(GL_QUADS);
        glTexCoord2d(0.0, 1.0);
        glVertex3d(-normalized_width, -normalized_height, 0.0);
        glTexCoord2d(1.0, 1.0);
        glVertex3d(normalized_width, -normalized_height, 0.0);
        glTexCoord2d(1.0, 0.0);
        glVertex3d(normalized_width, normalized_height, 0.0);
        glTexCoord2d(0.0, 0.0);
        glVertex3d(-normalized_width, normalized_height, 0.0);
        glEnd();
        glDisable(GL_TEXTURE_2D);
    }
}

void ImageViewGl::mouseMoveEvent(QMouseEvent *event) {
    if (m_MouseNotify == true) {
        QPoint point = toImagePosition(event);
        emit mouseMoved(point.x(), point.y());
    }
}

void ImageViewGl::mousePressEvent(QMouseEvent *event) {
    if (m_MouseNotify == true) {
        QPoint point = toImagePosition(event);
        emit mousePressed(point.x(), point.y(), event->button());
    }
}

void ImageViewGl::setImage(const cv::Mat &image) {
    m_Mutex.lock();
    image.copyTo(m_Image);
    m_UpdateFlag = true;
    m_Mutex.unlock();
}

QPoint ImageViewGl::toImagePosition(QMouseEvent *event) {
    QPoint point = event->pos();
    double widget_x = static_cast<double>(point.x()) / static_cast<double>(width());
    double widget_y = static_cast<double>(point.y()) / static_cast<double>(height());

    double normalized_width, normalized_height;
    double ratio_image = static_cast<double>(m_Image.cols) / static_cast<double>(m_Image.rows);
    double ratio_widget = static_cast<double>(width()) / static_cast<double>(height());
    if (ratio_image < ratio_widget) {
        // 表示画像の縦幅をウィジェットに合わせる
        normalized_width = ratio_image / ratio_widget;
        normalized_height = 1.0;
    } else {
        // 表示画像の横幅をウィジェットに合わせる
        normalized_width = 1.0;
        normalized_height = ratio_widget / ratio_image;
    }

    int image_x = static_cast<int>(round(((widget_x - 0.5) / normalized_width + 0.5) * m_Image.cols));
    int image_y = static_cast<int>(round(((widget_y - 0.5) / normalized_height + 0.5) * m_Image.rows));
    return QPoint(image_x, image_y);
}
