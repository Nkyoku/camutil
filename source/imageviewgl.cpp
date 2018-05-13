#include "imageviewgl.h"
#include <QMetaObject>
#include <QOpenGLTexture>

ImageViewGl::ImageViewGl(QWidget *parent)
    : QOpenGLWidget(parent), QOpenGLFunctions()
{

}

ImageViewGl::~ImageViewGl() {
    delete m_Texture;
}

void ImageViewGl::convertBgrToRgb(void) {
    m_BgrFlag = true;
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
            if (m_BgrFlag == false) {
                switch (m_Image.channels()) {
                case 1:
                    pixel_format = QOpenGLTexture::Luminance;
                    break;
                case 2:
                    pixel_format = QOpenGLTexture::RG;
                    break;
                case 3:
                    pixel_format = QOpenGLTexture::RGB;
                    break;
                case 4:
                    pixel_format = QOpenGLTexture::RGBA;
                    break;
                default:
                    Q_ASSERT(false);
                }
            } else {
                switch (m_Image.channels()) {
                case 1:
                    pixel_format = QOpenGLTexture::Luminance;
                    break;
                case 2:
                    pixel_format = QOpenGLTexture::RG;
                    break;
                case 3:
                    pixel_format = QOpenGLTexture::BGR;
                    break;
                case 4:
                    pixel_format = QOpenGLTexture::BGRA;
                    break;
                default:
                    Q_ASSERT(false);
                }
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
            m_Texture->setData(0, 0, pixel_format, pixel_type, m_Image.data);
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

void ImageViewGl::setImage(const cv::Mat &image) {
    m_Mutex.lock();
    image.copyTo(m_Image);
    m_UpdateFlag = true;
    m_Mutex.unlock();
}
