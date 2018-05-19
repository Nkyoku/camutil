#include "preview.h"
#include "../imageviewgl.h"
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>

VideoPreviewThread::~VideoPreviewThread(){
    quitThread();
}

QString VideoPreviewThread::initializeOnce(QWidget *parent) {
    QGridLayout *layout = new QGridLayout;
    parent->setLayout(layout);
    layout->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    layout->addWidget(new QLabel(tr("No properties")));
    return tr("Preview");
}

void VideoPreviewThread::initialize(QWidget *parent) {
    QSize size = m_VideoInput->sourceResolution();
    m_Undistort.load(size.width() / 2, size.height());
    
    QGridLayout *grid_layout = new QGridLayout;
    parent->setLayout(grid_layout);

    m_Left = new ImageViewGl;
    m_Left->convertBgrToRgb();
    grid_layout->addWidget(m_Left);

    m_Right = new ImageViewGl;
    m_Right->convertBgrToRgb();
    grid_layout->addWidget(m_Right, 0, 1);

    connect(this, &VideoThread::update, m_Left, QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    connect(this, &VideoThread::update, m_Right, QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
}

void VideoPreviewThread::uninitialize(void) {
    m_Undistort.destroy();
}

void VideoPreviewThread::processImage(const cv::Mat &input_image) {
    int width = input_image.cols / 2;
    int height = input_image.rows;
    m_Left->setImage(m_Undistort.undistort(cv::Mat(input_image, cv::Rect(0, 0, width, height)), 0));
    m_Right->setImage(m_Undistort.undistort(cv::Mat(input_image, cv::Rect(width, 0, width, height)), 1));
}
