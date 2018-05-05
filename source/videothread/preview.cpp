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
    QGridLayout *grid_layout = new QGridLayout;
    parent->setLayout(grid_layout);

    m_Left = new ImageViewGl;
    m_Left->convertBgrToRgb();
    grid_layout->addWidget(m_Left);

    m_Right = new ImageViewGl;
    m_Right->convertBgrToRgb();
    grid_layout->addWidget(m_Right);

    connect(this, &VideoThread::update, m_Left, QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
    connect(this, &VideoThread::update, m_Right, QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
}

void VideoPreviewThread::processImage(const cv::Mat &input_image) {
    int width = input_image.cols;
    int height = input_image.rows;
    m_Left->setImage(cv::Mat(input_image, cv::Rect(0, 0, width / 2, height)));
    m_Right->setImage(cv::Mat(input_image, cv::Rect(width / 2, 0, width / 2, height)));
}
