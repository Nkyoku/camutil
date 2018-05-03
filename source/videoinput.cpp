#include "videoinput.h"
#include <opencv2/videoio.hpp>

VideoInput::VideoInput(QObject *parent) 
    : QObject(parent)
{
    m_CvCapture = new cv::VideoCapture;
}

VideoInput::~VideoInput() {
    QMutexLocker locker(&m_Mutex);
    delete m_CvCapture;
}

bool VideoInput::openUvcCamera(int index, const QSize &resolution, double framerate) {
    close();
    QMutexLocker locker(&m_Mutex);
    if (m_CvCapture->open(index) == true) {
        bool result = true;
        result &= m_CvCapture->set(cv::CAP_PROP_FRAME_WIDTH, resolution.width());
        result &= m_CvCapture->set(cv::CAP_PROP_FRAME_HEIGHT, resolution.height());
        result &= m_CvCapture->set(cv::CAP_PROP_FPS, framerate);
        if (result && openCommon(kUvc)) {
            return true;
        }
        m_CvCapture->release();
    }
    return false;
}

bool VideoInput::openMovie(const QString &path_to_file) {
    close();
    QMutexLocker locker(&m_Mutex);
    if (m_CvCapture->open(path_to_file.toStdString()) == true) {
        if (openCommon(kMovie) == true) {
            return true;
        }
        m_CvCapture->release();
    }
    return false;
}

bool VideoInput::openSequence(const QString &path_to_first_file, double framerate) {
    close();
    QMutexLocker locker(&m_Mutex);
    if (m_CvCapture->open(path_to_first_file.toStdString()) == true) {
        if (m_CvCapture->set(cv::CAP_PROP_FPS, framerate) == true) {
            if (openCommon(kSequence) == true) {
                return true;
            }
        }
        m_CvCapture->release();
    }
    return false;
}

void VideoInput::close(void) {
    QMutexLocker locker(&m_Mutex);
    m_CvCapture->release();
}

bool VideoInput::isOpened(void) const {
    return m_CvCapture->isOpened();
}

bool VideoInput::startOrPausePlaying(void) {
    if (m_IsPlaying == true) {
        pausePlaying();
        return true;
    } else {
        return startPlaying();
    }
}

bool VideoInput::startPlaying(void) {
    QMutexLocker locker(&m_Mutex);
    if (m_CvCapture->isOpened() == true) {
        m_IsPlaying = true;
        return true;
    } else {
        return false;
    }
}

void VideoInput::pausePlaying(void) {
    QMutexLocker locker(&m_Mutex);
    if (m_CvCapture->isOpened() == true) {
        m_IsPlaying = false;
    }
}

void VideoInput::stopPlaying(void) {
    QMutexLocker locker(&m_Mutex);
    if (m_CvCapture->isOpened() == true) {
        m_IsPlaying = false;
        if (m_Type != kUvc) {
            m_CvCapture->set(cv::CAP_PROP_POS_FRAMES, 0);
        }
        m_CurrentFrameNumber = 0;
        emit frameNumberChanged(0);
    }
}

int VideoInput::setFrameNumber(int frame_number) {
    QMutexLocker locker(&m_Mutex);
    if ((m_CvCapture->isOpened() == true) && (m_Type != kUvc)) {
        if (frame_number < 0) {
            frame_number = 0;
        } else if (m_FrameCount <= frame_number) {
            frame_number = m_FrameCount - 1;
        }
        m_CvCapture->set(cv::CAP_PROP_POS_FRAMES, frame_number);
        m_CurrentFrameNumber = static_cast<int>(m_CvCapture->get(cv::CAP_PROP_POS_FRAMES));
        emit frameNumberChanged(m_CurrentFrameNumber);
        return m_CurrentFrameNumber;
    }
    return -1;
}

bool VideoInput::isSeekable(void) const {
    return m_CvCapture->isOpened() && (m_Type != kUvc);
}

bool VideoInput::readFrame(cv::Mat &frame) {
    if (m_IsPlaying == true) {
        QMutexLocker locker(&m_Mutex);
        if (m_CvCapture->read(frame) == true) {
            if (m_Type != kUvc) {
                emit frameNumberChanged(m_CurrentFrameNumber++);
            }
            return true;
        } else {
            m_IsPlaying = false;
            return false;
        }
    } else {
        return false;
    }
}

bool VideoInput::openCommon(SourceType type) {
    int width = static_cast<int>(m_CvCapture->get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(m_CvCapture->get(cv::CAP_PROP_FRAME_HEIGHT));
    double framerate = m_CvCapture->get(cv::CAP_PROP_FPS);
    int frame_count = (type != kUvc) ? static_cast<int>(m_CvCapture->get(cv::CAP_PROP_FRAME_COUNT)) : 1;
    m_IsPlaying = true;
    m_Type = type;
    m_Width = width;
    m_Height = height;
    m_Framerate = framerate;
    m_FrameCount = frame_count;
    emit frameCountChanged(frame_count);
    m_CurrentFrameNumber = 0;
    emit frameNumberChanged(0);
    return true;
}
