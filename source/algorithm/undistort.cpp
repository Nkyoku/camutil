#include "undistort.h"
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

Undistort::Undistort(void) {

}

bool Undistort::load(int width, int height) {
    m_Width = width;
    m_Height = height;
    
    cv::FileStorage storage(generateFileName(m_Width, m_Height), cv::FileStorage::READ);
    if (storage.isOpened() == false) {
        destroy();
        return false;
    }

    cv::FileNode child_node[2] = { storage["Left"], storage["Right"] };
    for (int side = 0; side < 2; side++) {
        child_node[side]["CameraMatrix"] >> m_CameraMatrix[side];
        child_node[side]["DistortionCoefficients"] >> m_DistortionCoefficients[side];
        child_node[side]["RectificationMatrix"] >> m_RectificationMatrix[side];
        child_node[side]["ProjectionMatrix"] >> m_ProjectionMatrix[side];
        if (m_CameraMatrix[side].empty() || m_DistortionCoefficients[side].empty() || m_RectificationMatrix[side].empty() || m_ProjectionMatrix[side].empty()) {
            destroy();
            return false;
        }
    }
    storage["DisparityMatrix"] >> m_DisparityMatrix;
    if (m_DisparityMatrix.empty() == true) {
        destroy();
        return false;
    }

    generateMap(cv::Size(m_Width, m_Height));

    m_IsCalibrated = true;

    return true;
}

bool Undistort::save(void) {
    for (int side = 0; side < 2; side++) {
        if (m_CameraMatrix[side].empty() || m_DistortionCoefficients[side].empty() || m_RectificationMatrix[side].empty() || m_ProjectionMatrix[side].empty()) {
            return false;
        }
    }
    if (m_DisparityMatrix.empty()) {
        return false;
    }

    cv::FileStorage storage(generateFileName(m_Width, m_Height), cv::FileStorage::WRITE);
    if (storage.isOpened() == false) {
        return false;
    }

    for (int side = 0; side < 2; side++) {
        storage << ((side == 0) ? "Left" : "Right") << "{";
        storage << "CameraMatrix" << m_CameraMatrix[side];
        storage << "DistortionCoefficients" << m_DistortionCoefficients[side];
        storage << "RectificationMatrix" << m_RectificationMatrix[side];
        storage << "ProjectionMatrix" << m_ProjectionMatrix[side];
        storage << "}";
    }
    storage << "DisparityMatrix" << m_DisparityMatrix;

    return true;
}

bool Undistort::undistort(const cv::Mat &distorted_image, cv::Mat &undistorted_image, int side) const {
    if (m_Map1[0].empty() || m_Map1[1].empty() || m_Map2[0].empty() || m_Map2[1].empty()) {
        return false;
    }
    cv::remap(distorted_image, undistorted_image, m_Map1[side], m_Map2[side], cv::INTER_LINEAR);
    return true;
}

cv::Mat Undistort::undistort(const cv::Mat &distorted_image, int side, bool empty_if_uncalibrated) const {
    cv::Mat result;
    if (undistort(distorted_image, result, side) == false) {
        if (empty_if_uncalibrated == false) {
            return distorted_image.clone();
        } else {
            return cv::Mat();
        }
    }
    return result;
}

bool Undistort::reprojectImageTo3D(const cv::Mat &disparity, cv::Mat &output, bool missing_value) const {
    if (m_DisparityMatrix.empty() == true) {
        return false;
    }
    cv::reprojectImageTo3D(disparity, output, m_DisparityMatrix, missing_value);
    return true;
}

bool Undistort::reprojectPointsTo3D(const cv::Mat &disparity, cv::Mat &output) const {
    if (m_DisparityMatrix.empty() == true) {
        return false;
    }
    cv::perspectiveTransform(disparity, output, m_DisparityMatrix);
    return true;
}

bool Undistort::calibrate(int width, int height, const std::vector<std::vector<cv::Point3f>> &object_points, const std::vector<std::vector<cv::Point2f>> &image_points_left, const std::vector<std::vector<cv::Point2f>> &image_points_right) {
    if (object_points.empty() || image_points_left.empty() || image_points_right.empty()) {
        return false;
    }

    cv::Size size(width, height);

    std::vector<cv::Mat> rvecs, tvecs;
    cv::calibrateCamera(object_points, image_points_left, size, m_CameraMatrix[0], m_DistortionCoefficients[0], rvecs, tvecs);
    cv::calibrateCamera(object_points, image_points_right, size, m_CameraMatrix[1], m_DistortionCoefficients[1], rvecs, tvecs);

    cv::Mat R, T, E, F;
    cv::stereoCalibrate(object_points, image_points_left, image_points_right, m_CameraMatrix[0], m_DistortionCoefficients[0], m_CameraMatrix[1], m_DistortionCoefficients[1], size, R, T, E, F);

    cv::stereoRectify(m_CameraMatrix[0], m_DistortionCoefficients[0], m_CameraMatrix[1], m_DistortionCoefficients[1], size, R, T, m_RectificationMatrix[0], m_RectificationMatrix[1], m_ProjectionMatrix[0], m_ProjectionMatrix[1], m_DisparityMatrix);

    generateMap(size);

    return true;
}

void Undistort::destroy(void) {
    m_IsCalibrated = false;
    for (int side = 0; side < 2; side++) {
        m_CameraMatrix[side] = cv::Mat();
        m_DistortionCoefficients[side] = cv::Mat();
        m_RectificationMatrix[side] = cv::Mat();
        m_ProjectionMatrix[side] = cv::Mat();
        m_Map1[side] = cv::Mat();
        m_Map2[side] = cv::Mat();
    }
}

void Undistort::generateMap(const cv::Size &size) {
    cv::initUndistortRectifyMap(m_CameraMatrix[0], m_DistortionCoefficients[0], m_RectificationMatrix[0], m_ProjectionMatrix[0], size, CV_32FC1, m_Map1[0], m_Map2[0]);
    cv::initUndistortRectifyMap(m_CameraMatrix[1], m_DistortionCoefficients[1], m_RectificationMatrix[1], m_ProjectionMatrix[1], size, CV_32FC1, m_Map1[1], m_Map2[1]);
}

std::string Undistort::generateFileName(int width, int height) {
    std::ostringstream node_name;
    node_name << "calibration" << width << "x" << height << ".xml";
    return node_name.str();
}
