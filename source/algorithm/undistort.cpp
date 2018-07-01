#include "undistort.h"
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

Undistort::Undistort(void) {

}

bool Undistort::load(int width_, int height_) {
    m_Width = width_;
    m_Height = height_;
    
    cv::FileStorage storage(generateFileName(m_Width, m_Height), cv::FileStorage::READ);
    if (storage.isOpened() == false) {
        destroy();
        return false;
    }

    // 補正情報を読み込む
    cv::FileNode child_node[2] = { storage["Left"], storage["Right"] };
    for (int side = 0; side < 2; side++) {
        // 読み込む
        child_node[side]["RawCameraMatrix"] >> m_RawCameraMatrix[side];
        child_node[side]["CameraMatrix"] >> m_CameraMatrix[side];
        child_node[side]["RawDistortionCoefficients"] >> m_RawDistortionCoefficients[side];
        child_node[side]["DistortionCoefficients"] >> m_DistortionCoefficients[side];
        child_node[side]["RectificationMatrix"] >> m_RectificationMatrix[side];
        child_node[side]["ProjectionMatrix"] >> m_ProjectionMatrix[side];

        // サイズチェック
        if ((m_RawCameraMatrix[side].cols != 3) || (m_RawCameraMatrix[side].rows != 3)) {
            m_RawCameraMatrix[side] = cv::Mat();
        }
        if ((m_CameraMatrix[side].cols != 3) || (m_CameraMatrix[side].rows != 3)) {
            m_CameraMatrix[side] = cv::Mat();
        }
        if ((m_RawDistortionCoefficients[side].rows != 1) ||
            ((m_RawDistortionCoefficients[side].cols != 4) && (m_RawDistortionCoefficients[side].cols != 5) && (m_RawDistortionCoefficients[side].cols != 8))) {
            m_RawDistortionCoefficients[side] = cv::Mat();
        }
        if ((m_DistortionCoefficients[side].rows != 1) ||
            ((m_DistortionCoefficients[side].cols != 4) && (m_DistortionCoefficients[side].cols != 5) && (m_DistortionCoefficients[side].cols != 8))) {
            m_DistortionCoefficients[side] = cv::Mat();
        }
        if ((m_RectificationMatrix[side].cols != 3) || (m_RectificationMatrix[side].rows != 3)) {
            m_RectificationMatrix[side] = cv::Mat();
        }
        if ((m_ProjectionMatrix[side].cols != 4) || (m_ProjectionMatrix[side].rows != 3)) {
            m_ProjectionMatrix[side] = cv::Mat();
        }

        m_IsCalibrated[side] = !m_RawCameraMatrix[side].empty() && !m_RawDistortionCoefficients[side].empty();
    }
    storage["DisparityMatrix"] >> m_DisparityMatrix;
    storage["RotationMatrix"] >> m_RotationMatrix;
    storage["TranslationVector"] >> m_TranslationVector;
    storage["RollAdjustment"] >> m_RollAdjustment;
    storage["PitchAdjustment"] >> m_PitchAdjustment;
    if ((m_DisparityMatrix.cols != 4) || (m_DisparityMatrix.rows != 4)) {
        m_DisparityMatrix = cv::Mat();
    }
    if ((m_RotationMatrix.cols != 3) || (m_RotationMatrix.rows != 3)) {
        m_RotationMatrix = cv::Mat();
    }
    if ((m_TranslationVector.cols != 1) || (m_TranslationVector.rows != 3)) {
        m_TranslationVector = cv::Mat();
    }

    m_IsStereoRectified =
        !m_CameraMatrix[0].empty() && !m_CameraMatrix[1].empty() &&
        !m_DistortionCoefficients[0].empty() && !m_DistortionCoefficients[1].empty() &&
        !m_RectificationMatrix[0].empty() && !m_RectificationMatrix[1].empty() &&
        !m_ProjectionMatrix[0].empty() && !m_ProjectionMatrix[1].empty() &&
        !m_DisparityMatrix.empty() && !m_RotationMatrix.empty() && !m_TranslationVector.empty();

    // 補正マップを生成する
    if (m_IsStereoRectified == true) {
        generateMapStereo();
    } else {
        for (int side = 0; side < 2; side++) {
            if (m_IsCalibrated[side] == true) {
                generateMap(side);
            }
        }
    }

    return true;
}

bool Undistort::save(int new_width, int new_height) {
    // いずれのキャリブレーションも行われていない場合は保存を行わない
    if (!m_IsCalibrated[0] && !m_IsCalibrated[1] && !m_IsStereoRectified) {
        return false;
    }

    // スケーリング係数を計算
    if ((new_width <= 0) || (new_height <= 0)) {
        new_width = m_Width;
        new_height = m_Height;
    }
    double scale_x = static_cast<double>(new_width) / m_Width;
    double scale_y = static_cast<double>(new_height) / m_Height;

    cv::FileStorage storage(generateFileName(new_width, new_height), cv::FileStorage::WRITE);
    if (storage.isOpened() == false) {
        return false;
    }

    for (int side = 0; side < 2; side++) {
        storage << ((side == 0) ? "Left" : "Right") << "{";
        if (m_IsCalibrated[side] == true) {
            cv::Mat raw_camera_matrix = m_RawCameraMatrix[side].clone();
            raw_camera_matrix.at<double>(0, 0) *= scale_x;
            raw_camera_matrix.at<double>(0, 2) *= scale_x;
            raw_camera_matrix.at<double>(1, 1) *= scale_y;
            raw_camera_matrix.at<double>(1, 2) *= scale_y;
            storage << "RawCameraMatrix" << raw_camera_matrix;
            storage << "RawDistortionCoefficients" << m_RawDistortionCoefficients[side];
        }
        if (m_IsStereoRectified == true) {
            cv::Mat camera_matrix = m_CameraMatrix[side].clone();
            cv::Mat projection_matrix = m_ProjectionMatrix[side].clone();
            camera_matrix.at<double>(0, 0) *= scale_x;
            camera_matrix.at<double>(0, 2) *= scale_x;
            camera_matrix.at<double>(1, 1) *= scale_y;
            camera_matrix.at<double>(1, 2) *= scale_y;
            projection_matrix.at<double>(0, 0) *= scale_x;
            projection_matrix.at<double>(0, 2) *= scale_x;
            projection_matrix.at<double>(1, 1) *= scale_y;
            projection_matrix.at<double>(1, 2) *= scale_y;
            storage << "CameraMatrix" << camera_matrix;
            storage << "DistortionCoefficients" << m_DistortionCoefficients[side];
            storage << "RectificationMatrix" << m_RectificationMatrix[side];
            storage << "ProjectionMatrix" << projection_matrix;
        }
        storage << "}";
    }
    if (m_IsStereoRectified == true) {
        storage << "DisparityMatrix" << m_DisparityMatrix;
        storage << "RotationMatrix" << m_RotationMatrix;
        storage << "TranslationVector" << m_TranslationVector;
        storage << "RollAdjustment" << m_RollAdjustment;
        storage << "PitchAdjustment" << m_PitchAdjustment;
    }

    return true;
}

bool Undistort::undistort(const cv::Mat &distorted_image, cv::Mat &undistorted_image, int side) const {
    if (m_Map1[side].empty() || m_Map2[side].empty()) {
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

bool Undistort::reprojectPointsTo3D(const std::vector<cv::Point3f> &disparities, std::vector<cv::Point3f> &output) const {
    if (m_DisparityMatrix.empty() == true) {
        return false;
    }
    cv::perspectiveTransform(disparities, output, m_DisparityMatrix);
    return true;
}

bool Undistort::projectPointsTo2D(const std::vector<cv::Point3f> &points3d, std::vector<cv::Point2f> &points2d, int side) const {
    if (m_DisparityMatrix.empty() == true) {
        return false;
    }
    cv::Mat zero = cv::Mat::zeros(3, 1, CV_32F);
    //cv::projectPoints(points3d, zero, zero, m_CameraMatrix[side], m_DistortionCoefficients[side], points2d);
    cv::projectPoints(points3d, zero, zero, m_CameraMatrix[side], cv::Mat(), points2d); // カメラからの入力画像はすでに歪み補正されている
    return true;
}

bool Undistort::calibrate(int side, int width_, int height_, const std::vector<std::vector<cv::Point3f>> &object_points, const std::vector<std::vector<cv::Point2f>> &image_points) {
    if (object_points.empty() || image_points.empty() || (object_points.size() != image_points.size())) {
        return false;
    }

    cv::Size size(m_Width, m_Height);
    if ((m_Width != width_) || (m_Height != height_)) {
        m_Width = width_;
        m_Height = height_;
        destroy();
    }

    std::vector<cv::Mat> rvecs, tvecs;
    cv::calibrateCamera(object_points, image_points, size, m_RawCameraMatrix[side], m_RawDistortionCoefficients[side], rvecs, tvecs);
    m_CameraMatrix[side] = cv::Mat();
    m_DistortionCoefficients[side] = cv::Mat();

    generateMap(side);

    m_IsCalibrated[side] = true;
    m_IsStereoRectified = false;

    return true;
}

bool Undistort::stereoRectify(int width_, int height_, const std::vector<std::vector<cv::Point3f>> &object_points, const std::vector<std::vector<cv::Point2f>> &image_points_left, const std::vector<std::vector<cv::Point2f>> &image_points_right) {
    if (object_points.empty() || image_points_left.empty() || image_points_right.empty() ||
        (object_points.size() != image_points_left.size()) || (object_points.size() != image_points_right.size())) {
        return false;
    }
    if (!m_IsCalibrated[0] || !m_IsCalibrated[1]) {
        return false;
    }
    if ((m_Width != width_) || (m_Height != height_)) {
        return false;
    }

    cv::Size size(width_, height_);
    cv::Mat E, F;
    m_RawCameraMatrix[0].copyTo(m_CameraMatrix[0]);
    m_RawCameraMatrix[1].copyTo(m_CameraMatrix[1]);
    m_RawDistortionCoefficients[0].copyTo(m_DistortionCoefficients[0]);
    m_RawDistortionCoefficients[1].copyTo(m_DistortionCoefficients[1]);
    cv::stereoCalibrate(object_points, image_points_left, image_points_right, m_CameraMatrix[0], m_DistortionCoefficients[0], m_CameraMatrix[1], m_DistortionCoefficients[1], size, m_RotationMatrix, m_TranslationVector, E, F, cv::CALIB_USE_INTRINSIC_GUESS);

    cv::stereoRectify(m_CameraMatrix[0], m_DistortionCoefficients[0], m_CameraMatrix[1], m_DistortionCoefficients[1], size, m_RotationMatrix, m_TranslationVector, m_RectificationMatrix[0], m_RectificationMatrix[1], m_ProjectionMatrix[0], m_ProjectionMatrix[1], m_DisparityMatrix, cv::CALIB_ZERO_DISPARITY);

    m_RollAdjustment = 0.0;
    m_PitchAdjustment = 0.0;
    generateMapStereo();

    m_IsStereoRectified = true;

    return true;
}

bool Undistort::adjustRectification(double roll, double pitch) {
    if (m_IsStereoRectified == false) {
        return false;
    }
    m_RollAdjustment = roll;
    m_PitchAdjustment = pitch;
    generateMapStereo();
    return true;
}

void Undistort::destroy(void) {
    m_IsCalibrated[0] = false;
    m_IsCalibrated[1] = false;
    m_IsStereoRectified = false;
    for (int side = 0; side < 2; side++) {
        m_RawCameraMatrix[side] = cv::Mat();
        m_CameraMatrix[side] = cv::Mat();
        m_RawDistortionCoefficients[side] = cv::Mat();
        m_DistortionCoefficients[side] = cv::Mat();
        m_RectificationMatrix[side] = cv::Mat();
        m_ProjectionMatrix[side] = cv::Mat();
        m_Map1[side] = cv::Mat();
        m_Map2[side] = cv::Mat();
    }
    m_DisparityMatrix = cv::Mat();
    m_RotationMatrix = cv::Mat();
    m_TranslationVector = cv::Mat();
    m_RollAdjustment = 0.0;
    m_PitchAdjustment = 0.0;
}

double Undistort::baselineLength(void) const {
    if (m_IsStereoRectified == true) {
        return abs(m_TranslationVector.at<double>(0, 0));
    } else {
        return 0.0;
    }
}

void Undistort::generateMap(int side) {
    cv::initUndistortRectifyMap(m_RawCameraMatrix[side], m_RawDistortionCoefficients[side], cv::Mat(), m_RawCameraMatrix[side], cv::Size(m_Width, m_Height), CV_32FC1, m_Map1[side], m_Map2[side]);
}

void Undistort::generateMapStereo(void) {
    cv::Size size(m_Width, m_Height);

    double pitch_matrix_data[9] = {
        1.0, 0.0, 0.0,
        0.0, cos(m_PitchAdjustment), -sin(m_PitchAdjustment),
        0.0, sin(m_PitchAdjustment), cos(m_PitchAdjustment)
    };
    cv::Mat pitch_matrix(3, 3, CV_64F, pitch_matrix_data);
    double roll_matrix_data[9] = {
        cos(m_RollAdjustment), -sin(m_RollAdjustment), 0.0,
        sin(m_RollAdjustment), cos(m_RollAdjustment), 0.0,
        0.0, 0.0, 1.0
    };
    cv::Mat roll_matrix(3, 3, CV_64F, roll_matrix_data);
    cv::Mat new_rotation = pitch_matrix * roll_matrix * m_RectificationMatrix[1];

    cv::initUndistortRectifyMap(m_CameraMatrix[0], m_DistortionCoefficients[0], m_RectificationMatrix[0], m_ProjectionMatrix[0], size, CV_32FC1, m_Map1[0], m_Map2[0]);
    cv::initUndistortRectifyMap(m_CameraMatrix[1], m_DistortionCoefficients[1], new_rotation, m_ProjectionMatrix[1], size, CV_32FC1, m_Map1[1], m_Map2[1]);
}

std::string Undistort::generateFileName(int width_, int height_) {
    std::ostringstream node_name;
    node_name << "calibration" << width_ << "x" << height_ << ".xml";
    return node_name.str();
}
