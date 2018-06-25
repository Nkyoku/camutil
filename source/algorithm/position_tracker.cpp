#include "position_tracker.h"
#include "geometry_2d.h"
#include <opencv2/imgproc.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <random>

const std::vector<cv::Vec4f> PositionTracker::kLineTemplate = {
    // 外側の矩形
    cv::Vec4f(-4.5 + kThickness1 / 2, -3.0 + kThickness1 / 2, +4.5 - kThickness1 / 2, -3.0 + kThickness1 / 2),
    cv::Vec4f(+4.5 - kThickness1 / 2, -3.0 + kThickness1 / 2, +4.5 - kThickness1 / 2, +3.0 - kThickness1 / 2),
    cv::Vec4f(+4.5 - kThickness1 / 2, +3.0 - kThickness1 / 2, -4.5 + kThickness1 / 2, +3.0 - kThickness1 / 2),
    cv::Vec4f(-4.5 + kThickness1 / 2, +3.0 - kThickness1 / 2, -4.5 + kThickness1 / 2, -3.0 + kThickness1 / 2),

    // Aサイドゴールエリア
    cv::Vec4f(-4.5 + kThickness1 / 2, -2.5 + kThickness1 / 2, -3.5 - kThickness1 / 2, -2.5 + kThickness1 / 2),
    cv::Vec4f(-3.5 - kThickness1 / 2, -2.5 + kThickness1 / 2, -3.5 - kThickness1 / 2, +2.5 - kThickness1 / 2),
    cv::Vec4f(-3.5 - kThickness1 / 2, +2.5 - kThickness1 / 2, -4.5 + kThickness1 / 2, +2.5 - kThickness1 / 2),

    // Bサイドゴールエリア
    cv::Vec4f(+4.5 - kThickness1 / 2, -2.5 + kThickness1 / 2, +3.5 + kThickness1 / 2, -2.5 + kThickness1 / 2),
    cv::Vec4f(+3.5 + kThickness1 / 2, -2.5 + kThickness1 / 2, +3.5 + kThickness1 / 2, +2.5 - kThickness1 / 2),
    cv::Vec4f(+3.5 + kThickness1 / 2, +2.5 - kThickness1 / 2, +4.5 - kThickness1 / 2, +2.5 - kThickness1 / 2),

    // センターライン
    cv::Vec4f(0.0, -3.0 + kThickness1 / 2, 0.0, +3.0 - kThickness1 / 2),

    // センターサークル(8角形で近似)
    cv::Vec4f(0.0, -1.0823922, +0.76536686, -0.76536686) * (0.75 - kThickness1 / 2),
    cv::Vec4f(+0.76536686, -0.76536686, +1.0823922, 0.0) * (0.75 - kThickness1 / 2),
    cv::Vec4f(+1.0823922, 0.0, +0.76536686, +0.76536686) * (0.75 - kThickness1 / 2),
    cv::Vec4f(+0.76536686, +0.76536686, 0.0, +1.0823922) * (0.75 - kThickness1 / 2),
    cv::Vec4f(0.0, +1.0823922, -0.76536686, +0.76536686) * (0.75 - kThickness1 / 2),
    cv::Vec4f(-0.76536686, +0.76536686, -1.0823922, 0.0) * (0.75 - kThickness1 / 2),
    cv::Vec4f(-1.0823922, 0.0, -0.76536686, -0.76536686) * (0.75 - kThickness1 / 2),
    cv::Vec4f(-0.76536686, -0.76536686, 0.0, -1.0823922) * (0.75 - kThickness1 / 2),
};




PositionTracker::PositionTracker(void) {
    


}

void PositionTracker::setKnownRollAndPitchAngle(double roll_angle, double pitch_angle) {
	m_KnownRollAngle = roll_angle;
	m_KnownPitchAngle = pitch_angle;
	m_IsKnownRollAndPitchAngleProvided = true;
}

void PositionTracker::unsetKnownRollAndPitchAngle(void) {
	m_KnownRollAngle = 0.0;
	m_KnownPitchAngle = 0.0;
	m_IsKnownRollAndPitchAngleProvided = false;
}

double PositionTracker::estimateFieldPlane(const std::vector<cv::Vec4f> &line_segments, const GradientBasedStereoMatching &stereo, const Undistort &undistort, cv::Point3d *field_centroid, cv::Point3d *field_normal) {
	int width = undistort.width();
	int height = undistort.height();
	
	// 垂直に近い(明らかに水平でない)線分を抽出する
	m_VerticalLineSegments.clear();
	for (const cv::Vec4f &segment : line_segments) {
		cv::Point2d vector(segment[2] - segment[0], segment[3] - segment[1]);
		double cos_theta = vector.x / sqrt(vector.x * vector.x + vector.y * vector.y);
		if (abs(cos_theta) < kHorizontalCosAngle){
			m_VerticalLineSegments.push_back(segment);
		}
	}
    if (m_VerticalLineSegments.size() < 3) {
        // 平面推定を行うにはデータが足りない
        return 1.0;
    }

	// 線分上の点の視差を求め、三次元空間上の座標を計算する
	m_DisparitiesOfPoints.clear();
	m_NumberOfPointsOnLines.resize(m_VerticalLineSegments.size());
	for (int index = 0; index < static_cast<int>(m_VerticalLineSegments.size()); index++) {
        m_NumberOfPointsOnLines[index] = getPointsOnLineSegment(m_VerticalLineSegments[index], m_DisparitiesOfPoints, width, height);
	}
	if (static_cast<int>(m_DisparitiesOfPoints.size()) < kNumberOfRansacSamples) {
        // 平面推定を行うにはデータが足りない
        return 1.0;
	}
	stereo.compute(m_DisparitiesOfPoints, kMaximumDisparity);
	undistort.reprojectPointsTo3D(m_DisparitiesOfPoints, m_Points3d);

    // 既知のロール角とピッチ角が与えられているときはそこから法線ベクトルを計算する
    cv::Point3d known_normal;
    if (m_IsKnownRollAndPitchAngleProvided == true) {
        double cos_pitch_angle = cos(m_KnownPitchAngle);
        known_normal.x = sin(m_KnownRollAngle) * cos_pitch_angle;
        known_normal.y = cos(m_KnownRollAngle) * cos_pitch_angle;
        known_normal.z = sin(m_KnownPitchAngle);
    }

	// RANSACによって平面を推定する
	std::mt19937 mt;
	cv::Mat sample_points(kNumberOfRansacSamples, 3, CV_32F);
    std::vector<cv::Point3f> used_points(kNumberOfRansacSamples);
    double min_error = std::numeric_limits<double>::max();
	for (int trial = 0; trial < kRansacTrials; trial++) {
        // kNumberOfRansacSamples個のサンプルを選ぶ
		for (int i = 0; i < kNumberOfRansacSamples; i++) {
			int sample_index = mt() % static_cast<int>(m_Points3d.size());
			const cv::Point3f &point = m_Points3d[sample_index];
			sample_points.at<float>(i, 0) = point.x;
            sample_points.at<float>(i, 1) = point.y;
            sample_points.at<float>(i, 2) = point.z;
            used_points[i] = m_DisparitiesOfPoints[sample_index];
		}

		// PCAによって重心と固有ベクトルを計算する
		// 最も固有値の小さい3番目の固有ベクトルを平面の法線ベクトルと仮定する
		cv::PCA pca(sample_points, cv::Mat(), cv::PCA::DATA_AS_ROW, 3);
		cv::Point3d mean, normal;
		mean.x = pca.mean.at<float>(0, 0);
        mean.y = pca.mean.at<float>(0, 1);
        mean.z = pca.mean.at<float>(0, 2);
		if (m_IsKnownRollAndPitchAngleProvided == true) {
            normal = known_normal;
        } else {
            normal.x = pca.eigenvectors.at<float>(2, 0);
            normal.y = pca.eigenvectors.at<float>(2, 1);
            normal.z = pca.eigenvectors.at<float>(2, 2);
        }

		// 平面からの距離の2乗の合計が最小のものを選ぶ
        double error = 0.0;
		for (const cv::Point3f &point : m_Points3d) {
			double distance = abs((cv::Point3d(point) - mean).dot(normal));
            error += pow(std::min(distance, kRansacThreshold), 2);
		}
        if (error < min_error) {
            min_error = error;
			m_FieldMean = mean;
			m_FieldNormal = normal;
			m_PCAMeans = pca.mean;
			m_PCAEigenVectors = pca.eigenvectors;
			m_PCAEigenValues = pca.eigenvalues;
            m_UsedPoints = used_points;
		}
	}

    // 結果を返す
	if (field_centroid != nullptr) {
		*field_centroid = m_FieldMean;
	}
	if (field_normal != nullptr) {
		*field_normal = m_FieldNormal;
	}

    return min_error / (m_Points3d.size() * pow(kRansacThreshold, 2));
}

double PositionTracker::calculateRollAngle(void) const {
    if (m_IsKnownRollAndPitchAngleProvided == true) {
        return m_KnownRollAngle;
    } else {
        return atan(m_FieldNormal.x / m_FieldNormal.y);
    }
}

double PositionTracker::calculatePitchAngle(void) const {
    if (m_IsKnownRollAndPitchAngleProvided == true) {
        return m_KnownPitchAngle;
    } else {
        return atan(m_FieldNormal.z / sqrt(pow(m_FieldNormal.x, 2) + pow(m_FieldNormal.y, 2)));
    }
}

double PositionTracker::calculateHeight(void) const {
    return abs(m_FieldMean.dot(m_FieldNormal));
}

int PositionTracker::getPointsOnLineSegment(const cv::Vec4f &line_segment, std::vector<cv::Point3f> &points, int width, int height) {
	int ax = static_cast<int>(round(line_segment[0]));
	int ay = static_cast<int>(round(line_segment[1]));
	int bx = static_cast<int>(round(line_segment[2]));
	int by = static_cast<int>(round(line_segment[3]));
	if (bx < ax) {
		std::swap(ax, bx);
		std::swap(ay, by);
	}
	ax = std::max(0, ax);
	ay = std::max(0, std::min(ay, height - 1));
	bx = std::min(bx, width - 1);
	by = std::max(0, std::min(by, height - 1));
	int dx = bx - ax;
	int dy = abs(by - ay);
	int sy = (ay < by) ? 1 : -1;
	if (dx > dy) {
		//傾きが1より小さい場合
		int error = -dx;
		points.reserve(points.size() + dx + 1);
		for (int len = dx; 0 <= len; len--) {
			points.push_back(cv::Point3f(ax, ay, 0.0f));
			ax++;
			error += 2 * dy;
			if (0 <= error) {
				ay += sy;
				error -= 2 * dx;
			}
		}
		return dx + 1;
	}
	else {
		//傾きが1以上の場合
		int error = -dy;
		points.reserve(points.size() + dy + 1);
		for (int len = dy; 0 <= len; len--) {
			points.push_back(cv::Point3f(ax, ay, 0.0f));
			ay += sy;
			error += 2 * dx;
			if (0 <= error) {
				ax++;
				error -= 2 * dy;
			}
		}
		return dy + 1;
	}
}
