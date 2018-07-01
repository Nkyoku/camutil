#include "field_detector.h"
#include "geometry_2d.h"
#define _USE_MATH_DEFINES
#include <math.h>

const FieldDetector::LabRegion FieldDetector::kDefaultGrassRegion = {
    0, 255,
    20, 180,
    M_PI * 0.6667, -M_PI * 0.8333
};

const FieldDetector::LabRegion FieldDetector::kDefaultWhiteRegion = {
    180, 255,
    0, 20,
    -M_PI, M_PI
};

FieldDetector::FieldDetector(void) {
    m_Lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD, 0.5);
}

const cv::Mat& FieldDetector::detect(const cv::Mat &bgr_image, std::vector<cv::Vec4f> &line_segments, std::vector<cv::Vec4f> *edge_line_segments) {
    int width = bgr_image.cols;
    int height = bgr_image.rows;
    double diagonal = sqrt(width * width + height * height);
    
    // 出力値を初期化する
    line_segments.clear();
    m_LongerLineSegments.clear();

    do {
        // 芝を検知する
        cv::cvtColor(bgr_image, m_LabImage, cv::COLOR_BGR2Lab);
        detectGrassRegion(m_LabImage, m_GrassBinaryImage, m_GrassRectangle, 1);
        if (m_GrassRectangle.empty() == true) {
            break;
        }

        // 線分を抽出し、長い線分を選ぶ
        cv::cvtColor(bgr_image, m_GrayscaleImage, cv::COLOR_BGR2GRAY);
        cv::dilate(m_GrayscaleImage, m_DilatedImage, cv::Mat());
        m_Lsd->detect(cv::Mat(m_DilatedImage, m_GrassRectangle), m_LineSegments);
        int offset_x = m_GrassRectangle.x;
        int offset_y = m_GrassRectangle.y;
        for (cv::Vec4f &segment : m_LineSegments) {
            segment[0] += offset_x;
            segment[1] += offset_y;
            segment[2] += offset_x;
            segment[3] += offset_y;
        }
        selectLongSegments(m_LineSegments, m_LongerLineSegments, diagonal * kLineLengthThreshold);
        connectMultipleSegments(m_LongerLineSegments, m_LongerLineSegments, diagonal * kGapThreshold, diagonal * kSameSegmentThreshold, kParallelAngle);

        // 平行な線分を合成し、白線を抽出する
        edgePolarityCheck(m_LongerLineSegments, m_GrayscaleImage, m_EdgePolarity);
        combineParallelSegments(m_LongerLineSegments, m_EdgePolarity, m_GrayscaleImage, m_WhiteLines, diagonal * kNeighborSegmentThreshold);
        connectMultipleSegments(m_WhiteLines, line_segments, diagonal * kGapThreshold, diagonal * kSameSegmentThreshold, kParallelAngle);
    } while (false);

	// 検知した線分を出力する
	if (edge_line_segments != nullptr) {
		edge_line_segments->resize(m_LongerLineSegments.size());
		std::copy(m_LongerLineSegments.begin(), m_LongerLineSegments.end(), edge_line_segments->begin());
	}

    return m_GrayscaleImage;
}

void FieldDetector::detectGrassRegion(const cv::Mat &lab_image, cv::Mat &binary, cv::Rect &rectangle, int scale) {
    static constexpr double kThreshold1 = 0.05;
    static constexpr double kThreshold2 = 0.50;
    static constexpr double kThreshold3 = 0.95;
    
    int width = lab_image.cols;
    int height = lab_image.rows;

    // 芝の色を検知して2値化する
    binary.create(height, width, CV_8U);
    lab_image.forEach<cv::Vec3b>([&](const cv::Vec3b &value, const int pos[2]) {
        binary.at<uint8_t>(pos[0], pos[1]) = isInsideLab(value[0], value[1], value[2], kDefaultGrassRegion) ? 255 : 0;
    });

    // 垂直方向の画素分布から芝の範囲を推定する
    double y_start = 0.0, y_end = 0.0;
    do {
        // 垂直方向に画素を累積する
        std::vector<double> cumulation(height);
        double acc = 0.0;
        for (int y = 0; y < height; y++) {
            const uint8_t *ptr = binary.ptr(y);
            int sum = 0;
            for (int x = 0; x < width; x++) {
                sum += *ptr++;
            }
            acc += sum;
            cumulation[y] = acc;
        }
        double total_pixels = cumulation[height - 1];
        if (total_pixels == 0.0) {
            break;
        }

        // 累積値が10%->50%->90%となるところのY座標で分布範囲を推定する
        int y1, y2, y3;
        for (y1 = 0; y1 < height; y1++) {
            if ((total_pixels * kThreshold1) <= cumulation[y1]) {
                break;
            }
        }
        for (y2 = y1; y2 < height; y2++) {
            if ((total_pixels * kThreshold2) <= cumulation[y2]) {
                break;
            }
        }
        for (y3 = y2; y3 < height; y3++) {
            if ((total_pixels * kThreshold3) <= cumulation[y3]) {
                break;
            }
        }
        double c1 = cumulation[y1] / total_pixels;
        double c2 = cumulation[y2] / total_pixels;
        double c3 = cumulation[y3] / total_pixels;
        if ((c1 == c2) || (c2 == c3)) {
            break;
        }
        y_start = (c2 * y1 - c1 * y2) / (c2 - c1);
        y_end = (c3 - 1.0) * y2 - (c2 - 1.0) * y3 / (c3 - c2);
    } while (false);

    // 水平方向の画素分布から芝の範囲を推定する
    double x_start = 0.0, x_end = 0.0;
    do {
        std::vector<double> cumulation(width, 0.0);
        for (int y = 0; y < height; y++) {
            const uint8_t *ptr = binary.ptr(y);
            for (int x = 0; x < width; x++) {
                cumulation[x] += *ptr++;
            }
        }
        for (int x = 1; x < width; x++) {
            cumulation[x] += cumulation[x - 1];
        }
        double total_pixels = cumulation[width - 1];
        if (total_pixels == 0.0) {
            break;
        }

        // 累積値が10%->50%->90%となるところのX座標で分布範囲を推定する
        int x1, x2, x3;
        for (x1 = 0; x1 < width; x1++) {
            if ((total_pixels * kThreshold1) <= cumulation[x1]) {
                break;
            }
        }
        for (x2 = x1; x2 < width; x2++) {
            if ((total_pixels * kThreshold2) <= cumulation[x2]) {
                break;
            }
        }
        for (x3 = x2; x3 < width; x3++) {
            if ((total_pixels * kThreshold3) <= cumulation[x3]) {
                break;
            }
        }
        double c1 = cumulation[x1] / total_pixels;
        double c2 = cumulation[x2] / total_pixels;
        double c3 = cumulation[x3] / total_pixels;
        if ((c1 == c2) || (c2 == c3)) {
            break;
        }
        x_start = (c2 * x1 - c1 * x2) / (c2 - c1);
        x_end = (c3 - 1.0) * x2 - (c2 - 1.0) * x3 / (c3 - c2);
    } while (false);

    int x_start_int = std::max(0, std::min(static_cast<int>(round(x_start)), width - 1));
    int x_end_int = std::max(0, std::min(static_cast<int>(round(x_end)), width));
    int y_start_int = std::max(0, std::min(static_cast<int>(round(y_start)), height - 1));
    int y_end_int = std::max(0, std::min(static_cast<int>(round(y_end)), height));
    
    rectangle.x = x_start_int;
    rectangle.y = y_start_int;
    rectangle.width = x_end_int - x_start_int;
    rectangle.height = y_end_int - y_start_int;
}

bool FieldDetector::isInsideLab(int L, int a, int b, const LabRegion &region) {
    a -= 128;
    b -= 128;
    if ((region.L_min <= L) && (L <= region.L_max)) {
        int chroma_squared = a * a + b * b;
        int chroma_min_squared = region.chroma_min * region.chroma_min;
        int chroma_max_squared = region.chroma_max * region.chroma_max;
        if ((chroma_min_squared <= chroma_squared) && (chroma_squared <= chroma_max_squared)) {
            double hue = atan2(b, a);
            if (region.hue_min < region.hue_max) {
                if ((region.hue_min <= hue) && (hue <= region.hue_max)) {
                    return true;
                }
            } else {
                if ((hue <= region.hue_max) || (region.hue_min <= hue)) {
                    return true;
                }
            }
        }
    }
    return false;
}

/*int FieldDetector::isInsideLab(int L, int a, int b, const std::vector<LabRegion> &region_list) {
    a -= 128;
    b -= 128;
    int result = 0;
    int bitmask = 1;
    int chroma_squared = a * a + b * b;
    double hue = atan2(b, a);
    for (const auto &region : region_list) {
        if ((region.L_min <= L) && (L <= region.L_max)) {
            int chroma_min_squared = region.chroma_min * region.chroma_min;
            int chroma_max_squared = region.chroma_max * region.chroma_max;
            if ((chroma_min_squared <= chroma_squared) && (chroma_squared <= chroma_max_squared)) {
                if (region.hue_min < region.hue_max) {
                    if ((region.hue_min <= hue) && (hue <= region.hue_max)) {
                        result |= bitmask;
                    }
                } else {
                    if ((hue <= region.hue_max) || (region.hue_min <= hue)) {
                        result |= bitmask;
                    }
                }
            }
        }
        bitmask <<= 1;
    }
    return result;
}*/

void FieldDetector::selectLongSegments(const std::vector<cv::Vec4f> &input_segments, std::vector<cv::Vec4f> &output_segments, double min_length) {
    double min_length2 = min_length * min_length;
    int num_of_outputs = 0;
    output_segments.resize(input_segments.size());
    for (int index = 0; index < static_cast<int>(input_segments.size()); index++) {
        const cv::Vec4f &segment = input_segments[index];
        if (min_length2 <= segmentLength2(segment)) {
            output_segments[num_of_outputs] = segment;
            num_of_outputs++;
        }
    }
    output_segments.resize(num_of_outputs);
}

void FieldDetector::selectNthLongerSegments(std::vector<cv::Vec4f> &line_segments, int max_count) {
    if (static_cast<int>(line_segments.size()) < max_count) {
        max_count = static_cast<int>(line_segments.size());
    }
    std::nth_element(line_segments.begin(), line_segments.begin() + max_count, line_segments.end(), [](const cv::Vec4f &a, const cv::Vec4f &b) {
        return (segmentLength2(b) < segmentLength2(a));
    });
}

void FieldDetector::edgePolarityCheck(const std::vector<cv::Vec4f> &line_segments, const cv::Mat &binary_image, std::vector<bool> &polarities) {
    int width = binary_image.cols;
    int height = binary_image.rows;
    polarities.resize(line_segments.size());
    for (int index = 0; index < static_cast<int>(line_segments.size()); index++) {
        const cv::Vec4f &segment = line_segments[index];
        double center_x = (segment[0] + segment[2]) * 0.5;
        double center_y = (segment[1] + segment[3]) * 0.5;
        double rlength = 1.0 / sqrt(pow(segment[0] - segment[2], 2) + pow(segment[1] - segment[3], 2));
        double normal_x = -(segment[3] - segment[1]) * rlength;
        double normal_y = (segment[2] - segment[0]) * rlength;
        bool result = false;
        for (int px = 1; px < 10; px++) {
            int ax = static_cast<int>(round(center_x + normal_x * px));
            int ay = static_cast<int>(round(center_y + normal_y * px));
            int bx = static_cast<int>(round(center_x - normal_x * px));
            int by = static_cast<int>(round(center_y - normal_y * px));
            if ((0 <= ax) && (ax < width) && (0 <= ay) && (ay < height) && (0 <= bx) && (bx < width) && (0 <= by) && (by < height)) {
                uint8_t color_a = binary_image.at<uint8_t>(ay, ax);
                uint8_t color_b = binary_image.at<uint8_t>(by, bx);
                if (color_a != color_b) {
                    result = (color_b < color_a);
                    break;
                }
            }
        }
        polarities[index] = result;
    }
}

void FieldDetector::combineParallelSegments(const std::vector<cv::Vec4f> &input_segments, const std::vector<bool> &edge_polarities, const cv::Mat &binary_image, std::vector<cv::Vec4f> &output_segments, double threshold) {
    int width = binary_image.cols;
    int height = binary_image.rows;
    int num_of_outputs = 0;
    output_segments.resize(input_segments.size());
    for (int i = 0; i < static_cast<int>(input_segments.size()); i++) {
        // 線分A
        cv::Point2d a1(input_segments[i][0], input_segments[i][1]);
        cv::Point2d a2(input_segments[i][2], input_segments[i][3]);
        cv::Point2d vector_a = a2 - a1;
        double length_a = normalizeAndLength(vector_a);
        bool edge_a = edge_polarities[i];
        for (int j = i + 1; j < static_cast<int>(input_segments.size()); j++) {
            // 線分B
            cv::Point2d b1(input_segments[j][0], input_segments[j][1]);
            cv::Point2d b2(input_segments[j][2], input_segments[j][3]);
            cv::Point2d vector_b = b2 - b1;
            double length_b = normalizeAndLength(vector_b);
            bool edge_b = edge_polarities[j];

            // 線分A,Bの成す角がほぼ平行であるか調べる
            double cos_angle = vector_a.dot(vector_b);
            if (abs(cos_angle) < kParallelAngle) {
                continue;
            }
            if (cos_angle < 0) {
                std::swap(b1, b2);
                vector_b *= -1;
                edge_b = !edge_b;
            }

            // エッジ極性が不正なものを除外する
            if (edge_a == edge_b) {
                continue;
            }
            if (isPointOnLeftSideOfLine(b1, a1, a2) == true) {
                // 画像空間中ではY軸が反転しているのでB1は線分Aの向かって右にある
                if (edge_a == false) {
                    // 線分Aの右が白ではないので不正
                    continue;
                }
            } else {
                if (edge_a == true) {
                    continue;
                }
            }

            // 線分同士が近いか調べる
            if (threshold < distanceBetweenSegmentsSimple(a1, a2, b1, b2)) {
                continue;
            }
            if (threshold < distanceBetweenSegments(a1, a2, b1, b2)) {
                continue;
            }

            // 線分Cは線分A,Bの平均ベクトル
            cv::Point2d c1((a1 + b1) * 0.5);
            cv::Point2d c2((a2 + b2) * 0.5);

            // 線分Cの中心点が白色か調べる
            cv::Point2d c3((c1 + c2) * 0.5);
            cv::Point c3_int(static_cast<int>(round(c3.x)), static_cast<int>(round(c3.y)));
            if ((c3_int.x < 0) && (width <= c3_int.x) && (c3_int.y < 0) && (height <= c3_int.y)) {
                continue;
            }
            if (binary_image.at<uint8_t>(c3_int.y, c3_int.x) == 0) {
                continue;
            }

            cv::Vec4f &result = output_segments[num_of_outputs];
            result[0] = static_cast<float>(c1.x);
            result[1] = static_cast<float>(c1.y);
            result[2] = static_cast<float>(c2.x);
            result[3] = static_cast<float>(c2.y);
            num_of_outputs++;
        }
    }
    output_segments.resize(num_of_outputs);
}

/*void FieldDetector::getIntersections(const std::vector<cv::Vec4f> &input_segments, std::vector<Intersection> &intersections, const std::vector<cv::Range> &ranges) {
    int height = static_cast<int>(ranges.size());
    int number_of_intersections = 0;
    intersections.resize(input_segments.size() * (input_segments.size() - 1) / 2);
    for (int segment1_index = 0; segment1_index < static_cast<int>(input_segments.size()); segment1_index++) {
        const cv::Vec4f &segment1 = input_segments[segment1_index];
        cv::Point2d a1(segment1[0], segment1[1]);
        cv::Point2d a2(segment1[2], segment1[3]);
        cv::Point2d vector_a(a2 - a1);
        a1 -= vector_a * (kStretchRate - 1.0);
        a2 += vector_a * (kStretchRate - 1.0);
        for (int segment2_index = segment1_index + 1; segment2_index < static_cast<int>(input_segments.size()); segment2_index++) {
            const cv::Vec4f &segment2 = input_segments[segment2_index];
            cv::Point2d b1(segment2[0], segment2[1]);
            cv::Point2d b2(segment2[2], segment2[3]);
            cv::Point2d vector_b(b2 - b1);
            b1 -= vector_b * (kStretchRate - 1.0);
            b2 += vector_b * (kStretchRate - 1.0);
            bool included1, included2;
            cv::Point2d intersection = segmentIntersection(a1, a2, b1, b2, &included1, &included2);
            int y = static_cast<int>(round(intersection.y));
            if ((included1 && included2) && (0 <= y) && (y < height)) {
                int x = static_cast<int>(round(intersection.x));
                if ((ranges[y].start <= x) && (x < ranges[y].end)) {
                    Intersection &result = intersections[number_of_intersections];
                    result.point = intersection;
                    result.segment1 = segment1_index;
                    result.segment2 = segment2_index;
                    number_of_intersections++;
                }
            }
        }
    }
    intersections.resize(number_of_intersections);
}*/
