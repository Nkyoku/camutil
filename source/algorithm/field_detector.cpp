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

const std::vector<cv::Vec4f> FieldDetector::kFieldTemplate = {
    { 0, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 0, 0, 0 }
};

FieldDetector::FieldDetector(void) {
    m_Lsd = cv::createLineSegmentDetector();
}

cv::Mat& FieldDetector::detectGrass(const cv::Mat &lab_image, double scale, std::vector<cv::Point2d> *rectangle) {
    // 芝の色を検知して2値化する
    m_BinaryGrass.create(lab_image.rows, lab_image.cols, CV_8UC1);
    lab_image.forEach<cv::Vec3b>([&](const cv::Vec3b &value, const int pos[2]) {
        m_BinaryGrass.at<uint8_t>(pos[0], pos[1]) = isInsideLab(value[0], value[1], value[2], kDefaultGrassRegion) ? 255 : 0;
    });

    // 芝の領域のモーメントを求め、領域を矩形で近似する
    cv::Moments moments = cv::moments(m_BinaryGrass, true);
    if (moments.m00 == 0) {
        m_GrassContours[0].x = 0.0;
        m_GrassContours[0].y = 0.0;
        m_GrassContours[1].x = 0.0;
        m_GrassContours[1].y = 0.0;
        m_GrassContours[2].x = 0.0;
        m_GrassContours[2].y = 0.0;
        m_GrassContours[3].x = 0.0;
        m_GrassContours[3].y = 0.0;
        m_EnlargedGrassContours[0].x = 0.0;
        m_EnlargedGrassContours[0].y = 0.0;
        m_EnlargedGrassContours[1].x = 0.0;
        m_EnlargedGrassContours[1].y = 0.0;
        m_EnlargedGrassContours[2].x = 0.0;
        m_EnlargedGrassContours[2].y = 0.0;
        m_EnlargedGrassContours[3].x = 0.0;
        m_EnlargedGrassContours[3].y = 0.0;
        m_GrassContourWidth = 0.0;
        m_GrassContourHeight = 0.0;

    } else {
        double center_x = moments.m10 / moments.m00;
        double center_y = moments.m01 / moments.m00;
        double mu20 = moments.m20 / moments.m00 - center_x * center_x;
        double mu02 = moments.m02 / moments.m00 - center_y * center_y;
        double mu11 = moments.m11 / moments.m00 - center_x * center_y;
        double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        double half_length1 = sqrt(1.5 * (mu20 + mu02 + sqrt(4 * mu11 * mu11 + pow(mu20 - mu02, 2))));
        double half_length2 = sqrt(1.5 * (mu20 + mu02 - sqrt(4 * mu11 * mu11 + pow(mu20 - mu02, 2))));
        center_x *= scale;
        center_y *= scale;
        half_length1 *= scale;
        half_length2 *= scale;
        m_GrassContours[0].x = center_x - half_length1 * cos_theta - half_length2 * sin_theta;
        m_GrassContours[0].y = center_y - half_length1 * sin_theta + half_length2 * cos_theta;
        m_GrassContours[1].x = center_x + half_length1 * cos_theta - half_length2 * sin_theta;
        m_GrassContours[1].y = center_y + half_length1 * sin_theta + half_length2 * cos_theta;
        m_GrassContours[2].x = center_x + half_length1 * cos_theta + half_length2 * sin_theta;
        m_GrassContours[2].y = center_y + half_length1 * sin_theta - half_length2 * cos_theta;
        m_GrassContours[3].x = center_x - half_length1 * cos_theta + half_length2 * sin_theta;
        m_GrassContours[3].y = center_y - half_length1 * sin_theta - half_length2 * cos_theta;
        half_length1 *= kGrassMargin;
        half_length2 *= kGrassMargin;
        m_EnlargedGrassContours[0].x = center_x - half_length1 * cos_theta - half_length2 * sin_theta;
        m_EnlargedGrassContours[0].y = center_y - half_length1 * sin_theta + half_length2 * cos_theta;
        m_EnlargedGrassContours[1].x = center_x + half_length1 * cos_theta - half_length2 * sin_theta;
        m_EnlargedGrassContours[1].y = center_y + half_length1 * sin_theta + half_length2 * cos_theta;
        m_EnlargedGrassContours[2].x = center_x + half_length1 * cos_theta + half_length2 * sin_theta;
        m_EnlargedGrassContours[2].y = center_y + half_length1 * sin_theta - half_length2 * cos_theta;
        m_EnlargedGrassContours[3].x = center_x - half_length1 * cos_theta + half_length2 * sin_theta;
        m_EnlargedGrassContours[3].y = center_y - half_length1 * sin_theta - half_length2 * cos_theta;
        m_GrassContourWidth = half_length1 * 2;
        m_GrassContourHeight = half_length2 * 2;
    }

    if (rectangle != nullptr) {
        rectangle->resize(m_EnlargedGrassContours.size());
        std::copy(m_EnlargedGrassContours.begin(), m_EnlargedGrassContours.end(), rectangle->begin());
    }

    return m_BinaryGrass;
}

cv::Mat& FieldDetector::detectLines(const cv::Mat &lab_image, std::vector<cv::Vec4f> *line_segments) {
    int height = lab_image.rows;
    int width = lab_image.cols;

    // 芝の領域に含まれる白色の成分のみを抽出する
    // 水平スキャンラインと芝の領域の交点を計算し、始点・終点をrangesに格納する
    std::vector<cv::Range> ranges(height);
    for (int y = 0; y < height; y++) {
        int start_x = width, end_x = 0;
        for (int edge = 0; edge < 4; edge++) {
            double x1 = m_EnlargedGrassContours[edge].x;
            double y1 = m_EnlargedGrassContours[edge].y;
            double x2 = m_EnlargedGrassContours[(edge + 1) % 4].x;
            double y2 = m_EnlargedGrassContours[(edge + 1) % 4].y;
            if (y2 < y1) {
                std::swap(x1, x2);
                std::swap(y1, y2);
            }
            int y1_int = static_cast<int>(ceil(y1));
            int y2_int = static_cast<int>(floor(y2));
            if (((y1_int <= y) && (y <= y2_int)) && (y1_int != y2_int)) {
                double dx = (x2 - x1) / (y2 - y1);
                double sx = x1 - dx * y1;
                double x = sx + dx * y;
                start_x = std::min(start_x, static_cast<int>(floor(x)));
                end_x = std::max(end_x, static_cast<int>(ceil(x)) + 1);
            }
        }
        ranges[y] = cv::Range(start_x, end_x);
    }

    // 白線の色を検知して2値化する
    m_BinaryLines.create(height, width, CV_8UC1);
    lab_image.forEach<cv::Vec3b>([&](const cv::Vec3b &value, const int pos[2]) {
        const cv::Range &range = ranges[pos[0]];
        bool inside_grass = (range.start <= pos[1]) && (pos[1] < range.end);
        m_BinaryLines.at<uint8_t>(pos[0], pos[1]) = (inside_grass && isInsideLab(value[0], value[1], value[2], kDefaultWhiteRegion)) ? 255 : 0;
    });

    // 線分を抽出し、長い線分を選ぶ
    m_Lsd->detect(m_BinaryLines, m_LineSegments);
    double line_threshold2 = (width * width + height * height) * pow(kLineLengthThreshold, 2);
    selectLongSegments(m_LineSegments, line_threshold2, m_LongLineSegments);
    //if (kMaximumLineCount < static_cast<int>(m_LongLineSegments.size())) {
    //    selectNthLongerSegments(m_LongLineSegments, kMaximumLineCount);
    //    m_LongLineSegments.resize(kMaximumLineCount);
    //}

    // 平行な線分を合成し、白線を抽出する
    edgePolarityCheck(m_LongLineSegments, m_BinaryLines, m_EdgePolarity);
    double neighbor_threshold = sqrt(width * width + height * height) * kNeighborSegmentThreshold;
    combineParallelSegments(m_LongLineSegments, m_EdgePolarity, m_BinaryLines, neighbor_threshold, m_WhiteLines);



    




    if (3 <= m_LongLineSegments.size()) {
        // 線分を無限に延長したときの交点をすべて求める
        // 平行度の高い線分同士の交点は除外する
        /*m_LineIntersections.create(m_LongLineSegments.size() * (m_LongLineSegments.size() - 1) / 2, 2, CV_32F);
        int number_of_points = 0;
        for (int segment1_index = 0; segment1_index < static_cast<int>(m_LongLineSegments.size()); segment1_index++) {
            const cv::Vec4f &segment1 = m_LongLineSegments[segment1_index];
            cv::Point2d a1(segment1[0], segment1[1]), a2(segment1[2], segment1[3]);
            cv::Point2d a = a2 - a1;
            for (int segment2_index = segment1_index + 1; segment2_index < static_cast<int>(m_LongLineSegments.size()); segment2_index++) {
                const cv::Vec4f &segment2 = m_LongLineSegments[segment2_index];
                cv::Point2d b1(segment2[0], segment2[1]), b2(segment2[2], segment2[3]);
                cv::Point2d b = b2 - b1;
                double cos_theta = a.dot(b) / sqrt((a.x * a.x + a.y * a.y) * (b.x + b.x + b.y * b.y));
                if (cos_theta < kParallelAngle) {
                    cv::Point2d intersection = a1 + a * b.cross(b1 - a1) / b.cross(a);
                    m_LineIntersections.at<float>(number_of_points, 0) = static_cast<float>(intersection.x);
                    m_LineIntersections.at<float>(number_of_points, 1) = static_cast<float>(intersection.y);
                    number_of_points++;
                }
            }
        }
        m_LineIntersections.resize(number_of_points);*/

        // 線分の交点のうち、芝の領域内にあるものを抽出する
        m_LineIntersections.create(m_LongLineSegments.size() * (m_LongLineSegments.size() - 1) / 2, 2, CV_32F);
        int number_of_points = 0;
        for (int segment1_index = 0; segment1_index < static_cast<int>(m_LongLineSegments.size()); segment1_index++) {
            const cv::Vec4f &segment1 = m_LongLineSegments[segment1_index];
            cv::Point2d a1(segment1[0], segment1[1]), a2(segment1[2], segment1[3]);
            for (int segment2_index = segment1_index + 1; segment2_index < static_cast<int>(m_LongLineSegments.size()); segment2_index++) {
                const cv::Vec4f &segment2 = m_LongLineSegments[segment2_index];
                cv::Point2d b1(segment2[0], segment2[1]), b2(segment2[2], segment2[3]);
                bool inside1, inside2;
                cv::Point2d intersection = segmentIntersection(a1, a2, b1, b2, &inside1, &inside2);
                if ((inside1 || inside2) && (0 <= intersection.y) && (intersection.y < (height - 1))) {
                    int x = static_cast<int>(round(intersection.x));
                    int y = static_cast<int>(round(intersection.y));
                    if ((ranges[y].start <= x) && (x < ranges[y].end)) {
                        m_LineIntersections.at<float>(number_of_points, 0) = static_cast<float>(intersection.x);
                        m_LineIntersections.at<float>(number_of_points, 1) = static_cast<float>(intersection.y);
                        number_of_points++;
                    }
                }
            }
        }
        m_LineIntersections.resize(number_of_points);

        /*// 交点を3つのクラスタに分類する
        m_LineIntersections.resize(number_of_points);
        //cv::Mat intersections(m_LineIntersections, cv::Rect(0, 0, number_of_points, 2));
        cv::Mat clusters(number_of_points, 1, CV_32SC1);
        cv::Mat centers;
        cv::kmeans(m_LineIntersections, 3, clusters, cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 3, 1.0), 1, cv::KMEANS_PP_CENTERS, centers);

        m_VanishingPoints.resize(3);
        for (int index = 0; index < 3; index++) {
            m_VanishingPoints[index].x = centers.at<float>(index, 0);
            m_VanishingPoints[index].y = centers.at<float>(index, 1);
        }*/



    }

    





    if (line_segments != nullptr) {
        line_segments->resize(m_LongLineSegments.size());
        std::copy(m_LongLineSegments.begin(), m_LongLineSegments.end(), line_segments->begin());
    }

    return m_BinaryLines;
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

int FieldDetector::isInsideLab(int L, int a, int b, const std::vector<LabRegion> &region_list) {
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
}

void FieldDetector::selectLongSegments(const std::vector<cv::Vec4f> &input_segments, double min_length2, std::vector<cv::Vec4f> &output_segments) {
    output_segments.clear();
    for (const auto &segment : input_segments) {
        if (min_length2 <= segmentLength2(segment)) {
            output_segments.push_back(segment);
        }
    }
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

void FieldDetector::combineParallelSegments(const std::vector<cv::Vec4f> &input_segments, const std::vector<bool> &edge_polarities, const cv::Mat &binary_image, double threshold, std::vector<cv::Vec6f> &output_segments) {
    int width = binary_image.cols;
    int height = binary_image.rows;
    output_segments.clear();
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

            /*// 線分Bの各点B1,B2から線分Aに降ろした垂線との交点C1,C2を計算する
            double length_a1_c1 = vector_a.dot(b1 - a1);
            double length_a1_c2 = vector_a.dot(b2 - a1);
            cv::Point2d c1(a1 + length_a1_c1 * vector_a);
            cv::Point2d c2(a1 + length_a1_c2 * vector_a);

            // 線分AとBの間にある線分Dを求める
            cv::Point2d d1((b1 + c1) * 0.5);
            cv::Point2d d2((b2 + c2) * 0.5);
            cv::Point2d vector_d = d2 - d1;
            double length_d = sqrt(pow(vector_d.x, 2) + pow(vector_d.y, 2));
            vector_d *= 1.0 / length_d;

            // 線分Aの各点A1,A2から線分Dに降ろした垂線との交点E1,E2を計算する
            double length_d1_e1_clamped = std::min(std::max(vector_a.dot(a1 - d1), 0.0), length_d);
            double length_d1_e2_clamped = std::min(std::max(vector_a.dot(a2 - d1), 0.0), length_d);
            if (length_d1_e1_clamped == length_d1_e2_clamped) {
            continue;
            }
            cv::Point2d e1(d1 + length_d1_e1_clamped * vector_d);
            cv::Point2d e2(d1 + length_d1_e2_clamped * vector_d);

            double thickness = distance(a1 + (length_a1_c1 + length_a1_c2) * 0.5 * vector_a, (b1 + b2) * 0.5);*/

            // 線分Cは線分A,Bの平均ベクトル
            cv::Point2d c1((a1 + b1) * 0.5);
            cv::Point2d c2((a2 + b2) * 0.5);
            cv::Point2d c3((c1 + c2) * 0.5);

            // 線分Cの中心点が白色か調べる
            cv::Point c3_int(static_cast<int>(round(c3.x)), static_cast<int>(round(c3.y)));
            if ((c3_int.x < 0) && (width <= c3_int.x) && (c3_int.y < 0) && (height <= c3_int.y)) {
                continue;
            }
            if (binary_image.at<uint8_t>(c3_int.y, c3_int.x) == 0) {
                continue;
            }

            // 線分Cの中心C3から線分Aに降ろした垂線の長さの2倍が線分Cの太さとなる
            double thickness = 2.0 * abs(c3.cross(a2 - a1) + a2.cross(a1)) / length_a;

            cv::Vec6f result;
            result[0] = static_cast<float>(c1.x);
            result[1] = static_cast<float>(c1.y);
            result[2] = static_cast<float>(c2.x);
            result[3] = static_cast<float>(c2.y);
            result[4] = static_cast<float>(thickness);
            result[5] = static_cast<float>(abs(cos_angle));
            output_segments.push_back(result);
        }
    }
}

void FieldDetector::connectGap(const std::vector<cv::Vec6f> &input_segments, double threshold, std::vector<cv::Vec6f> &output_segments) {
    int number_of_outputs = 0;
    output_segments.resize(input_segments.size());
    for (int i = 0; i < static_cast<int>(input_segments.size()); i++) {
        // 線分A
        cv::Point2d a1(input_segments[i][0], input_segments[i][1]);
        cv::Point2d a2(input_segments[i][2], input_segments[i][3]);
        cv::Point2d vector_a = a2 - a1;
        double length_a = normalizeAndLength(vector_a);
        for (int j = i + 1; j < static_cast<int>(input_segments.size()); j++) {
            // 線分B
            cv::Point2d b1(input_segments[j][0], input_segments[j][1]);
            cv::Point2d b2(input_segments[j][2], input_segments[j][3]);
            cv::Point2d vector_b = b2 - b1;
            double length_b = normalizeAndLength(vector_b);

            // 線分A,Bの成す角がほぼ平行であるか調べる
            double cos_angle = (a2 - a1).dot(vector_b);
            if (abs(cos_angle) < kParallelAngle) {
                continue;
            }
            if (cos_angle < 0) {
                std::swap(b1, b2);
                vector_b *= -1;
            }

            // 線分同士が近いか調べる
            if (length_a < length_b) {
                if (threshold < distanceBetweenLineAndSegment(b1, b2, a1, a2)) {
                    continue;
                }
            } else {
                if (threshold < distanceBetweenLineAndSegment(a1, a2, b1, b2)) {
                    continue;
                }
            }

            // 線分を合成する
            combine2Segments(a1, a2, b1, b2, a1, a2);
            vector_a = a2 - a1;
            length_a = normalizeAndLength(vector_a);
        }

        cv::Vec6f result;
        result[0] = static_cast<float>(a1.x);
        result[1] = static_cast<float>(a1.y);
        result[2] = static_cast<float>(a2.x);
        result[3] = static_cast<float>(a2.y);
        result[4] = static_cast<float>(input_segments[i][4]);
        result[5] = static_cast<float>(input_segments[i][5]);
        output_segments.push_back(result);
        number_of_outputs++;
    }
    output_segments.resize(number_of_outputs);
}
