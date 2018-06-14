#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// フィールドの芝、白線の検知を行う
class FieldDetector {
public:
    // コンストラクタ
    FieldDetector(void);

    // 芝の検知を行う
    // rectangleに芝の範囲を示す矩形の4つの頂点の座標を返す
    // 検知した芝の2値化画像を返値として返す
    cv::Mat& detectGrass(const cv::Mat &lab_image, double scale = 1.0, std::vector<cv::Point2d> *rectangle = nullptr);

    // 白線の検知を行う
    // rectangleにはdetectGrass()で検知した芝の領域を示す矩形を渡す
    // line_segmentsに検知した白線の線分リストを返す
    // 検知した白線の2値化画像を返値として返す
    cv::Mat& detectLines(const cv::Mat &lab_image, std::vector<cv::Vec4f> &line_segments, std::vector<cv::Vec4f> *edge_line_segments = nullptr);



private:
    // L*a*b*色空間での扇状の立体を表す構造体
    struct LabRegion {
        // 明度L*の下限と上限[0:255]
        int L_min, L_max;

        // 彩度sqrt(a*^2+b*^2)の下限と上限[0:180]
        int chroma_min, chroma_max;

        // 色相atan2(b*,a*)の下限と上限[-pi:pi]
        double hue_min, hue_max;
    };

    // 芝の色を示すL*a*b*色空間の領域の初期値
    static const LabRegion kDefaultGrassRegion;

    // 白線の色を示すL*a*b*色空間の領域の初期値
    static const LabRegion kDefaultWhiteRegion;

    // 芝の領域のマージン
    static constexpr double kGrassMargin = 1.2;

    // 検出する白線の最低長(入力画像サイズに対する割合)
    static constexpr double kLineLengthThreshold = 1.0 / 32.0;

    // 検出する白線の本数の最大数
    static constexpr int kMaximumLineCount = 32;

    // 2本の直線が平行だと見なすcosθ
    static constexpr double kParallelAngle = 255.0 / 256.0;

    // 2本の線分が近いと見なす距離(入力画像サイズに対する割合)
    static constexpr double kNeighborSegmentThreshold = 1.0 / 64.0;

    // 2本の線分がもともと同一のものだと見なす距離(入力画像サイズに対する割合)
    static constexpr double kSameSegmentThreshold = 1.0 / 512.0;

    // 2本の線分が同一だと見なす断絶の距離(入力画像サイズに対する割合)
    static constexpr double kGapThreshold = 1.0 / 8.0;

    // 交点検知の線分の伸縮率
    static constexpr double kStretchRate = 1.2;

    // フィールドの白線テンプレート
    static const std::vector<cv::Vec4f> kFieldTemplate;

    // 交点の情報を格納する構造体
    struct Intersection {
        // 交点の座標
        cv::Point2d point;

        // 交点を形成する線分の番号
        int segment1, segment2;
    };

    // 線分検知器
    cv::Ptr<cv::LineSegmentDetector> m_Lsd;

    // 芝の2値化画像
    cv::Mat m_BinaryGrass;

    // 芝の領域の頂点リスト
    std::array<cv::Point2d, 4> m_GrassContours, m_EnlargedGrassContours;

    // 芝の領域の幅と高さ
    double m_GrassContourWidth, m_GrassContourHeight;

    // 白線の2値化画像
    cv::Mat m_BinaryLines;

    // 白線の線分リスト
    std::vector<cv::Vec4f> m_LineSegments, m_LongerLineSegments;

    // 線分のエッジ極性
    std::vector<bool> m_EdgePolarity;

    // 合成された線分のリスト
    std::vector<cv::Vec4f> m_WhiteLines;

    // L*a*b*の値がLabRegionの内側か判定する
    static bool isInsideLab(int L, int a, int b, const LabRegion &region);

    // L*a*b*の値が複数のLabRegionの内側か判定する
    static int isInsideLab(int L, int a, int b, const std::vector<LabRegion> &region_list);

    // 線分の長さの2乗を計算する
    static double segmentLength2(const cv::Vec4f &segment) {
        return pow(segment[0] - segment[2], 2) + pow(segment[1] - segment[3], 2);
    }

    // 閾値より長い線分を抽出する
    // min_length2に抽出する線分の最低長の2乗を与える
    static void selectLongSegments(const std::vector<cv::Vec4f> &input_segments, std::vector<cv::Vec4f> &output_segments, double min_length);

    // 長い線分のみを抽出する
    // max_countに抽出する線分の最大数を指定する
    // 処理はline_segmentsに対してin-placeで行われ、抽出された有効な線分の本数を返す
    static void selectNthLongerSegments(std::vector<cv::Vec4f> &line_segments, int max_count);

    // 線分のエッジ極性を調べる
    // 線分の向かって右側に白色があればtrueが格納される
    static void edgePolarityCheck(const std::vector<cv::Vec4f> &line_segments, const cv::Mat &binary_image, std::vector<bool> &polarities);

    // 平行な線分を太さを持つ線分に合成する
    static void combineParallelSegments(const std::vector<cv::Vec4f> &input_segments, const std::vector<bool> &edge_polarities, const cv::Mat &binary_image, std::vector<cv::Vec4f> &output_segments, double threshold);

    // 線分の交点を求める
    static void getIntersections(const std::vector<cv::Vec4f> &input_segments, std::vector<Intersection> &intersections, const std::vector<cv::Range> &ranges);

};
