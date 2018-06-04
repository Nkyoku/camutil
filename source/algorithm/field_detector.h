#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// フィールドの芝、白線およびゴールの検知を行う
class FieldDetector {
public:
    // コンストラクタ
    FieldDetector(void);

    // 芝の検知を行う
    // rectangleに芝の範囲を示す矩形の4つの頂点の座標を返す
    // 検知した芝の2値化画像を返値として返す
    cv::Mat& detectGrass(const cv::Mat &lab_image, std::vector<cv::Point2d> &rectangle, double margin = kGrassMargin);

    // 白線の検知を行う
    // rectangleにはdetectGrass()で検知した芝の領域を示す矩形を渡す
    // line_segmentsに検知した白線の線分リストを返す
    // 検知した白線の2値化画像を返値として返す
    cv::Mat& detectLines(const cv::Mat &lab_image, const std::vector<cv::Point2d> &rectangle, std::vector<cv::Vec4f> &line_segments);

    // 白線をテンプレートと比較して3次元の位置を推定する




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

    // 線分検知器
    cv::Ptr<cv::LineSegmentDetector> m_Lsd;

    // 芝の2値化画像
    cv::Mat m_BinaryGrass;

    // 白線の2値化画像
    cv::Mat m_BinaryLines;



    // L*a*b*の値がLabRegionの内側か判定する
    static bool isInsideLab(int L, int a, int b, const LabRegion &region);

    // L*a*b*の値が複数のLabRegionの内側か判定する
    static int isInsideLab(int L, int a, int b, const std::vector<LabRegion> &region_list);


};
