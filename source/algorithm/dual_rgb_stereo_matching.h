#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// ステレオマッチングを行う
class DualRgbStereoMatching {
public:
    // コンストラクタ
    DualRgbStereoMatching(void);

    // 偏差の最大値を設定する
    void setMaximumDisparity(int max_disparity);

    // ステレオマッチングを行う
    // left_image, right_imageに入力としてCV_8UC3型のBGR画像を与えると
    // disparity_mapにCV_32F型の偏差マップが出力される
    bool compute(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &disparity_map);





/*private:*/
    // コスト集計のウィンドウサイズ
    static const int kAggregationSize = 3;

    
    
    // 偏差の最大値
    int m_MaxDisparity = 1;

    // モノクロ画像の勾配の重み
    double m_GradientWeight1 = 3.0;

    // 平滑化画像の勾配の重み
    double m_GradientWeight2 = 1.0;

    // コスト生成の際の勾配の重み
    double m_CostWeightOfGradientX = 1.0, m_CostWeightOfGradientY = 1.0;

    // コスト生成の際の元画像の重み
    double m_CostWeightOfRaw = 1.0;

    // コスト生成の際のCensus画像の重み
    double m_CostWeightOfCensus = 1.0;



    // モノクロ画像
    cv::Mat m_MonochromeImage[2];

    // 平滑化画像
    cv::Mat m_FilteredImage[2];

    // Dual RGB Gradient画像
    cv::Mat m_DualRgbGradientXImage[2], m_DualRgbGradientYImage[2];

    // Census画像
    cv::Mat m_CensusImage[2];

    // コストボリューム
    std::vector<cv::Mat> m_CostVolume;

    

    // ある偏差でのコスト画像を計算する
    void calculateCostImage(const cv::Mat &left_image, const cv::Mat &right_image, int disparity, cv::Mat &cost_image);

    // コストを集計する
    void calculateCostAggregation(const cv::Mat &raw_image, cv::Mat &cost_image, int s, cv::Mat &aggregated_cost_image);

    // コストボリュームから偏差マップを生成する
    static void calculateDisparity(const std::vector<cv::Mat> &cost_volume, int max_disparity, cv::Mat &disparity_map);

    // X方向のDual RGB Gradient画像を生成する
    static void generateDualRgbGradientX(const cv::Mat &src1, const cv::Mat &src2, cv::Mat &dst);

    // Y方向のDual RGB Gradient画像を生成する
    static void generateDualRgbGradientY(const cv::Mat &src1, const cv::Mat &src2, cv::Mat &dst);

    // 7x5のCensus変換を行う
    static void doCensus7x5Transform(const cv::Mat &src, cv::Mat &dst);

};
