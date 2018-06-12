#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// ステレオマッチングを行う
class CensusBasedStereoMatching {
public:
    // コンストラクタ
    CensusBasedStereoMatching(void);

    // 偏差の最大値を設定する
    void setMaximumDisparity(int max_disparity);

    // ステレオマッチングを行う
    // left_image, right_imageに入力としてCV_8UC3型のBGR画像を与えると
    // disparity_mapにCV_32F型の偏差マップが出力される
    bool compute(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &disparity_map, cv::Mat &likelihood_map);





private:
    
    
    
    // 偏差の最大値
    int m_MaxDisparity = 1;



    // モノクロ画像
    cv::Mat m_MonochromeImage[2];

    // Census画像
    cv::Mat m_CensusImage[2];

    // コストボリューム
    std::vector<cv::Mat> m_CostVolume;

    

    // ある偏差でのコスト画像を計算する
    void calculateCostImage(const cv::Mat &left_image, const cv::Mat &right_image, int disparity, cv::Mat &cost_image);

    // コストボリュームから偏差マップを生成する
    static void calculateDisparity(const std::vector<cv::Mat> &cost_volume, int max_disparity, cv::Mat &disparity_map, cv::Mat &likelihood_map);

    // 7x5のCensus変換を行う
    static void doCensus7x5Transform(const cv::Mat &src, cv::Mat &dst);

};
