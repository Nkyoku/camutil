#pragma once

#include "../videothread.h"
#include "algorithm/undistort.h"
#include "algorithm/census_based_stereo_matching.h"
#include <QtWidgets/QDoubleSpinBox>

QT_FORWARD_DECLARE_CLASS(ImageViewGl);

class VideoStereoThread : public VideoThread{
	Q_OBJECT

public:
    VideoStereoThread(VideoInput *video_input);

    virtual ~VideoStereoThread();

    virtual QString initializeOnce(QWidget *parent) override;

    virtual void initialize(QWidget *parent) override;

    virtual void uninitialize(void) override;

    virtual void restoreSettings(const QSettings &settings) override;

    virtual void saveSettings(QSettings &settings) const override;

protected:
    virtual void processImage(const cv::Mat &input_image) override;

private:
    // 歪み補正器
    Undistort m_Undistort;

    // ステレオマッチング
    CensusBasedStereoMatching m_StereoMatching;

    // 低解像度デプスマップの比率
    int kCoarseRatio = 8;
   
    // 設定された最大偏差
    int m_MaxDisparity = 32;

    // 画像を表示するウィジェット
    ImageViewGl *m_Color[2], *m_Depth[2];

    // 注視点
    int m_WatchPointX = -1, m_WatchPointY = -1;

    // 標準解像度カラー画像
    cv::Mat m_ColorImage[2];

    // 低解像度カラー画像
    cv::Mat m_CoarseColorImage[2];

    // 低解像度モノクロ画像
    cv::Mat m_CoarseGrayImage[2];

    // 低解像度Census画像
    cv::Mat m_CoarseCensusImage[2];

    // コストボリューム
    std::vector<cv::Mat> m_CostVolume;

    // 低解像度デプスマップ
    cv::Mat m_CoarseDepthMap;

    // 低解像度確度マップ
    cv::Mat m_CoarseLikelihoodMap;

    // 標準解像度デプスマップ
    cv::Mat m_NormalDepthMap, m_NormalDepthMapFiltered;


    // 注視点を設定する
    Q_SLOT void watch(int x, int y) {
        m_WatchPointX = x;
        m_WatchPointY = y;
    }
};
