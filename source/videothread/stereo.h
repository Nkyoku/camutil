#pragma once

#include "../videothread.h"
#include "undistort.h"
#include <opencv2/calib3d.hpp>

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

    // 低解像度デプスマップの比率
    int kCoarseRatio = 4;
   
    // 設定された最大偏差
    int m_MaxDisparity = 32;

    // 表示画像
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

    // 標準解像度デプスマップ
    cv::Mat m_NormalDepthMap, m_NormalDepthMapFiltered;


    // Winner takes Allで視差を調べる
    static void findDisparity(const std::vector<cv::Mat> &cost_volume, cv::Mat &disparity, double scale = 1.0);

    // Winner takes Allで視差を調べてサブピクセル補間を行う
    static void findDisparitySubPixel(const std::vector<cv::Mat> &cost_volume, cv::Mat &disparity, double scale = 1.0);

    // ステレオマッチングを行って視差を計算する
    void stereoMatching(void);

    // 5x5のCensus変換を行う
    static void VideoStereoThread::doCensus5x5Transform(const cv::Mat &src, cv::Mat &dst);

    // 注視点を設定する
    Q_SLOT void watch(int x, int y);
};
