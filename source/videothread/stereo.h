#pragma once

#include "../videothread.h"
#include "algorithm/undistort.h"
#include "algorithm/census_based_stereo_matching.h"
#include "algorithm/area_interpolation.h"
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
    // 縮小比率1
    static constexpr int kHScale1 = 1, kVScale1 = 4;

    // 縮小比率2
    static constexpr int kHScale2 = 4, kVScale2 = 8;

    // 縮小比率3
    static constexpr int kHScale3 = 16, kVScale3 = 16;
    
    // 最大偏差
    static constexpr int kMaxDisparity = 8;

    // 歪み補正器
    Undistort m_Undistort;

    // ステレオマッチング
    CensusBasedStereoMatching m_StereoMatching[3];
    
    // 補間
    AreaInterpolation m_Interpolation;

    // 画像を表示するウィジェット
    ImageViewGl *m_Color[2], *m_Depth, *m_Disparity[3], *m_Likelihood[3];

    // 注視点
    int m_WatchPointX = -1, m_WatchPointY = -1;

    // オリジナル画像
    cv::Mat m_OriginalImage[2];

    // カラー画像
    cv::Mat m_ColorImage[6];

    // 偏差マップ
    cv::Mat m_DisparityMap[3];

    // 拡大された偏差マップ
    cv::Mat m_MagnifiedDisparityMap[2];

    // 尤度マップ
    cv::Mat m_LikelihoodMap[3];

    // デプスマップ
    cv::Mat m_DepthMap;

    // 注視点を設定する
    Q_SLOT void watch(int x, int y) {
        m_WatchPointX = x;
        m_WatchPointY = y;
    }
};
