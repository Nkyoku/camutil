#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>

class GaussianDoG {
public:
    // コンストラクタ
    // kernel_sizeはフィルタのカーネルサイズ
    // varianceはガウス分布の分散
    GaussianDoG(int kernel_size = 3, double variance = 1.0);

    // フィルタ処理を行う
    void compute(const cv::Mat &src);

    // 0度方向の一次微分を取得する
    const cv::Mat& gradient0deg(void) const {
        return m_DerivativeX;
    }

    // 90度方向の一次微分を取得する
    const cv::Mat& gradient90deg(void) const {
        return m_DerivativeY;
    }

    // 0度方向のDoGを取得する
    const cv::Mat& dog0deg(void) const {
        return m_DoGX;
    }

    // 90度方向のDoGを取得する
    const cv::Mat& dog90deg(void) const {
        return m_DoGY;
    }

private:
    // OpenCLコンテキスト
    cv::ocl::Context m_ClContext;

    // OpenCLプログラム
    cv::ocl::Program m_ClProgramUnsigned, m_ClProgramSigned;
    
    // フィルタ係数(0次微分)
    cv::UMat m_Coefficients0th;

    // フィルタ係数(1次微分)
    cv::UMat m_Coefficients1st;

    // フィルタ係数(DoG)
    cv::UMat m_CoefficientsDoG;

    // ブラー画像
    cv::UMat m_BlurX, m_BlurY;

    // 一次微分画像
    cv::Mat m_DerivativeX, m_DerivativeY;

    // DoG画像
    cv::Mat m_DoGX, m_DoGY;
};
