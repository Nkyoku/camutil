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

    // 45度方向の一次微分を取得する
    const cv::Mat& gradient45deg(void) const {
        return m_Derivative45;
    }

    // 90度方向の一次微分を取得する
    const cv::Mat& gradient90deg(void) const {
        return m_DerivativeY;
    }

    // 135度方向の一次微分を取得する
    const cv::Mat& gradient135deg(void) const {
        return m_Derivative135;
    }

    // 0度方向のDoGを取得する
    const cv::Mat& dog0deg(void) const {
        return m_DoGX;
    }

    // 45度方向のDoGを取得する
    const cv::Mat& dog45deg(void) const {
        return m_DoG45;
    }

    // 90度方向のDoGを取得する
    const cv::Mat& dog90deg(void) const {
        return m_DoGY;
    }

    // 135度方向のDoGを取得する
    const cv::Mat& dog135deg(void) const {
        return m_DoG135;
    }

private:
    // 1次微分係数に乗じる補正係数
    static constexpr double k1stScale = 2.0;

    // DoG係数に乗じる補正係数
    static constexpr double kDoGScale = 2.0;

    // OpenCLコンテキスト
    cv::ocl::Context m_ClContext;

    // OpenCLプログラム
    cv::ocl::Program m_ClProgram, m_ClProgramOffset;
    
    // フィルタ係数(0次微分)
    cv::UMat m_Coefficients0th, m_Coefficients0thDiagonal;

    // フィルタ係数(1次微分)
    cv::UMat m_Coefficients1st, m_Coefficients1stDiagonal;

    // フィルタ係数(DoG)
    cv::UMat m_CoefficientsDoG, m_CoefficientsDoGDiagonal;

    // ブラー画像
    cv::UMat m_BlurX, m_BlurY, m_Blur45, m_Blur135;

    // 一次微分画像
    cv::Mat m_DerivativeX, m_DerivativeY, m_Derivative45, m_Derivative135;

    // DoG画像
    cv::Mat m_DoGX, m_DoGY, m_DoG45, m_DoG135;
};
