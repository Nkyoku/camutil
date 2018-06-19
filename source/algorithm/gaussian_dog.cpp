#include "gaussian_dog.h"
#include <opencv2/imgproc.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <fstream>

GaussianDoG::GaussianDoG(int kernel_size, double variance) {
    CV_Assert((1 <= kernel_size) && ((kernel_size % 2) == 1));

    // 正規分布関数
    auto gaussian0th = [](double x, double sigma) {
        return exp(-(x * x) / (2 * sigma * sigma)) / (sqrt(2 * M_PI) * sigma);
    };

    // 正規分布関数の一次微分
    auto gaussian1st = [](double x, double sigma) {
        return -x / (sqrt(2 * M_PI) * pow(sigma, 3)) * exp(-(x * x) / (2 * sigma * sigma));
    };

    // 水平，垂直方向のフィルタ係数を生成する
    cv::Mat coef_0th(kernel_size, 1, CV_32F);
    cv::Mat coef_1st(kernel_size, 1, CV_32F);
    cv::Mat coef_dog(kernel_size, 1, CV_32F);
    int half_size = kernel_size / 2;
    for (int x = -half_size; x <= half_size; x++) {
        coef_0th.at<float>(x + half_size) = gaussian0th(x, variance);
        coef_1st.at<float>(x + half_size) = k1stScale * gaussian1st(-x, variance);
        coef_dog.at<float>(x + half_size) = kDoGScale * (gaussian0th(x, variance * 0.5) - gaussian0th(x, variance));
    }
    m_Coefficients0th = coef_0th.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    m_Coefficients1st = coef_1st.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    m_CoefficientsDoG = coef_dog.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    
    // OpenCLのカーネルを作成する
    m_ClContext.create(cv::ocl::Device::TYPE_GPU);
    CV_Assert(m_ClContext.ndevices() != 0);

    std::ifstream ifs("cl/gaussian_dog.cl");
    CV_Assert(ifs.fail() == false);
    std::string source_text((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    cv::ocl::ProgramSource program_source(source_text);
    cv::String error_message;
    m_ClProgram = m_ClContext.getProg(program_source, cv::format("-D N=%d -D OFFSET=0.0f", kernel_size), error_message);
    if (!error_message.empty()) {
        CV_Error(cv::Error::OpenCLApiCallError, error_message.c_str());
    }
    m_ClProgramOffset = m_ClContext.getProg(program_source, cv::format("-D N=%d -D OFFSET=128.0f", kernel_size), error_message);
    if (!error_message.empty()) {
        CV_Error(cv::Error::OpenCLApiCallError, error_message.c_str());
    }
}

void GaussianDoG::compute(const cv::Mat &src) {
    /*
    // CPUコード (係数をcv::Matに直す必要あり)
    cv::sepFilter2D(src, m_DerivativeX, src.depth(), m_Coefficients1st, m_Coefficients0th, cv::Point(-1, -1), 128.0);
    cv::sepFilter2D(src, m_DerivativeY, src.depth(), m_Coefficients0th, m_Coefficients1st, cv::Point(-1, -1), 128.0);
    cv::sepFilter2D(src, m_DoGX, src.depth(), m_CoefficientsDoG, m_Coefficients0th, cv::Point(-1, -1), 128.0);
    cv::sepFilter2D(src, m_DoGY, src.depth(), m_Coefficients0th, m_CoefficientsDoG, cv::Point(-1, -1), 128.0);
    */

    size_t global_size[2];
    global_size[0] = src.cols;
    global_size[1] = src.rows;

    size_t local_size[2];
    local_size[0] = 32;
    local_size[1] = 8;

    if ((m_DerivativeX.rows != src.rows) || (m_DerivativeX.cols != src.cols)) {
        // メモリーを確保する
        m_BlurX.create(src.rows, src.cols, CV_8U, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        m_BlurY.create(src.rows, src.cols, CV_8U, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        m_DerivativeX.create(src.rows, src.cols, CV_8U);
        m_DerivativeY.create(src.rows, src.cols, CV_8U);
        m_DoGX.create(src.rows, src.cols, CV_8U);
        m_DoGY.create(src.rows, src.cols, CV_8U);
    }

    // ブラーを掛ける
    cv::ocl::Kernel blur_x_kernel("convoluteX", m_ClProgram);
    cv::ocl::Kernel blur_y_kernel("convoluteY", m_ClProgram);
    cv::UMat src_umat = src.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    blur_x_kernel.args(cv::ocl::KernelArg::ReadOnly(src_umat), cv::ocl::KernelArg::ReadWriteNoSize(m_BlurX), cv::ocl::KernelArg::PtrReadOnly(m_Coefficients0th));
    blur_x_kernel.run(2, global_size, local_size, false);
    blur_y_kernel.args(cv::ocl::KernelArg::ReadOnly(src_umat), cv::ocl::KernelArg::ReadWriteNoSize(m_BlurY), cv::ocl::KernelArg::PtrReadOnly(m_Coefficients0th));
    blur_y_kernel.run(2, global_size, local_size, false);

    // 正規分布関数の一次微分で畳み込む
    cv::ocl::Kernel derivative_x_kernel("convoluteX", m_ClProgramOffset);
    cv::ocl::Kernel derivative_y_kernel("convoluteY", m_ClProgramOffset);
    cv::UMat derivative_x_umat = m_DerivativeX.getUMat(cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat derivative_y_umat = m_DerivativeY.getUMat(cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    derivative_x_kernel.args(cv::ocl::KernelArg::ReadOnly(m_BlurY), cv::ocl::KernelArg::ReadWriteNoSize(derivative_x_umat), cv::ocl::KernelArg::PtrReadOnly(m_Coefficients1st));
    derivative_x_kernel.run(2, global_size, local_size, false);
    derivative_y_kernel.args(cv::ocl::KernelArg::ReadOnly(m_BlurX), cv::ocl::KernelArg::ReadWriteNoSize(derivative_y_umat), cv::ocl::KernelArg::PtrReadOnly(m_Coefficients1st));
    derivative_y_kernel.run(2, global_size, local_size, false);

    // DoGを畳み込む
    cv::ocl::Kernel dog_x_kernel("convoluteX", m_ClProgramOffset);
    cv::ocl::Kernel dog_y_kernel("convoluteY", m_ClProgramOffset);
    cv::UMat dog_x_umat = m_DoGX.getUMat(cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat dog_y_umat = m_DoGY.getUMat(cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    dog_x_kernel.args(cv::ocl::KernelArg::ReadOnly(m_BlurY), cv::ocl::KernelArg::ReadWriteNoSize(dog_x_umat), cv::ocl::KernelArg::PtrReadOnly(m_CoefficientsDoG));
    dog_x_kernel.run(2, global_size, local_size, false);
    dog_y_kernel.args(cv::ocl::KernelArg::ReadOnly(m_BlurX), cv::ocl::KernelArg::ReadWriteNoSize(dog_y_umat), cv::ocl::KernelArg::PtrReadOnly(m_CoefficientsDoG));
    dog_y_kernel.run(2, global_size, local_size, true);
}
