#include <opencv2/core.hpp>

class WeightedLinearRegression{
public:
    // 1次線形回帰の係数を計算する
    bool compute1stOrder(const std::vector<float> &X, const std::vector<float> &Y, const std::vector<float> &W, std::vector<double> &coefficients);

private:
    // Least squares and var/covar matrix
    cv::Mat V, invertedV;

    // Vector for LSQ
    std::vector<double> B;
};
