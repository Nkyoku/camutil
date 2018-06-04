#include "weighted_linear_regression.h"

bool WeightedLinearRegression::compute1stOrder(const std::vector<float> &X, const std::vector<float> &Y, const std::vector<float> &W, std::vector<double> &coefficients) {
    // Y[j] = j-th observed data point
    // X[j] = j-th value
    // W[j] = j-th weight value

    // N = Number of linear terms
    static const int N = 2;

    auto f = [](double x, int i) {
        return pow(x, i);
    };

    // Number of data points
    int M = static_cast<int>(Y.size());

    // Degrees of freedom
    int NDF = M - N;
    if (NDF < 1) {
        // If not enough data, don't attempt regression
        return false;
    }

    // Form Least Squares Matrix
    V.create(N, N, CV_64F);
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < M; k++) {
                //sum += W[k] * X[i, k] * X[j, k];
                sum += W[k] * f(X[k], i) * f(X[k], j);
            }
            V.at<double>(i, j) = sum;
            if (i != j) {
                V.at<double>(j, i) = sum;
            }
        }
    }

    B.resize(N);
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int k = 0; k < M; k++) {
            sum += W[k] * f(X[k], i) * Y[k];
        }
        B[i] = sum;
    }

    // V now contains the raw least squares matrix
    cv::invert(V, invertedV);

    // V now contains the inverted least square matrix
    // Matrix multpily to get coefficients C = V^(-1)B
    coefficients.resize(N);
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += invertedV.at<double>(i, j) * B[j];
        }
        coefficients[i] = sum;
    }

    return true;
}
