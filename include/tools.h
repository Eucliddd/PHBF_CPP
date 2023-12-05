#ifndef TOOLS_H
#define TOOLS_H
#include <Eigen/Dense>
/*
#define SCALEDATA(X) (((X).rowwise() - (X).colwise().minCoeff().transpose()).array().rowwise() / \
                        ((X).colwise().maxCoeff() - (X).colwise().minCoeff()).transpose().array())
*/
// inline Eigen::MatrixXd SCALER(const Eigen::MatrixXd& X, const Eigen::VectorXd& min, const Eigen::VectorXd& max) {
//     return (((X).rowwise() - min.transpose()).array().rowwise() / ((max - min).transpose().array() + 1e-5));
// }
inline auto SCALEDATA(const Eigen::MatrixXd& X) noexcept {
    Eigen::RowVectorXd min = std::move(X.colwise().minCoeff());
    Eigen::RowVectorXd max = std::move(X.colwise().maxCoeff());
    auto SCALER = [&]() -> Eigen::MatrixXd {
        return (((X).rowwise() - min).array().rowwise() / ((max - min).array() + 1e-5));
    };
    return SCALER();
}

template <typename T, std::size_t N>
constexpr inline std::size_t size(T (&)[N]) noexcept {
    return N;
}

#endif