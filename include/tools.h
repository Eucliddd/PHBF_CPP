#ifndef TOOLS_H
#define TOOLS_H
#include <Eigen/Dense>
#include <chrono>
#include <functional>
#include <optional>
#include <type_traits>
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

template <typename Func, typename... Args> 
requires !std::is_void_v<std::invoke_result_t<Func, Args...>>
auto TIME(Func&& func, Args&&... args){
    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    auto result = std::invoke(func, std::forward<Args>(args)...);
    //std::forward<Func>(func)(std::forward<Args>(args)...);

    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();

    return std::make_tuple(std::optional<decltype(result)>(std::move(result)), elapsed);
}

template <typename Func, typename... Args> 
requires std::is_void_v<std::invoke_result_t<Func, Args...>>
auto TIME(Func&& func, Args&&... args){
    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    std::invoke(func, std::forward<Args>(args)...);
    //std::forward<Func>(func)(std::forward<Args>(args)...);

    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();

    return std::make_tuple(0, elapsed);
}


#endif