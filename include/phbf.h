#ifndef PHBF_H
#define PHBF_H

#include <vector>
#include <Eigen/Dense>
#include "dynamic_bitset.hpp"
#include <bitset>

#define TEST_SCALER 0

const int DELTA = 256;

class PHBF {
public:
    PHBF(const int hash_count, const int dim, const int sample_factor = 100, const std::string& method = "gaussian") noexcept;

    void initialize(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) noexcept;
    void bulk_add(const Eigen::MatrixXd& X) noexcept;
    void initandadd(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) noexcept;
    auto lookup(const Eigen::MatrixXd& X) const;
    long double compute_fpr(const Eigen::MatrixXd& X) const;

    inline long model_size() const noexcept{
        return vectors.rows() * vectors.cols() * sizeof(double);
    }

    inline long array_size() const noexcept{
        return bit_array.size() * bit_array[0].size() / 8;
    }

    private:
        int hash_count;
        //int delta;
        std::vector<std::bitset<DELTA>> bit_array; // specify the number of bits in the bitset
        std::string method;
        int sample_factor;
        int dim;
        //size_t size;
        Eigen::MatrixXd vectors;

        Eigen::MatrixXi compute_hashes(const Eigen::MatrixXd& X) const noexcept;
        Eigen::MatrixXd _select_vectors(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) const noexcept;
        Eigen::MatrixXd _normalize_vectors(Eigen::MatrixXd vectors) noexcept;
#if TEST_SCALER
        Eigen::VectorXd min;
        Eigen::VectorXd max;
#endif
};

#endif
