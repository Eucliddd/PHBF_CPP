#ifndef PHBF_H
#define PHBF_H

#include <vector>
#include <Eigen/Dense>
#include "dynamic_bitset.hpp"
#include <bitset>

#define DELTA 256

class PHBF {
public:
    PHBF(int hash_count, int dim, int sample_factor = 100, const std::string& method = "gaussian");

    void initialize(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
    void bulk_add(const Eigen::MatrixXd& X);
    bool* lookup(const Eigen::MatrixXd& X);
    long double compute_fpr(const Eigen::MatrixXd& X);

    inline long model_size() {
        return vectors.rows() * vectors.cols() * sizeof(double);
    }

    inline long array_size(){
        return bit_array.size() * bit_array[0].size() / 8;
    }

    private:
        int hash_count;
        //int delta;
        std::vector<std::bitset<DELTA>> bit_array; // specify the number of bits in the bitset
        std::string method;
        int sample_factor;
        int dim;
        Eigen::MatrixXd vectors;

        Eigen::MatrixXi compute_hashes(const Eigen::MatrixXd& X);
        Eigen::MatrixXd _select_vectors(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
        Eigen::MatrixXd _normalize_vectors(Eigen::MatrixXd vectors);
};

#endif
