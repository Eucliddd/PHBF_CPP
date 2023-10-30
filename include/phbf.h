#ifndef HPBF_H
#define HPBF_H

#include <vector>
#include <Eigen/Dense>
#include "dynamic_bitset.hpp"

class PHBF {
public:
    PHBF(int size, int hash_count, int dim, int sample_factor = 100, const std::string& method = "gaussian");

    void initialize(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
    Eigen::MatrixXi compute_hashes(const Eigen::MatrixXd& X);
        void bulk_add(const Eigen::MatrixXd& X);
        std::vector<bool> lookup(const Eigen::MatrixXd& X);
        double compute_fpr(const Eigen::MatrixXd& X);

    private:
        int hash_count;
        int delta;
        std::vector<sul::dynamic_bitset<>> bit_array; // specify the number of bits in the bitset
        std::string method;
        int sample_factor;
        int dim;
        Eigen::MatrixXd vectors;

        Eigen::MatrixXd _select_vectors(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
        Eigen::MatrixXd _normalize_vectors(Eigen::MatrixXd vectors);
};

#endif
