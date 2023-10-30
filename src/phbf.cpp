/**
 * @file phbf.cpp
 * @brief Implementation of the PHBF class
 * @note The implementation is based on the paper "New wine in bottle".
 * However, the implementation is not exactly the same as the paper. 
 * The original paper using MinMaxScaler to normalize each row after projection.
 * We use normalize() to normalize each row of X and vectors before projection.
*/

#include <algorithm>
#include "phbf.h"

PHBF::PHBF(int size, int hash_count, int dim, int sample_factor, const std::string& method)
    : hash_count(hash_count), delta(size / hash_count), method(method), sample_factor(sample_factor), dim(dim) {
    bit_array.resize(hash_count);
    for (int i = 0; i < hash_count; ++i) {
        bit_array[i].resize(delta,0);
    }
}
// select best "hash_count" vectors from "sample_size*hash_count" vectors
/*
    * 1. generate "sample_size*hash_count" vectors
    * 2. normalize them
    * 3. compute the projection of X and Y on the vectors
    * 4. compute the hash values of X and Y
    * 5. compute the overlap of the hash values of X and Y
    * 6. select the best "hash_count" vectors
*/
Eigen::MatrixXd PHBF::_select_vectors(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    int sample_size = hash_count * sample_factor;
    Eigen::MatrixXd candidates(sample_size, dim);

    // Generate random vectors
    if (method == "gaussian") {
        candidates = Eigen::MatrixXd::Random(sample_size, dim);
    }
    else if (method == "optimistic") {
        candidates = Eigen::MatrixXd::Zero(sample_size, dim);
        candidates.col(0).setOnes();
    }
    else {
        candidates = Eigen::MatrixXd::Random(sample_size, dim);
    }

    // Normalize the vectors
    Eigen::MatrixXd candidates_normalized = candidates.rowwise().normalized();
    Eigen::MatrixXd X_normalized = X.rowwise().normalized();
    Eigen::MatrixXd Y_normalized = Y.rowwise().normalized();

    Eigen::MatrixXd pos_projections_normalized = X_normalized * candidates_normalized.transpose();
    Eigen::MatrixXd neg_projections_normalized = Y_normalized * candidates_normalized.transpose();

    Eigen::MatrixXi pos_hash_values = (pos_projections_normalized.array().abs() * (delta - 1)).cast<int>();
    Eigen::MatrixXi neg_hash_values = (neg_projections_normalized.array().abs() * (delta - 1)).cast<int>();

    // Compute the overlap of the hash values of X and Y
    Eigen::VectorXi overlaps(sample_size);
    for (int i = 0; i < sample_size; ++i) {
        overlaps[i] = (pos_hash_values.row(i).array() == neg_hash_values.row(i).array()).count();
    }

    Eigen::VectorXi best_hashes_idx(sample_size);
    for (int i = 0; i < sample_size; ++i) {
        best_hashes_idx[i] = i;
    }

    // Sort the hash functions by the overlap
    std::sort(best_hashes_idx.data(), best_hashes_idx.data() + best_hashes_idx.size(),
              [&overlaps](int i, int j) { return overlaps[i] > overlaps[j]; });

    // Select the best hash functions
    Eigen::MatrixXd best_hashes(hash_count, dim);
    for (int i = 0; i < hash_count; ++i) {
        best_hashes.row(i) = candidates_normalized.row(best_hashes_idx[i]);
    }
    return best_hashes;
}

Eigen::MatrixXd PHBF::_normalize_vectors(Eigen::MatrixXd vectors) {
    for (int i = 0; i < vectors.rows(); ++i) {
        vectors.row(i).normalize();
    }
    return vectors;
}

void PHBF::initialize(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    Eigen::MatrixXd vectors = _select_vectors(X, Y);
    //vectors = _normalize_vectors(vectors);
    this->vectors = vectors.transpose();
}

// Compute the hash values of X
Eigen::MatrixXi PHBF::compute_hashes(const Eigen::MatrixXd& X) {
    Eigen::MatrixXd projections = X.rowwise().normalized() * vectors;
    Eigen::MatrixXi hash_values = (projections.array().abs() * (delta - 1)).cast<int>();

    // Eigen::MatrixXi hash_values(indexes.rows(), indexes.cols());
    // for (int i = 0; i < projections.rows(); ++i) {
    //     for (int j = 0; j < projections.cols(); ++j) {
    //         if (indexes(i, j) > 1 || indexes(i, j) < 0) {
    //             hash_values(i, j) = -1;
    //         }
    //         else {
    //             hash_values(i, j) = indexes(i, j);
    //         }
    //     }
    // }

    return hash_values;
}

void PHBF::bulk_add(const Eigen::MatrixXd& X) {
    Eigen::MatrixXi indexes = compute_hashes(X);

    for (int i = 0; i < indexes.cols(); ++i) {
        for (int j = 0; j < indexes.rows(); ++j) {
            int hash_value = indexes(j, i);
            bit_array[i].set(hash_value);
        }
    }
}

std::vector<bool> PHBF::lookup(const Eigen::MatrixXd& X) {
    std::vector<bool> results(X.rows(), true);
    Eigen::MatrixXi hash_values = compute_hashes(X);

    for (int i = 0; i < hash_values.rows(); ++i) {
        for (int j = 0; j < hash_values.cols(); ++j) {
            int hash_value = hash_values(i, j);
            if (!bit_array[j][hash_value]) {
                results[i] = false;
                break;
            }

        }
    }
    return results;
}

double PHBF::compute_fpr(const Eigen::MatrixXd& X) {
    int fp = 0;
    int tn = 0;
    std::vector<bool> results = lookup(X);

    for (bool result : results) {
        if (result) {
            fp++;
        }
        else {
            tn++;
        }
    }

    return static_cast<double>(fp) / (fp + tn);
}
