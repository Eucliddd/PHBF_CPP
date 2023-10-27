#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class HPBF {
public:
    HPBF(int size, int hash_count, int dim, int sample_factor = 100, std::string method = "gaussian")
        : hash_count(hash_count), size(size / hash_count), method(method), sample_factor(sample_factor), dim(dim) {
        bit_array.resize(hash_count);
        for (int i = 0; i < hash_count; ++i) {
            bit_array[i].resize(size);
            bit_array[i].setZero();
        }
    }

    MatrixXd _select_vectors(MatrixXd X, MatrixXd Y) {
        int sample_size = hash_count * sample_factor;
        MatrixXd candidates(sample_size, dim);

        if (method == "gaussian") {
            candidates = MatrixXd::Random(sample_size, dim);
        }
        else if (method == "optimistic") {
            candidates = MatrixXd::Zero(sample_size, dim);
            candidates.col(0).setOnes();
        }
        else {
            candidates = MatrixXd::Random(sample_size, dim);
        }

        MatrixXd pos_projections = X * candidates.transpose();
        MatrixXd neg_projections = Y * candidates.transpose();

        MatrixXd pos_projections_normalized = pos_projections.rowwise().normalized();
        MatrixXd neg_projections_normalized = neg_projections.rowwise().normalized();

        MatrixXd pos_hash_values = (pos_projections_normalized.array() * (size - 1)).cast<int>();
        MatrixXd neg_hash_values = (neg_projections_normalized.array() * (size - 1)).cast<int>();

        VectorXi overlaps(sample_size);
        for (int i = 0; i < sample_size; ++i) {
            overlaps[i] = (pos_hash_values.row(i).array() == neg_hash_values.row(i).array()).count();
        }

        VectorXi best_hashes_idx(sample_size);
        for (int i = 0; i < sample_size; ++i) {
            best_hashes_idx[i] = i;
        }

        std::sort(best_hashes_idx.data(), best_hashes_idx.data() + best_hashes_idx.size(),
                  [&overlaps](int i, int j) { return overlaps[i] > overlaps[j]; });

        MatrixXd best_hashes = candidates.block(0, 0, hash_count, dim);

        return best_hashes;
    }

    MatrixXd _normalize_vectors(MatrixXd vectors) {
        for (int i = 0; i < vectors.rows(); ++i) {
            vectors.row(i).normalize();
        }
        return vectors;
    }

    void initialize(MatrixXd X, MatrixXd Y) {
        MatrixXd vectors = _select_vectors(X, Y);
        vectors = _normalize_vectors(vectors);
        this->vectors = vectors.transpose();
    }

    MatrixXi compute_hashes(MatrixXd X) {
        MatrixXd projections = X * vectors;
        projections = (projections.array() * (size - 1)).cast<int>();

        MatrixXi hash_values(projections.rows(), projections.cols());
        for (int i = 0; i < projections.rows(); ++i) {
            for (int j = 0; j < projections.cols(); ++j) {
                if (projections(i, j) > 1 || projections(i, j) < 0) {
                    hash_values(i, j) = -1;
                }
                else {
                    hash_values(i, j) = projections(i, j);
                }
            }
        }

        return hash_values;
    }

    void bulk_add(MatrixXd X) {
        MatrixXd projections = X * vectors;
        projections = (projections.array() * (size - 1)).cast<int>();

        for (int i = 0; i < projections.cols(); ++i) {
            for (int j = 0; j < projections.rows(); ++j) {
                int hash_value = projections(j, i);
                if (hash_value != -1) {
                    bit_array[i](hash_value) = 1;
                }
            }
        }
    }

    std::vector<bool> lookup(MatrixXd X) {
        std::vector<bool> results(X.rows(), true);
        MatrixXi hash_values = compute_hashes(X);

        for (int i = 0; i < hash_values.rows(); ++i) {
            for (int j = 0; j < hash_values.cols(); ++j) {
                int hash_value = hash_values(i, j);
                if (hash_value != -1) {
                    if (bit_array[j](hash_value) == 0) {
                        results[i] = false;
                        break;
                    }
                }
                else {
                    results[i] = false;
                    break;
                }
            }
        }

        return results;
    }

    double compute_fpr(MatrixXd X) {
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

private:
    int hash_count;
    int size;
    std::vector<MatrixXi> bit_array;
    std::string method;
    int sample_factor;
    int dim;
    MatrixXd vectors;
};
