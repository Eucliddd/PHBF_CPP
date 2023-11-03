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
#include "tools.h"
#include <fstream>
#include <iostream>
#include <set>


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
        // std::ofstream outFile("gaussian.csv");
        // outFile << "candidates:" << std::endl;
        // for (int i = 0; i < sample_size; ++i) {
        //     for (int j = 0; j < dim; ++j) {
        //         outFile << candidates(i, j) << ",";
        //     }
        //     outFile << std::endl;
        // }
        // outFile.close();
    }
    else if (method == "optimistic") {
        candidates = Eigen::MatrixXd::Zero(sample_size, dim);
        candidates.col(0).setOnes();
    }
    else {
        candidates = Eigen::MatrixXd::Random(sample_size, dim);
    }

    // Normalize the vectors
    //Eigen::MatrixXd candidates_normalized = candidates.rowwise().normalized();
    //Eigen::MatrixXd X_normalized = X.rowwise().normalized();
    //Eigen::MatrixXd Y_normalized = Y.rowwise().normalized();

    // Eigen::MatrixXd pos_projections = X * candidates.transpose();
    // Eigen::MatrixXd neg_projections = Y * candidates.transpose();

    // // MinMaxScaler to normalize each column after projection
    // Eigen::MatrixXd pos_projections_normalized = SCALEDATA(X * candidates.transpose());
    // Eigen::MatrixXd neg_projections_normalized = SCALEDATA(Y * candidates.transpose());



    // std::cout<<X.array().isNaN().any()<<std::endl;
    // std::cout<<Y.array().isNaN().any()<<std::endl;
    // std::cout<<candidates.array().isNaN().any()<<std::endl;

    Eigen::MatrixXi pos_hash_values = (SCALEDATA(X * candidates.transpose()).array().abs() * (delta - 1)).cast<int>();
    Eigen::MatrixXi neg_hash_values = (SCALEDATA(Y * candidates.transpose()).array().abs() * (delta - 1)).cast<int>();

    // std::ofstream outFile("hash_values.csv");
    // outFile << "pos_hash_values:" << std::endl;
    // for (int i = 0; i < pos_hash_values.rows(); ++i) {
    //     for (int j = 0; j < pos_hash_values.cols(); ++j) {
    //         outFile << pos_hash_values(i, j) << ",";
    //     }
    //     outFile << std::endl;
    // }
    // outFile << "neg_hash_values:" << std::endl;
    // for (int i = 0; i < neg_hash_values.rows(); ++i) {
    //     for (int j = 0; j < neg_hash_values.cols(); ++j) {
    //         outFile << neg_hash_values(i, j) << ",";
    //     }
    //     outFile << std::endl;
    // }
    // outFile.close();

    // Compute the overlap of the hash values of X and Y
    Eigen::VectorXi overlaps(sample_size);
    for (int i = 0; i < sample_size; ++i) {
        // overlap[i] is the size of the intersect between the set of pos_hash_values.col(i) and the set of neg_hash_values.col(i)
        std::set<int> pos_set(pos_hash_values.col(i).data(), pos_hash_values.col(i).data() + pos_hash_values.col(i).size());
        std::set<int> neg_set(neg_hash_values.col(i).data(), neg_hash_values.col(i).data() + neg_hash_values.col(i).size());
        std::vector<int> intersect;
        std::set_intersection(pos_set.begin(), pos_set.end(), neg_set.begin(), neg_set.end(), std::back_inserter(intersect));
        overlaps[i] = intersect.size();
    }

    Eigen::VectorXi best_hashes_idx(sample_size);
    for (int i = 0; i < sample_size; ++i) {
        best_hashes_idx[i] = i;
    }

    // Sort the hash functions by the overlap
    std::sort(best_hashes_idx.data(), best_hashes_idx.data() + best_hashes_idx.size(),
              [&overlaps](int i, int j) { return overlaps[i] < overlaps[j]; });

    // Select the best hash functions
    Eigen::MatrixXd best_hashes(hash_count, dim);
    for (int i = 0; i < hash_count; ++i) {
        best_hashes.row(i) = candidates.row(best_hashes_idx[i]);
    }

    std::ofstream outFile1("overlaps&best_overlaps.csv");
    // outFile1 << "overlaps:" << std::endl;
    // for (int i = 0; i < overlaps.size(); ++i) {
    //     outFile1 << overlaps[i] << ",";
    // }
    // outFile1 << std::endl;
    outFile1 << "best_overlaps:" << std::endl;
    for (int i = 0; i < hash_count; ++i) {
         outFile1 << overlaps[best_hashes_idx[i]] << ",";
    }
    outFile1.close();

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
    // Eigen::MatrixXd projections = X * vectors;

    // // MinMaxScaler to normalize each column after projection
    // Eigen::MatrixXd projections_normalized = SCALEDATA(X * vectors);

    // Eigen::MatrixXi hash_values = (SCALEDATA(X * vectors).array().abs() * (delta - 1)).cast<int>();

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

    return (SCALEDATA(X * vectors).array().abs() * (delta - 1)).cast<int>();
}

void PHBF::bulk_add(const Eigen::MatrixXd& X) {
    Eigen::MatrixXi indexes = compute_hashes(X);

    std::ofstream outFile("pos_hash_values.csv");
    outFile << "pos_hash_values:" << std::endl;
    for (int i = 0; i < indexes.rows(); ++i) {
        for (int j = 0; j < indexes.cols(); ++j) {
            outFile << indexes(i, j) << ",";
        }
        outFile << std::endl;
    }
    outFile.close();

    for (int i = 0; i < indexes.cols(); ++i) {
        for (int j = 0; j < indexes.rows(); ++j) {
            int hash_value = indexes(j, i);
            bit_array[i].set(hash_value);
        }
    }
}

bool* PHBF::lookup(const Eigen::MatrixXd& X) {
    bool* results = new bool[X.rows()];
    std::fill(results, results + X.rows(), true);
    Eigen::MatrixXi hash_values = compute_hashes(X);

    std::ofstream outFile("neg_hash_values.csv");
    outFile << "neg_hash_values:" << std::endl;
    for (int i = 0; i < hash_values.rows(); ++i) {
        for (int j = 0; j < hash_values.cols(); ++j) {
            outFile << hash_values(i, j) << ",";
        }
        outFile << std::endl;
    }
    outFile.close();

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

long double PHBF::compute_fpr(const Eigen::MatrixXd& X) {
    long double fp = 0;
    long double tn = 0;
    bool* results = lookup(X);

    for (int i = 0; i < X.rows(); ++i) {
        if (results[i]) {
            fp++;
        }
        else {
            tn++;
        }
    }

    return static_cast<long double>(fp) / (fp + tn);
}
