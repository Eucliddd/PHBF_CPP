#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif

#ifndef EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_SSE4_2
#endif

#include <iostream>
#include "phbf.h"
#include "data_loader.h"
#define DELTA 32

void testRandom() {
    int size = 3200;
    int hash_count = size / DELTA;
    int dim = 100;
    int sample_factor = 100;
    PHBF *phbf = new PHBF(size, hash_count, dim, sample_factor, "gaussian");
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(1000, dim);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Random(1000, dim);
    phbf->initialize(X, Y);
    phbf->bulk_add(X);
    std::cout << "FPR: " << phbf->compute_fpr(Y) << std::endl;
}

void testMnist() {
    Data *data = loadMnist({0, 1, 4});
    int dim = 784;
    int sample_factor = 100;
    double bytes_per_element[5] = {0.01,0.08,0.16,0.23,0.3};
    //for (int i = 0; i < 5; ++i) {
        int size = (int) (data->X_train.rows() * 0.1 * 8);
        int hash_count = size / DELTA;
        PHBF *phbf = new PHBF(size, hash_count, dim, sample_factor, "gaussian");
        phbf->initialize(data->X_train, data->X_test);
        std::cout << "FPR: " << phbf->compute_fpr(data->X_test) << std::endl;
    //}
}

void testMaliciousUrls(){
    Data *data = loadMaliciousUrls(16000,2700);
    int dim = 79;
    int sample_factor = 100;
    double bytes_per_element[5] = {0.01,0.08,0.16,0.23,0.3};
    //for (int i = 0; i < 5; ++i) {
        int size = (int) (data->X_train.rows() * bytes_per_element[1] * 8);
        int hash_count = size / DELTA;
        PHBF *phbf = new PHBF(size, hash_count, dim, sample_factor, "gaussian");
        phbf->initialize(data->X_train, data->X_test);
        phbf->bulk_add(data->X_train);
        std::cout << "FPR: " << phbf->compute_fpr(data->X_test) << std::endl;
        free(phbf);
    //}
    free(data);
}

int main() {
    testMnist();
    //std::cout << "Hello, World!\n";
    return 0;
}