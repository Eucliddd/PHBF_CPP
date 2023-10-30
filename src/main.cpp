#include <iostream>
#include "phbf.h"
#include "data_loader.h"
#define DELTA 32

void testMnist() {
    Data *data = loadMnist({0, 1, 4});
    int dim = 784;
    int sample_factor = 100;
    double bytes_per_element[5] = {0.01,0.08,0.16,0.23,0.3};
    for (int i = 0; i < 5; ++i) {
        int size = (int) (data->X.rows() * bytes_per_element[i] * 8);
        int hash_count = size / DELTA;
        PHBF *phbf = new PHBF(size, hash_count, dim, sample_factor, "gaussian");
        phbf->initialize(data->X, data->X_test);
        std::cout << "FPR: " << phbf->compute_fpr(data->X_test) << std::endl;
    }
}

int main() {
    testMnist();
    std::cout << "Hello, World!\n";
    return 0;
}