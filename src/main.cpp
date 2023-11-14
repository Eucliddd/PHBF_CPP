#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif

#ifndef EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_SSE4_2
#endif

#include <iostream>
#include <fstream>
#include "phbf.h"
#include "data_loader.h"
#include "habf.h"
#include "dataloader.h"
#include <chrono>

using namespace std::chrono;

void testRandom() {
    int size = 6400;
    int hash_count = size / DELTA;
    int dim = 100;
    int sample_factor = 100;
    double per = (double) size / (1000 * 8);
    PHBF *phbf = new PHBF(hash_count, dim, sample_factor, "gaussian");
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(1000, dim);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Random(1000, dim);
    phbf->initialize(X, Y);
    phbf->bulk_add(X);
    std::cout << "FPR: " << phbf->compute_fpr(Y) 
                  << " ,byte per key: " << per 
                  << " ,hash functions: " << hash_count 
                  << std::endl;
}

void testMnist() {
    Data *data = loadMnist({0, 1, 4});
    dataloader dl;
    dl.load(data->X_train, data->X_test);
    int dim = 784;
    int sample_factor = 100;
    double bytes_per_element[5] = {0.01,0.08,0.16,0.23,0.3};
    std::ofstream outFile1("phbf_mnist.csv");
    outFile1 << "byte_per_key,FPR,construction_time,query_time" << std::endl;
    std::ofstream outFile2("habf_mnist.csv");
    outFile2 << "byte_per_key,FPR,construction_time,query_time" << std::endl;

    for (int i = 0; i < 5; ++i) {
        int hash_count = ceil((data->X_train.rows() * bytes_per_element[i] * 8) / (DELTA + dim));
        int total_size = hash_count * (DELTA + dim);
        //int size = hash_count * DELTA;
        double per = (double) total_size / (data->X_train.rows() * 8);

        PHBF* phbf = new PHBF(hash_count, dim, sample_factor, "gaussian");

        auto t1 = steady_clock::now();
        phbf->initialize(data->X_train, data->X_test);
        phbf->bulk_add(data->X_train);
        auto t2 = steady_clock::now();
        double construction_time = duration_cast<milliseconds>(t2 - t1).count();

        t1 = steady_clock::now();
        long double fpr = phbf->compute_fpr(data->X_test);
        t2 = steady_clock::now();
        double query_time = duration_cast<milliseconds>(t2 - t1).count();

        outFile1 << per << "," << fpr << "," << construction_time << "," << query_time << std::endl;
        delete phbf;
        phbf = nullptr;

        habf::HABFilter habf(per*8, dl.pos_keys_.size());

        t1 = steady_clock::now();
        habf.AddAndOptimize(dl.pos_keys_,dl.neg_keys_);
        t2 = steady_clock::now();
        construction_time = duration_cast<milliseconds>(t2 - t1).count();

        t1 = steady_clock::now();
        fpr = habf.compute_fpr(dl.neg_keys_);
        t2 = steady_clock::now();
        query_time = duration_cast<milliseconds>(t2 - t1).count();

        outFile2 << per << "," << fpr << "," << construction_time << "," << query_time << std::endl;
    }
    outFile1.close();
    outFile2.close();
    delete data;
}

void testMaliciousUrls(){
    Data *data = loadMaliciousUrls();
    dataloader dl;
    dl.load(data->X_train, data->X_test);
    int dim = 79;
    int sample_factor = 100;
    double bytes_per_element[5] = {0.01,0.08,0.16,0.23,0.3};
    std::ofstream outFile1("phbf_malicious_urls.csv");
    outFile1 << "byte_per_key,FPR,construction_time,query_time" << std::endl;
    std::ofstream outFile2("habf_malicious_urls.csv");
    outFile2 << "byte_per_key,FPR,construction_time,query_time" << std::endl;

    for (int i = 0; i < 5; ++i) {
        int hash_count = ceil((data->X_train.rows() * bytes_per_element[i] * 8) / (DELTA + dim));
        int total_size = hash_count * (DELTA + dim);
        //int size = hash_count * DELTA;
        double per = (double) total_size / (data->X_train.rows() * 8);

        PHBF* phbf = new PHBF(hash_count, dim, sample_factor, "gaussian");

        auto t1 = steady_clock::now();
        phbf->initialize(data->X_train, data->X_test);
        phbf->bulk_add(data->X_train);
        auto t2 = steady_clock::now();
        double construction_time = duration_cast<milliseconds>(t2 - t1).count();

        t1 = steady_clock::now();
        long double fpr = phbf->compute_fpr(data->X_test);
        t2 = steady_clock::now();
        double query_time = duration_cast<milliseconds>(t2 - t1).count();
        delete phbf;
        phbf = nullptr;

        outFile1 << per << "," << fpr << "," << construction_time << "," << query_time << std::endl;
        
        habf::HABFilter habf(per*8, dl.pos_keys_.size());

        t1 = steady_clock::now();
        habf.AddAndOptimize(dl.pos_keys_,dl.neg_keys_);
        t2 = steady_clock::now();
        construction_time = duration_cast<milliseconds>(t2 - t1).count();

        t1 = steady_clock::now();
        fpr = habf.compute_fpr(dl.neg_keys_);
        t2 = steady_clock::now();
        query_time = duration_cast<milliseconds>(t2 - t1).count();

        outFile2 << per << "," << fpr << "," << construction_time << "," << query_time << std::endl;
    }
    outFile1.close();
    outFile2.close();
    delete data;
}

void testHIGGS(){
    Data* data = loadHiggs(25000,25000);
    dataloader dl;
    dl.load(data->X_train, data->X_test);
    //std::cout << data->X_train.rows() << "," << data->X_test.rows() << std::endl;
    //std::cout << dl.pos_keys_.size() << "," << dl.neg_keys_.size() << std::endl;
    int dim = 28;
    int sample_factor = 100;
    double bytes_per_element[5] = {0.01,0.08,0.16,0.23,0.3};
    std::ofstream outFile1("phbf_higgs.csv");
    outFile1 << "byte_per_key,FPR,construction_time,query_time" << std::endl;
    std::ofstream outFile2("habf_higgs.csv");
    outFile2 << "byte_per_key,FPR,construction_time,query_time" << std::endl;

    for (int i = 0; i < 5; ++i) {
        int hash_count = ceil((data->X_train.rows() * bytes_per_element[i] * 8) / (DELTA + dim));
        int total_size = hash_count * (DELTA + dim);
        //int size = hash_count * DELTA;
        double per = (double) total_size / (data->X_train.rows() * 8);

        PHBF* phbf = new PHBF(hash_count, dim, sample_factor, "gaussian");

        auto t1 = steady_clock::now();
        phbf->initialize(data->X_train, data->X_test);
        phbf->bulk_add(data->X_train);
        auto t2 = steady_clock::now();
        double construction_time = duration_cast<milliseconds>(t2 - t1).count();

        t1 = steady_clock::now();
        long double fpr = phbf->compute_fpr(data->X_test);
        t2 = steady_clock::now();
        double query_time = duration_cast<milliseconds>(t2 - t1).count();

        outFile1 << per << "," << fpr << "," << construction_time << "," << query_time << std::endl;
        delete phbf;
        phbf = nullptr;

        habf::HABFilter habf(per*8, dl.pos_keys_.size());

        t1 = steady_clock::now();
        habf.AddAndOptimize(dl.pos_keys_,dl.neg_keys_);
        t2 = steady_clock::now();
        construction_time = duration_cast<milliseconds>(t2 - t1).count();

        t1 = steady_clock::now();
        fpr = habf.compute_fpr(dl.neg_keys_);
        t2 = steady_clock::now();
        query_time = duration_cast<milliseconds>(t2 - t1).count();

        outFile2 << per << "," << fpr << "," << construction_time << "," << query_time << std::endl;
    }
    outFile1.close();
    outFile2.close();
    delete data;
}

int main() {
    //testRandom();
    // testMaliciousUrls();
    // std::cout << "Malicious Urls Complete!" << std::endl;
    // std::cout << "==============================" << std::endl;
    // testMnist();
    // std::cout << "Mnist Complete!" << std::endl;
    // std::cout << "==============================" << std::endl;
    testHIGGS();
    std::cout << "HIGGS Complete!" << std::endl;
    std::cout << "==============================" << std::endl;
    return 0;
}