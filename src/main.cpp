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
#include <unistd.h>
#include "tools.h"
#include <filesystem>

using namespace std::chrono;


using return_type = std::invoke_result_t<decltype(loadEmber)>;
using load_func = std::function<return_type()>;
std::unordered_map<std::string,load_func> loaders;
std::unordered_map<std::string, unsigned> dims = {
    {"mnist",784},
    {"malicious_urls",79},
    {"higgs",28},
    {"kitsune",116},
    {"ember",2381}
};

// void testRandom() {
//     const auto size = 6400;
//     int hash_count = size / DELTA;
//     int dim = 100;
//     int sample_factor = 100;
//     double per = (double) size / (1000 * 8);
//     auto phbf = std::make_unique<PHBF>(bpk, n, dim, sample_factor, "gaussian");
//     Eigen::MatrixXd X = Eigen::MatrixXd::Random(1000, dim);
//     Eigen::MatrixXd Y = Eigen::MatrixXd::Random(1000, dim);
//     phbf->initialize(X, Y);
//     phbf->bulk_add(X);
//     std::cout << "FPR: " << phbf->compute_fpr(Y) 
//                   << " ,byte per key: " << per 
//                   << " ,hash functions: " << hash_count 
//                   << std::endl;
// }

void testDataset(const std::string& dataset, const std::string& filter, const unsigned sample_factor, const double bpk) {
    
    
    //int sample_factor = 100;
    //double bytes_per_element[] = {0.01,0.08,0.16,0.23,0.3,0.4,0.5,0.6};
    if(filter == "phbf"){
        auto data{loaders[dataset]()};
        const unsigned dim = data->X_train.cols();
        //int hash_count = ceil((data->X_train.rows() * bpk * 8) / (DELTA + dim));
        //int total_size = hash_count * (DELTA + dim);
        //int size = hash_count * DELTA;
        const uint64_t n = data->X_train.rows();
        const size_t len = ceil((bpk * 8 * n ) / HASH_COUNT);
        assert(len > dim * sizeof(double) * 8);
        const double per = (len * HASH_COUNT) / (n * 8.0);
        const std::string phbf_csv = "phbf_" + dataset + ".csv";
        if(!std::filesystem::exists(phbf_csv)){
            std::ofstream outFile(phbf_csv);
            outFile << "byte_per_key,FPR,construction_time,query_time,model_size,array_size" << std::endl;
            outFile.close();
        }
        std::ofstream outFile(phbf_csv, std::ios_base::app);
        auto phbf = std::make_unique<PHBF>(bpk, n, dim, sample_factor, "gaussian");

        // auto t1 = steady_clock::now();
        // phbf->initialize(data->X_train, data->X_test);
        // phbf->bulk_add(data->X_train);
        // auto t2 = steady_clock::now();
        // auto construction_time = duration_cast<milliseconds>(t2 - t1).count();
        auto [_, construction_time] = TIME(&PHBF::initandadd, phbf.get(), data->X_train, data->X_test);

        std::cout << "PHBF size: " << phbf->model_size() << " + " 
                                   << phbf->array_size() << " = " 
                                   << phbf->array_size() + phbf->model_size() 
                                   << "Bytes" << std::endl; 

        auto [fpr, query_time] = TIME(&PHBF::compute_fpr, phbf.get(), data->X_test);

        outFile << per << "," << fpr.value() << "," << construction_time << "," << query_time << "," 
                << phbf->model_size() << "," << phbf->array_size() << std::endl;
        phbf = nullptr;
        data = nullptr;
        outFile.close();
    }
    else if (filter == "habf"){
        std::string habf_csv = "habf_" + dataset + ".csv";
        dataloader dl;
        dl.load(dataset,false);
        unsigned dim = dims[dataset];
        //int hash_count = ceil((dl.pos_keys_.size() * bpk * 8) / (DELTA + dim));
        //int total_size = hash_count * (DELTA + dim);
        //int size = hash_count * DELTA;
        const uint64_t n = dl.pos_keys_.size();
        double per = (ceil((bpk * 8 * n ) / HASH_COUNT) * HASH_COUNT) / (n * 8);
        //double per = (double) total_size / (dl.pos_keys_.size() * 8);
        if(!std::filesystem::exists(habf_csv)){
            std::ofstream outFile(habf_csv);
            outFile << "byte_per_key,FPR,construction_time,query_time,model_size,array_size" << std::endl;
            outFile.close();
        }
        std::ofstream outFile(habf_csv, std::ios_base::app);
        habf::HABFilter habf(per*8, dl.pos_keys_.size());

        auto [_, construction_time] = TIME(&habf::HABFilter::AddAndOptimize, &habf, dl.pos_keys_,dl.neg_keys_);

        std::cout << "HABF size: " << habf.getModelSize() / 8 << " + " 
                                   << habf.getArraySize() / 8<< " = " 
                                   << habf.getArraySize() / 8 + habf.getModelSize() / 8 
                                   << "Bytes" << std::endl;

        auto [fpr, query_time] = TIME(&habf::HABFilter::compute_fpr, &habf, dl.neg_keys_);

        outFile << per << "," << fpr.value() << "," << construction_time << "," << query_time << "," 
                << habf.getModelSize() / 8 << "," << habf.getArraySize() / 8 << std::endl;
        outFile.close();
    }
    else{
        std::cout << "Invalid Filter" << std::endl;
        std::cout << "Valid Filters: phbf, habf" << std::endl;
        return;
    }
    std::cout << "Dataset: " << dataset << " Complete!" << std::endl;
    std::cout << "==============================" << std::endl;
}


int main(int argc, char** argv) {

    loaders["mnist"] = std::bind(&loadMnist,std::vector<int>{0},6000,6000);
    loaders["malicious_urls"] = std::bind(&loadMaliciousUrls,16273,2709);
    loaders["higgs"] = std::bind(&loadHiggs,25000,25000);
    loaders["kitsune"] = std::bind(&loadKitsune,"Mirai");
    loaders["ember"] = std::bind(&loadEmber);

    char o = '0';
    unsigned dim = 0;
    unsigned sample_factor = 0;
    double bpr = 0.0;
    std::string dataset = "";
    std::string filter = "";
    while((o = getopt(argc, argv, "f:s:b:d:h::")) != -1){
        switch(o){
            case 's':
                sample_factor = atoi(optarg);
                break;
            case 'b':
                bpr = atof(optarg);
                break;
            case 'd':
                dataset = optarg;
                break;
            case 'f':
                filter = optarg;
                break;
            case 'h':
                std::cout << "Usage: ./PHBF_CPP -s sample_factor -b bytes_per_element -d dataset -f filter" << std::endl;
                std::cout << "Valid Datasets: mnist, malicious_urls, higgs, kitsune, ember" << std::endl;
                std::cout << "Valid Filters: phbf, habf" << std::endl;
                return 0;
            default:
                std::cout << "Invalid option" << std::endl;
                std::cout << "Usage: ./PHBF_CPP -s sample_factor -b bytes_per_element -d dataset" << std::endl;
                std::cout << "Valid Datasets: mnist, malicious_urls, higgs, kitsune, ember" << std::endl;
                std::cout << "Valid Filters: phbf, habf" << std::endl;
                return 1;
        }
    }

    if(loaders.find(dataset) == loaders.end()){
        std::cout << "Invalid Dataset" << std::endl;
        std::cout << "Valid Datasets: mnist, malicious_urls, higgs, kitsune, ember" << std::endl;
        return 1;
    }

    testDataset(dataset,filter,sample_factor,bpr);

    //testRandom();
    // testMaliciousUrls();
    // std::cout << "Malicious Urls Complete!" << std::endl;
    // std::cout << "==============================" << std::endl;
    // testMnist();
    // std::cout << "Mnist Complete!" << std::endl;
    // std::cout << "==============================" << std::endl;
    // testKitsune();
    // std::cout << "Kitsune Complete!" << std::endl;
    // std::cout << "==============================" << std::endl;
    // testHIGGS();
    // std::cout << "HIGGS Complete!" << std::endl;
    // std::cout << "==============================" << std::endl;
    return 0;
}

/*
void testMnist(const unsigned dim, const unsigned sample_factor, const double bpr) {
    auto data{loadMnist({0})};
    dataloader dl;
    dl.load(data->X_train, data->X_test);
    //int dim = 784;
    //int sample_factor = 100;
    //double bytes_per_element[] = {0.01,0.08,0.16,0.23,0.3,0.4,0.5,0.6};

    std::ofstream outFile1("phbf_mnist.csv", std::ios_base::app);
    outFile1 << "byte_per_key,FPR,construction_time,query_time" << std::endl;
    std::ofstream outFile2("habf_mnist.csv", std::ios_base::app);
    outFile2 << "byte_per_key,FPR,construction_time,query_time" << std::endl;

    {
        int hash_count = ceil((data->X_train.rows() * bpr * 8) / (DELTA + dim));
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
}

void testMaliciousUrls(const unsigned dim, const unsigned sample_factor, const double bpr){
    auto data{loadMaliciousUrls()};
    dataloader dl;
    dl.load(data->X_train, data->X_test);
    //int dim = 79;
    //int sample_factor = 100;
    //double bytes_per_element[] = {0.01,0.08,0.16,0.23,0.3,0.4,0.5,0.6};
    std::ofstream outFile1("phbf_malicious_urls.csv", std::ios_base::app);
    outFile1 << "byte_per_key,FPR,construction_time,query_time" << std::endl;
    std::ofstream outFile2("habf_malicious_urls.csv", std::ios_base::app);
    outFile2 << "byte_per_key,FPR,construction_time,query_time" << std::endl;

    {
        int hash_count = ceil((data->X_train.rows() * bpr * 8) / (DELTA + dim));
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
    //delete data;
}

void testHIGGS(const unsigned dim, const unsigned sample_factor, const double bpr){
    auto data{loadHiggs(25000,25000)};
    dataloader dl;
    dl.load(data->X_train, data->X_test);
    //std::cout << data->X_train.rows() << "," << data->X_test.rows() << std::endl;
    //std::cout << dl.pos_keys_.size() << "," << dl.neg_keys_.size() << std::endl;
    //int dim = 28;
    //int sample_factor = 100;
    //double bytes_per_element[] = {0.01,0.08,0.16,0.23,0.3};
    std::ofstream outFile1("phbf_higgs.csv", std::ios_base::app);
    outFile1 << "byte_per_key,FPR,construction_time,query_time" << std::endl;
    std::ofstream outFile2("habf_higgs.csv", std::ios_base::app);
    outFile2 << "byte_per_key,FPR,construction_time,query_time" << std::endl;

    {
        int hash_count = ceil((data->X_train.rows() * bpr * 8) / (DELTA + dim));
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
    //delete data;
}

void testKitsune(const unsigned dim, const unsigned sample_factor, const double bpr){
    auto data{loadKitsune()};
    dataloader dl;
    dl.load(data->X_train, data->X_test);
    //std::cout << data->X_train.rows() << "," << data->X_test.rows() << std::endl;
    //std::cout << dl.pos_keys_.size() << "," << dl.neg_keys_.size() << std::endl;
    //int dim = 116;
    //int sample_factor = 100;
    //double bytes_per_element[] = {0.01,0.08,0.16,0.23,0.3};
    std::ofstream outFile1("phbf_kitsune.csv", std::ios_base::app);
    outFile1 << "byte_per_key,FPR,construction_time,query_time" << std::endl;
    std::ofstream outFile2("habf_kitsune.csv", std::ios_base::app);
    outFile2 << "byte_per_key,FPR,construction_time,query_time" << std::endl;

    {
        int hash_count = ceil((data->X_train.rows() * bpr * 8) / (DELTA + dim));
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
    //delete data;
}

void testEmber(const unsigned sample_factor, const double bpr){
    auto data{loadEmber()};
    dataloader dl;
    dl.load(data->X_train, data->X_test);
    //std::cout << data->X_train.rows() << "," << data->X_test.rows() << std::endl;
    //std::cout << dl.pos_keys_.size() << "," << dl.neg_keys_.size() << std::endl;
    const auto dim = 2381;
    //int sample_factor = 100;
    //double bytes_per_element[] = {0.01,0.08,0.16,0.23,0.3};
    std::ofstream outFile1("phbf_ember.csv", std::ios_base::app);
    outFile1 << "byte_per_key,FPR,construction_time,query_time" << std::endl;
    std::ofstream outFile2("habf_ember.csv", std::ios_base::app);
    outFile2 << "byte_per_key,FPR,construction_time,query_time" << std::endl;

    {
        int hash_count = ceil((data->X_train.rows() * bpr * 8) / (DELTA + dim));
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
    //delete data;
}
*/