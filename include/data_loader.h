#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <assert.h>
#include <random>
#include <algorithm>
#include <arpa/inet.h>
#include <Eigen/Dense>
#include "tools.h"
#include <cassert>
#include <memory>

class Data {
public:
    Eigen::MatrixXd X_train;
    Eigen::MatrixXd X_test;
};


/**
 * @brief Load MNIST data and perform operations as needed
 * @param posDigits: vector of digits to be considered as positive
 * @param numPos: number of positive examples to be considered
 * @param numNeg: number of negative examples to be considered
 * @return X_train: training data
 * @return X_test: test data
 * @note X, X_test are all Eigen::MatrixXd
*/
auto loadMnist(const std::vector<int>& posDigits, const int numPos=6000, const int numNeg=6000) {
    // Load MNIST data
    std::cout << "Loading MNIST data..." << std::endl;
    const std::string mnistPath = "/data/mnist/";
    std::string trainImagesPath = mnistPath + "train-images.idx3-ubyte";
    std::string trainLabelsPath = mnistPath + "train-labels.idx1-ubyte";
    std::string testImagesPath = mnistPath + "t10k-images.idx3-ubyte";
    std::string testLabelsPath = mnistPath + "t10k-labels.idx1-ubyte";

    std::ifstream trainImagesFile(trainImagesPath, std::ios::binary);
    std::ifstream trainLabelsFile(trainLabelsPath, std::ios::binary);
    std::ifstream testImagesFile(testImagesPath, std::ios::binary);
    std::ifstream testLabelsFile(testLabelsPath, std::ios::binary);

    if (!trainImagesFile.is_open()) {
        std::cerr << "ERROR: Cannot open train-images-idx3-ubyte" << std::endl;
        exit(1);
    }

    if (!trainLabelsFile.is_open()) {
        std::cerr << "ERROR: Cannot open train-labels-idx1-ubyte" << std::endl;
        exit(1);
    }

    if (!testImagesFile.is_open()) {
        std::cerr << "ERROR: Cannot open t10k-images-idx3-ubyte" << std::endl;
        exit(1);
    }

    if (!testLabelsFile.is_open()) {
        std::cerr << "ERROR: Cannot open t10k-labels-idx1-ubyte" << std::endl;
        exit(1);
    }

    auto _Result = std::make_unique<Data>();

    // Read training images
    int magicNumber1 = 0, magicNumber2 = 0, numImages1 = 0, numImages2 = 0, numRows = 0, numCols = 0;
    trainImagesFile.read((char*)&magicNumber1, sizeof(magicNumber1));
    trainImagesFile.read((char*)&numImages1, sizeof(numImages1));
    trainImagesFile.read((char*)&numRows, sizeof(numRows));
    trainImagesFile.read((char*)&numCols, sizeof(numCols));
    trainLabelsFile.read((char*)&magicNumber2, sizeof(magicNumber2));
    trainLabelsFile.read((char*)&numImages2, sizeof(numImages2));
    magicNumber1 = ntohl(magicNumber1);
    numImages1 = ntohl(numImages1);
    numRows = ntohl(numRows);
    numCols = ntohl(numCols);
    magicNumber2 = ntohl(magicNumber2);
    numImages2 = ntohl(numImages2);
    assert(numImages1 == numImages2);

    // Read training data
    Eigen::MatrixXd X(numImages1, numRows * numCols);
    //Eigen::MatrixXd Y(numImages2, 10);
    _Result->X_train.resize(numPos, numRows * numCols);
    //_Result->Y = Eigen::MatrixXd(numPos, 10);
    //long posCount = 0, negCount = 0;
    std::vector<int> sampleIdx;
    for (int i = 0; i < numImages1; ++i) {
        auto pixels = std::make_unique<char[]>(numRows * numCols);
        trainImagesFile.read(pixels.get(), numRows * numCols);
        for (int j = 0; j < numRows * numCols; ++j) {
            X(i, j) = (unsigned char)pixels[j];
        }

        char label;
        trainLabelsFile.read(&label, 1);
        int labelInt = (unsigned char)label;
        if (std::find(posDigits.begin(), posDigits.end(), labelInt) != posDigits.end()) {
            //Y(i, labelInt) = 1;
            //++posCount;
            sampleIdx.push_back(i);
        }
        else {
            //Y(i, labelInt) = 0;
            //++negCount;
        }
    }

    // Randomly select numPos examples
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(sampleIdx.begin(), sampleIdx.end(), g);
    for (int i = 0; i < numPos; ++i) {
        _Result->X_train.row(i) = X.row(sampleIdx[i]);
        //_Result->Y.row(i) = Y.row(sampleIdx[i]);
    }


    // Read test images
    testImagesFile.read((char*)&magicNumber1, sizeof(magicNumber1));
    testImagesFile.read((char*)&numImages1, sizeof(numImages1));
    testImagesFile.read((char*)&numRows, sizeof(numRows));
    testImagesFile.read((char*)&numCols, sizeof(numCols));
    testLabelsFile.read((char*)&magicNumber2, sizeof(magicNumber2));
    testLabelsFile.read((char*)&numImages2, sizeof(numImages2));
    magicNumber1 = ntohl(magicNumber1);
    numImages1 = ntohl(numImages1);
    numRows = ntohl(numRows);
    numCols = ntohl(numCols);
    magicNumber2 = ntohl(magicNumber2);
    numImages2 = ntohl(numImages2);
    assert(numImages1 == numImages2);

    // Read test data
    Eigen::MatrixXd X_test(numImages1, numRows * numCols);
    //Eigen::MatrixXd Y_test(numImages2, 10);
    _Result->X_test.resize(numNeg, numRows * numCols);
    //_Result->Y_test = Eigen::MatrixXd(numNeg, 10);
    //posCount = 0, negCount = 0;
    sampleIdx.clear();
    for (int i = 0; i < numImages1; ++i) {
        auto pixels = std::make_unique<char[]>(numRows * numCols);
        testImagesFile.read(pixels.get(), numRows * numCols);
        for (int j = 0; j < numRows * numCols; ++j) {
            X_test(i, j) = (unsigned char)pixels[j];
        }

        char label;
        testLabelsFile.read(&label, 1);
        int labelInt = (unsigned char)label;
        if (std::find(posDigits.begin(), posDigits.end(), labelInt) != posDigits.end()) {
            //Y_test(i, labelInt) = 1;
            //++posCount;
        }
        else {
            //Y_test(i, labelInt) = 0;
            //++negCount;
            sampleIdx.push_back(i);
        }
    }

    // Randomly select numNeg examples
    std::shuffle(sampleIdx.begin(), sampleIdx.end(),g);
    for (int i = 0; i < numNeg; ++i) {
        _Result->X_test.row(i) = X_test.row(sampleIdx[i]);
        //_Result->Y_test.row(i) = Y_test.row(sampleIdx[i]);
    }

    // MinMaxScale each feature
    _Result->X_train = std::move(SCALEDATA(_Result->X_train));
    _Result->X_test = std::move(SCALEDATA(_Result->X_test));

    // Close files
    trainImagesFile.close();
    trainLabelsFile.close();
    testImagesFile.close();
    testLabelsFile.close();

    std::cout << "MNIST data loaded." << std::endl;

    // return X, X_test;
    return _Result;
}

/**
 * @brief Load KITSUNE data and perform operations as needed
 * @param attack: name of the attack to be considered
 * @return X_train: training data
 * @return X_test: test data
 * @note X, X_test are all Eigen::MatrixXd
*/
auto loadKitsune(const std::string& attack="Mirai"){

    std::cout << "Loading KITSUNE data..." << std::endl;

    std::ifstream xFile("/data/Kitsune/" + attack + "_dataset.csv");
    std::ifstream yFile("/data/Kitsune/" + attack + "_labels.csv");
    if (!xFile || !yFile) {
        std::cerr << "Failed to open data files." << std::endl;
        exit(1);
    }

    auto _Result = std::make_unique<Data>();

    // Read data
    std::vector<std::vector<double>> xData;
    std::vector<int> yData;
    std::vector<size_t> posIdx;
    std::vector<size_t> negIdx;
    std::string line;
    long posCount = 0, negCount = 0;
    while(std::getline(xFile, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while(std::getline(ss, cell, ',')) {
            try
            {
                row.push_back(std::stod(cell));
            }
            catch(const std::exception& e)
            {
                //std::cerr << e.what() << std::endl;
                row.push_back(0);
            }
        }
        xData.push_back(row);
    }

    while(std::getline(yFile, line)) {
        int label = std::stoi(line);
        yData.push_back(label);
        if (label == 1) {
            ++posCount;
            posIdx.push_back(yData.size() - 1);
        }
        else {
            ++negCount;
            negIdx.push_back(yData.size() - 1);
        }
    }

    // Convert to Eigen::MatrixXd
    _Result->X_train = Eigen::MatrixXd(posCount, xData[0].size());
    _Result->X_test = Eigen::MatrixXd(negCount, xData[0].size());

    for(size_t i=0; i < posCount; ++i) {
        _Result->X_train.row(i) = Eigen::VectorXd::Map(xData[posIdx[i]].data(), xData[posIdx[i]].size());
    }
    for(size_t i=0; i < negCount; ++i) {
        _Result->X_test.row(i) = Eigen::VectorXd::Map(xData[negIdx[i]].data(), xData[negIdx[i]].size());
    }
    
    // MinMaxScale each feature
    _Result->X_train = std::move(SCALEDATA(_Result->X_train)); 
    _Result->X_test = std::move(SCALEDATA(_Result->X_test));

    std::cout << "KITSUNE data loaded." << std::endl;

    return _Result;
}

auto loadEmber(){
    std::cout << "Loading EMBER data..." << std::endl;

    std::ifstream xFile("/data/EMBER/ember_pos.csv");
    std::ifstream yFile("/data/EMBER/ember_neg.csv");

    if (!xFile || !yFile) {
        std::cerr << "Failed to open data files." << std::endl;
        exit(1);
    }

    auto _Result = std::make_unique<Data>();
    _Result->X_train.resize(400000, 2381);
    _Result->X_test.resize(400000, 2381);
    // Read data
    std::string line;
    int i = 0;
    while(std::getline(xFile, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while(std::getline(ss, cell, ',')) {
            try
            {
                row.push_back(std::stod(cell));
            }
            catch(const std::exception& e)
            {
                row.push_back(0);
            }
        }
        _Result->X_train.row(i++) = Eigen::VectorXd::Map(row.data(), row.size());
    }
    std::cout << "X_train loaded:" << i << std::endl;
    
    i = 0;
    while(std::getline(yFile, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while(std::getline(ss, cell, ',')) {
            try
            {
                row.push_back(std::stod(cell));
            }
            catch(const std::exception& e)
            {
                row.push_back(0);
            }
        }
        _Result->X_test.row(i++) = Eigen::VectorXd::Map(row.data(), row.size());
    }
    std::cout << "X_test loaded:" << i << std::endl;

    // MinMaxScale each feature
    _Result->X_train = std::move(SCALEDATA(_Result->X_train));
    _Result->X_test = std::move(SCALEDATA(_Result->X_test));

    std::cout << "EMBER data loaded." << std::endl;

    return _Result;
}

/**
 * @brief Load HIGGS data and perform operations as needed
 * @param numPos: number of positive examples to be considered
 * @param numNeg: number of negative examples to be considered
 * @return X_train: training data
 * @return X_test: test data
 * @note X_train, X_test are all Eigen::MatrixXd
 * @note The original HIGGS dataset has 11 million examples.
*/
auto loadHiggs(const int numPos=56540, const int numNeg=54323){

    std::cout << "Loading HIGGS data..." << std::endl;

    const std::string higgsPath = "/data/HIGGS/HIGGS.csv";
    std::ifstream higgsFile(higgsPath);
    if (!higgsFile.is_open()) {
        std::cerr << "ERROR: Cannot open HIGGS.csv" << std::endl;
        exit(1);
    }

    auto _Result = std::make_unique<Data>();

    // Read data
    std::vector<std::vector<double>> xData;
    std::vector<int> yData;
    std::string line;
    std::vector<int> posIdx, negIdx;
    long posCount = 0, negCount = 0;
    while(std::getline(higgsFile, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');
        int label = std::stoi(cell);
        for(int i = 0; i < 28; i++){
            std::getline(ss, cell, ',');
            if (cell != "NaN" && cell != "nan" && cell != "inf" && cell != "-inf")
                row.push_back(std::stod(cell));
            else
                break;
        }
        if (row.size() != 28) {
            continue;
        }
        xData.push_back(row);
        if(label == 1){
            yData.push_back(1);
            posIdx.push_back(yData.size() - 1);
            ++posCount;
        }
        else{
            yData.push_back(0);
            negIdx.push_back(yData.size() - 1);
            ++negCount;
        }
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(posIdx.begin(), posIdx.end(), g);
    std::shuffle(negIdx.begin(), negIdx.end(), g);
    // Convert to Eigen::MatrixXd
    _Result->X_train.resize(numPos, xData[0].size());
    _Result->X_test.resize(numNeg, xData[0].size());

    for (int i = 0; i < numPos; ++i) {
        _Result->X_train.row(i) = Eigen::VectorXd::Map(xData[posIdx[i]].data(), xData[posIdx[i]].size());
    }
    for (int i = 0; i < numNeg; ++i) {
        _Result->X_test.row(i) = Eigen::VectorXd::Map(xData[negIdx[i]].data(), xData[negIdx[i]].size());
    }

    // MinMaxScale each feature
    _Result->X_train = std::move(SCALEDATA(_Result->X_train));
    _Result->X_test = std::move(SCALEDATA(_Result->X_test));

    std::cout << "HIGGS data loaded." << std::endl;

    return _Result;
}

Data* loadFacebook(){

}

/**
 * @brief Load malicious URLs data and perform operations as needed
 * @param numPos: number of positive examples to be considered
 * @param numNeg: number of negative examples to be considered
 * @return X_train: training data
 * @return X_test: test data
 * @note X_train, X_test are all Eigen::MatrixXd
*/
auto loadMaliciousUrls(const int numPos=16273, const int numNeg=2709){

    std::cout << "Loading malicious URLs data..." << std::endl;

    const std::string maliciousUrlsPath = "/data/malicious_urls/All.csv";
    std::ifstream maliciousUrlsFile(maliciousUrlsPath);
    if (!maliciousUrlsFile.is_open()) {
        std::cerr << "ERROR: Cannot open All.csv" << std::endl;
        exit(1);
    }
    
    auto _Result = std::make_unique<Data>();

    // Read data
    std::vector<std::vector<double>> xData;
    std::vector<int> yData;
    std::string line;
    std::vector<int> posIdx, negIdx;
    std::getline(maliciousUrlsFile, line); // skip the first line
    while(std::getline(maliciousUrlsFile, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        for(int i = 0; i < 79; i++){
            std::getline(ss, cell, ',');
            if(cell != "NaN")
                row.push_back(std::stod(cell));
            else
                break;
        }
        if (row.size() != 79) {
            continue;
        }
        xData.push_back(row);
        std::getline(ss, cell, ',');
        if(cell == "benign"){
            yData.push_back(0);
            negIdx.push_back(yData.size() - 1);
        }
        else{
            yData.push_back(1);
            posIdx.push_back(yData.size() - 1);
        }
    }
    // std::cout << "posIdx.size(): " << posIdx.size() << std::endl;
    // std::cout << "negIdx.size(): " << negIdx.size() << std::endl;
    // std::cout << "xData.size(): " << xData.size() << std::endl; 

    // Randomly select numPos examples
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(posIdx.begin(), posIdx.end(), g);
    std::shuffle(negIdx.begin(), negIdx.end(), g);
    _Result->X_train.resize(numPos, xData[0].size());
    _Result->X_test.resize(numNeg, xData[0].size());
    for (int i = 0; i < numPos; ++i) {
        _Result->X_train.row(i) = Eigen::VectorXd::Map(xData[posIdx[i]].data(), xData[posIdx[i]].size());
    }
    for (int i = 0; i < numNeg; ++i) {
        _Result->X_test.row(i) = Eigen::VectorXd::Map(xData[negIdx[i]].data(), xData[negIdx[i]].size());
    }

    // MinMaxScale each feature
    _Result->X_train = std::move(SCALEDATA(_Result->X_train));
    _Result->X_test = std::move(SCALEDATA(_Result->X_test));

    // std::ofstream outFile("malicious_urls.csv");
    // outFile << "X_train:" << std::endl;
    // for (int i = 0; i < _Result->X_train.rows(); ++i) {
    //     for (int j = 0; j < _Result->X_train.cols(); ++j) {
    //         outFile << _Result->X_train(i, j) << ",";
    //     }
    //     outFile << std::endl;
    // }
    // outFile << "X_test:" << std::endl;
    // for (int i = 0; i < _Result->X_test.rows(); ++i) {
    //     for (int j = 0; j < _Result->X_test.cols(); ++j) {
    //         outFile << _Result->X_test(i, j) << ",";
    //     }
    //     outFile << std::endl;
    // }
    // outFile.close();

    std::cout << "Malicious URLs data loaded." << std::endl;
    
    return _Result;
}

#endif