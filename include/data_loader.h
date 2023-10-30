#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <assert.h>
#include <random>
#include <algorithm>
#include <Eigen/Dense>


typedef struct Data {
    Eigen::MatrixXd X;
    Eigen::MatrixXd Y;
    Eigen::MatrixXd X_test;
    Eigen::MatrixXd Y_test;
} Data;

/**
 * @brief Load MNIST data and perform operations as needed
 * @param posDigits: vector of digits to be considered as positive
 * @param numPos: number of positive examples to be considered
 * @param numNeg: number of negative examples to be considered
 * @return X: training data
 * @return Y: training labels
 * @return X_test: test data
 * @return Y_test: test labels
 * @note X, Y, X_test, Y_test are all Eigen::MatrixXd
*/
Data* loadMnist(std::vector<int> posDigits, int numPos=6000, int numNeg=6000) {
    // Load MNIST data
    std::string mnistPath = "/Users/euclid/PHBF_CPP/data/mnist/";
    std::string trainImagesPath = mnistPath + "train-images.idx3-ubyte";
    std::string trainLabelsPath = mnistPath + "train-labels.idx1-ubyte";
    std::string testImagesPath = mnistPath + "t10k-images.idx3-ubyte";
    std::string testLabelsPath = mnistPath + "t10k-labels.idx1-ubyte";

    std::ifstream trainImagesFile(trainImagesPath, std::ios::binary);
    std::ifstream trainLabelsFile(trainLabelsPath, std::ios::binary);
    std::ifstream testImagesFile(testImagesPath, std::ios::binary);
    std::ifstream testLabelsFile(testLabelsPath, std::ios::binary);

    if (!trainImagesFile.is_open()) {
        std::cout << "ERROR: Cannot open train-images-idx3-ubyte" << std::endl;
        exit(1);
    }

    if (!trainLabelsFile.is_open()) {
        std::cout << "ERROR: Cannot open train-labels-idx1-ubyte" << std::endl;
        exit(1);
    }

    if (!testImagesFile.is_open()) {
        std::cout << "ERROR: Cannot open t10k-images-idx3-ubyte" << std::endl;
        exit(1);
    }

    if (!testLabelsFile.is_open()) {
        std::cout << "ERROR: Cannot open t10k-labels-idx1-ubyte" << std::endl;
        exit(1);
    }

    Data* _Result = new Data();

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
    Eigen::MatrixXd Y(numImages2, 10);
    _Result->X = Eigen::MatrixXd(numPos, numRows * numCols);
    _Result->Y = Eigen::MatrixXd(numPos, 10);
    long posCount = 0, negCount = 0;
    std::vector<int> sampleIdx;
    for (int i = 0; i < numImages1; ++i) {
        char* pixels = new char[numRows * numCols];
        trainImagesFile.read(pixels, numRows * numCols);
        for (int j = 0; j < numRows * numCols; ++j) {
            X(i, j) = (unsigned char)pixels[j];
        }
        delete[] pixels;

        char label;
        trainLabelsFile.read(&label, 1);
        int labelInt = (unsigned char)label;
        if (std::find(posDigits.begin(), posDigits.end(), labelInt) != posDigits.end()) {
            Y(i, labelInt) = 1;
            ++posCount;
            sampleIdx.push_back(i);
        }
        else {
            Y(i, labelInt) = 0;
            ++negCount;
        }
    }

    // Randomly select numPos examples
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(sampleIdx.begin(), sampleIdx.end(), g);
    for (int i = 0; i < numPos; ++i) {
        _Result->X.row(i) = X.row(sampleIdx[i]);
        _Result->Y.row(i) = Y.row(sampleIdx[i]);
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
    Eigen::MatrixXd Y_test(numImages2, 10);
    _Result->X_test = Eigen::MatrixXd(numNeg, numRows * numCols);
    _Result->Y_test = Eigen::MatrixXd(numNeg, 10);
    posCount = 0, negCount = 0;
    sampleIdx.clear();
    for (int i = 0; i < numImages1; ++i) {
        char* pixels = new char[numRows * numCols];
        testImagesFile.read(pixels, numRows * numCols);
        for (int j = 0; j < numRows * numCols; ++j) {
            X_test(i, j) = (unsigned char)pixels[j];
        }
        delete[] pixels;

        char label;
        testLabelsFile.read(&label, 1);
        int labelInt = (unsigned char)label;
        if (std::find(posDigits.begin(), posDigits.end(), labelInt) != posDigits.end()) {
            Y_test(i, labelInt) = 1;
            ++posCount;
        }
        else {
            Y_test(i, labelInt) = 0;
            ++negCount;
            sampleIdx.push_back(i);
        }
    }

    // Randomly select numNeg examples
    std::shuffle(sampleIdx.begin(), sampleIdx.end(),g);
    for (int i = 0; i < numNeg; ++i) {
        _Result->X_test.row(i) = X_test.row(sampleIdx[i]);
        _Result->Y_test.row(i) = Y_test.row(sampleIdx[i]);
    }


    // Close files
    trainImagesFile.close();
    trainLabelsFile.close();
    testImagesFile.close();
    testLabelsFile.close();

    // return X, Y, X_test, Y_test;
    return _Result;
}