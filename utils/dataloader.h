#ifndef DATALOADER
#define DATALOADER

#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <assert.h>
#include <algorithm>
#include "key.h"
#include <Eigen/Dense>
#include <sstream>

// Dataset directory
#define SHALLA_PATH  "../data/shalla_cost"
#define YCSB_PATH  "../data/ycsbt"

#define RANDOM_KEYSTR_PATH "../util/randomKeyStr.txt"
#define RANDOM_COST_TYPE zipf

enum {uniform, hotcost, normal, zipf};

const std::string root_path = "/data1/syx/data/";

// data loader for datasets and random data
class dataloader{
    public:
        std::vector<Slice *> pos_keys_; // positive keys
        std::vector<Slice *> neg_keys_; // negative keys

        ~dataloader();
        bool load(std::string data_name_, bool using_cost_);
        bool load(std::string data_name_, bool using_cost_, std::string epcho);
        bool load(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
        bool loadRandomKey(int positives_number, int negatives_number, bool using_cost_);

    private:
        bool loadShalla(bool using_cost_, std::string epcho);
        bool loadYCSB(bool using_cost_, std::string epcho);
        bool loadMnist(bool using_cost_, std::string epcho);
        bool loadMaliciousUrls(bool using_cost_, std::string epcho);
        bool loadHiggs(bool using_cost_, std::string epcho);
        bool loadEmber(bool using_cost_, std::string epcho);
        bool loadKitsune(bool using_cost_, std::string epcho);
};

// generate random keys and costs
class KeyBuilder{
    public:
        KeyBuilder();
        std::string GetKeyStr();
        bool ReadKeys(std::vector<Slice *> &v, int start_position_);
        void GenKeyStrAndToFile();
        void GenKeysUniformCosts(std::vector<Slice *> &keys, int interval);
        void GenKeysHotCosts(std::vector<Slice *> &keys, double hotNumberpro, int hotcost, int coldcost);
        void GenKeysNormalCosts(std::vector<Slice *> &keys, int u, int d);
        void GenKeysZipfCosts(std::vector<Slice *> &keys, double a, double c);
    private:
        std::vector<std::string> key_strs;
};

bool dataloader::load(std::string data_name_, bool using_cost_){
    if(data_name_ == "shalla") return loadShalla(using_cost_, "");
    else if(data_name_ == "ycsb") return loadYCSB(using_cost_, "");
    else if(data_name_ == "mnist") return loadMnist(using_cost_, "");
    else if(data_name_ == "malicious_urls") return loadMaliciousUrls(using_cost_, "");
    else if(data_name_ == "higgs") return loadHiggs(using_cost_, "");
    else if(data_name_ == "ember") return loadEmber(using_cost_, "");
    else if(data_name_ == "kitsune") return loadKitsune(using_cost_, "Mirai");
    else return false;
}

bool dataloader::load(std::string data_name_, bool using_cost_, std::string epcho){
    if(data_name_ == "shalla") return loadShalla(using_cost_, epcho);
    else if(data_name_ == "ycsb") return loadYCSB(using_cost_, epcho);
    else if (data_name_ == "mnist") return loadMnist(using_cost_, epcho);
    else if(data_name_ == "malicious_urls") return loadMaliciousUrls(using_cost_, epcho);
    else if(data_name_ == "higgs") return loadHiggs(using_cost_, epcho);
    else if(data_name_ == "ember") return loadEmber(using_cost_, epcho);
    else if(data_name_ == "kitsune") return loadKitsune(using_cost_, epcho);
    else return false;
}
bool dataloader::loadShalla(bool using_cost_, std::string epcho){
    std::cout << "shalla reading..."  << std::endl;
    std::ifstream is(SHALLA_PATH+epcho+".txt", std::ios::binary);
    if(is){
        std::string optype, keystr;
        double cost;
        while (is >> optype >> keystr >> cost) 
        { 
            Slice * key = new Slice();    
            key->str = keystr;   
            if(optype == "1") pos_keys_.push_back(key);
            else if(optype == "0"){    
                key->cost = using_cost_ ? cost : 1;
                neg_keys_.push_back(key); 
            }
        }
        is.close();
        return true;
    }
    is.close();
    return false;
}

bool dataloader::loadYCSB(bool using_cost_, std::string epcho){
    std::cout << "ycsb reading..."  << std::endl;
    std::ifstream is(YCSB_PATH+epcho+".txt");
    if(is){
        std::string optype, keystr;
        double cost;
        while (is >> optype >> keystr >> cost) 
        { 
            Slice * key = new Slice();    
            key->str = keystr;   
            if(optype == "FILTERKEY" || optype == "1") pos_keys_.push_back(key);
            else if(optype == "OTHERKEY" || optype == "0"){          
                key->cost = using_cost_ ? cost : 1;
                neg_keys_.push_back(key); 
            }
        }
        is.close();
        return true;
    }
    is.close();
    return false;
}

bool dataloader::loadRandomKey(int positives_number, int negatives_number, bool using_cost_){
    KeyBuilder kb;
    for(int i=0; i<positives_number; i++){
        Slice *key = new Slice();
        pos_keys_.push_back(key);
    }
    for(int i=0; i<negatives_number; i++){
        Slice *key = new Slice();
        neg_keys_.push_back(key);
    }
    if(!kb.ReadKeys(pos_keys_, 0)) return false; 
    if(!kb.ReadKeys(pos_keys_, 0)) return false; 
    if(using_cost_)
        switch (RANDOM_COST_TYPE)
        {
        case uniform:
            kb.GenKeysUniformCosts(neg_keys_, 5);
            break;
        case hotcost:
            kb.GenKeysHotCosts(neg_keys_, 0.01, 100, 1);
            break;
        case normal:
            kb.GenKeysNormalCosts(neg_keys_, 20, 10);
            break;
        case zipf:
            kb.GenKeysZipfCosts(neg_keys_, 1.25, 1.0);
            break;
        default:
            break;
        }
    return true;
}

bool dataloader::load(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y){
    for(int i=0; i<X.rows(); i++){
        Slice *key = new Slice();
        std::stringstream ss;
        ss << X.row(i);
        key->str = ss.str();
        pos_keys_.push_back(key);
    }
    for(int i=0; i<Y.rows(); i++){
        Slice *key = new Slice();
        std::stringstream ss;
        ss << Y.row(i);
        key->str = ss.str();
        key->cost = 1;
        neg_keys_.push_back(key);
    }
    return true;
}

bool dataloader::loadMaliciousUrls(bool using_cost_, std::string epcho){

    std::cout << "Malicious URLs reading..."  << std::endl;

    constexpr auto pos_num = 16273;
    constexpr auto neg_num = 2709;

    const std::string maliciousUrlsPath = rootpath + "malicious_urls/All.csv";
    std::ifstream maliciousUrlsFile(maliciousUrlsPath);
    if (!maliciousUrlsFile.is_open()) {
        std::cerr << "ERROR: Cannot open All.csv" << std::endl;
        return false;
    }

    std::string line;
    std::vector<Slice> _poskeys;
    std::vector<Slice> _negkeys;
    std::getline(maliciousUrlsFile, line); // skip the first line
    while (std::getline(maliciousUrlsFile, line)) {
        // if line contains NaN,nan,inf,-inf, skip it
        if (line.find("NaN") != std::string::npos || line.find("nan") != std::string::npos ||
            line.find("inf") != std::string::npos || line.find("-inf") != std::string::npos) {
            continue;
        }
        std::string tag = line.substr(line.find_last_of(',') + 1);
        if (tag == "benign") {
            _negkeys.push_back(Slice{line.substr(0, line.find_last_of(',')), 1});
        } else {
            _poskeys.push_back(Slice{line.substr(0, line.find_last_of(',')), 1});
        } 
    }
    maliciousUrlsFile.close();

    // shuffle
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(_poskeys.begin(), _poskeys.end(), g);
    std::shuffle(_negkeys.begin(), _negkeys.end(), g);

    for (int i = 0; i < pos_num; i++) {
        pos_keys_.push_back(new Slice(std::move(_poskeys[i])));
    }
    for (int i = 0; i < neg_num; i++) {
        neg_keys_.push_back(new Slice(std::move(_negkeys[i])));
    }

    std::cout << "Malicious URLs data loaded." << std::endl;
    return true;
}

bool dataloader::loadHiggs(bool using_cost_, std::string epcho){

    std::cout << "HIGGS reading..."  << std::endl;

    const std::string higgsPath = rootpath + "HIGGS/HIGGS.csv";
    std::ifstream higgsFile(higgsPath);
    if (!higgsFile.is_open()) {
        std::cerr << "ERROR: Cannot open HIGGS.csv" << std::endl;
        return false;
    }

    std::string line;
    std::getline(higgsFile, line); // skip the first line
    std::vector<Slice> _poskeys;
    std::vector<Slice> _negkeys;
    while (std::getline(higgsFile, line)) {
        // if line contains NaN,nan,inf,-inf, skip it
        if (line.find("NaN") != std::string::npos || line.find("nan") != std::string::npos ||
            line.find("inf") != std::string::npos || line.find("-inf") != std::string::npos) {
            continue;
        }
        std::string tag = line.substr(0, line.find_first_of(','));
        if (stoi(tag) == 1) {
            _poskeys.push_back(Slice{line.substr(line.find_first_of(',') + 1), 1});
        } else {
            _negkeys.push_back(Slice{line.substr(line.find_first_of(',') + 1), 1});
        }
    }
    higgsFile.close();

    // shuffle
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(_poskeys.begin(), _poskeys.end(), g);
    std::shuffle(_negkeys.begin(), _negkeys.end(), g);

    const int pos_size = 25000;
    const int neg_size = 25000;
    for (int i = 0; i < pos_size; i++) {
        pos_keys_.push_back(new Slice(std::move(_poskeys[i])));
    }
    for (int i = 0; i < neg_size; i++) {
        neg_keys_.push_back(new Slice(std::move(_negkeys[i])));
    }

    std::cout << "HIGGS data loaded." << std::endl;
    return true;
}

bool dataloader::loadEmber(bool using_cost_, std::string epcho){

    std::cout << "EMBER reading..."  << std::endl;

    std::ifstream xFile(rootpath + "EMBER/ember_pos.csv");
    std::ifstream yFile(rootpath + "EMBER/ember_neg.csv");

    if (!xFile || !yFile) {
        std::cerr << "Failed to open data files." << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(xFile, line)) {
        // if line contains NaN,nan,inf,-inf, skip it
        if (line.find("NaN") != std::string::npos || line.find("nan") != std::string::npos ||
            line.find("inf") != std::string::npos || line.find("-inf") != std::string::npos) {
            continue;
        }
        Slice * key = new Slice();
        key->str = line;
        pos_keys_.push_back(key);
    }
    xFile.close();

    while (std::getline(yFile, line)) {
        // if line contains NaN,nan,inf,-inf, skip it
        if (line.find("NaN") != std::string::npos || line.find("nan") != std::string::npos ||
            line.find("inf") != std::string::npos || line.find("-inf") != std::string::npos) {
            continue;
        }
        Slice * key = new Slice();
        key->str = line;
        key->cost = using_cost_ ? 1 : 1;
        neg_keys_.push_back(key);
    }
    yFile.close();

    std::cout << "EMBER data loaded." << std::endl;
    return true;
}

bool dataloader::loadKitsune(bool using_cost_, std::string epcho="Mirai"){

    std::cout << "KITSUNE reading..."  << std::endl;

    std::ifstream xFile(rootpath + "Kitsune/" + epcho + "_dataset.csv");
    std::ifstream yFile(rootpath + "Kitsune/" + epcho + "_labels.csv");
    if (!xFile || !yFile) {
        std::cerr << "Failed to open data files." << std::endl;
        exit(1);
    }

    std::string line;
    std::vector<Slice> keys;
    std::vector<int> labels;
    std::vector<size_t> posIdx;
    std::vector<size_t> negIdx;
    long posCount = 0, negCount = 0;
    while (std::getline(xFile, line)) {
        // if line contains NaN,nan,inf,-inf, skip it
        if (line.find("NaN") != std::string::npos || line.find("nan") != std::string::npos ||
            line.find("inf") != std::string::npos || line.find("-inf") != std::string::npos) {
            continue;
        }
        keys.push_back(Slice{line, 1});
    }
    xFile.close();
    
    while (std::getline(yFile, line)) {
        // if line contains NaN,nan,inf,-inf, skip it
        if (line.find("NaN") != std::string::npos || line.find("nan") != std::string::npos ||
            line.find("inf") != std::string::npos || line.find("-inf") != std::string::npos) {
            continue;
        }
        int label = std::stoi(line);
        labels.push_back(label);
        if (label == 1) {
            posCount++;
            posIdx.push_back(labels.size() - 1);
        } else {
            negCount++;
            negIdx.push_back(labels.size() - 1);
        }
    }
    yFile.close();

    for(size_t i = 0; i < posCount; i++){
        pos_keys_.push_back(new Slice(std::move(keys[posIdx[i]])));
    }
    for(size_t i = 0; i < negCount; i++){
        neg_keys_.push_back(new Slice(std::move(keys[negIdx[i]])));
    }

    std::cout << "KITSUNE data loaded." << std::endl;
    return true;
}

bool dataloader::loadMnist(bool using_cost_, std::string epcho){
    std::cout << "mnist reading..."  << std::endl;
    const std::string mnistPath = rootpath + "mnist/";
    std::string trainImagesPath = mnistPath + "train-images.idx3-ubyte";
    std::string trainLabelsPath = mnistPath + "train-labels.idx1-ubyte";
    std::string testImagesPath = mnistPath + "t10k-images.idx3-ubyte";
    std::string testLabelsPath = mnistPath + "t10k-labels.idx1-ubyte";

    std::ifstream trainImagesFile(trainImagesPath, std::ios::binary);
    std::ifstream trainLabelsFile(trainLabelsPath, std::ios::binary);
    std::ifstream testImagesFile(testImagesPath, std::ios::binary);
    std::ifstream testLabelsFile(testLabelsPath, std::ios::binary);

    if (!trainImagesFile || !trainLabelsFile || !testImagesFile || !testLabelsFile) {
        std::cerr << "Failed to open data files." << std::endl;
        exit(1);
    }

    const std::array<int,1> posDigits = {0};
    std::vector<size_t> sampleIdx;

    // read train images
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

    std::vector<Slice> _poskeys;
    std::vector<Slice> _negkeys;
    for (size_t i = 0; i < numImages1; i++) {
        char* pixels = new char[numRows * numCols];
        trainImagesFile.read(pixels, numRows * numCols);
        _poskeys.push_back(Slice{std::string(pixels, numRows * numCols), 1});
        delete[] pixels;

        char label;
        trainLabelsFile.read(&label, 1);
        int digit = (int)label;
        if (std::find(posDigits.begin(), posDigits.end(), digit) != posDigits.end()) {
            sampleIdx.push_back(i);
        }
    }
    trainImagesFile.close();
    trainLabelsFile.close();

    // Randomly select 6000 examples
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(sampleIdx.begin(), sampleIdx.end(), g);
    for (size_t i = 0; i < 6000; i++) {
        pos_keys_.push_back(new Slice(std::move(_poskeys[sampleIdx[i]])));
    }

    // read test images
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

    sampleIdx.clear();
    for (size_t i = 0; i < numImages1; i++) {
        char* pixels = new char[numRows * numCols];
        testImagesFile.read(pixels, numRows * numCols);
        _negkeys.push_back(Slice{std::string(pixels, numRows * numCols), 1});
        delete[] pixels;

        char label;
        testLabelsFile.read(&label, 1);
        int digit = (int)label;
        if (std::find(posDigits.begin(), posDigits.end(), digit) == posDigits.end()) {
            sampleIdx.push_back(i);
        }
    }
    testImagesFile.close();
    testLabelsFile.close();

    // Randomly select 6000 examples
    std::shuffle(sampleIdx.begin(), sampleIdx.end(), g);
    for (size_t i = 0; i < 6000; i++) {
        neg_keys_.push_back(new Slice(std::move(_negkeys[sampleIdx[i]])));
    }

    std::cout << "mnist data loaded." << std::endl;

    return true;
}

dataloader::~dataloader(){
    for(Slice *key : pos_keys_)
        delete key;
    for(Slice *key : neg_keys_)
        delete key;
}

KeyBuilder::KeyBuilder(){
    std::fstream ifs(RANDOM_KEYSTR_PATH);
    if(ifs.is_open()){
        std::string s;
        while(std::getline(ifs,s))
            key_strs.push_back(s);
    }else{
        std::cout << "Keystr file not exists, generate again..." << std::endl;
        GenKeyStrAndToFile();
    }
    ifs.close();
}
std::string KeyBuilder::GetKeyStr(){
    int k=rand()%10+1;
    char arr[10];
    for(int i=1;i<=k;i++){
        int x,s;                         
        s=rand()%2;                     
        if(s==1) x=rand()%('Z'-'A'+1)+'A';        
        else x=rand()%('z'-'a'+1)+'a';      
        arr[i-1] = x;                  
    }
    return std::string(arr,k);
}

bool KeyBuilder::ReadKeys(std::vector<Slice *> &v, int start_position_){
    int size_ = v.size();
    if(start_position_ + size_ >= key_strs.size()) return false;
    for(int j=0; j<size_; j++)
        v[j]->str = key_strs[start_position_+j];
    return true;
}

void KeyBuilder::GenKeyStrAndToFile(){
    int gen_key_size_ = 200000;
    int i = key_strs.size();
    std::ofstream ofs(RANDOM_KEYSTR_PATH);
    while(i < gen_key_size_){
        if(0 == i%10000) std::cout << i << "keys have been created..." << std::endl;
        std::string str = GetKeyStr();
        if(std::find(key_strs.begin(), key_strs.end(), str) == key_strs.end()){
            key_strs.push_back(str);
            ofs << str << std::endl;
            i++;
        }
    }
    ofs.close();
}

void KeyBuilder::GenKeysUniformCosts(std::vector<Slice *> &keys, int interval){
    for(int i=0; i<keys.size(); i++)
        keys[i]->cost = 1+(i+1)*interval;
}
void KeyBuilder::GenKeysHotCosts(std::vector<Slice *> &keys, double hotNumberpro, int hotcost, int coldcost){
    int hotNumSize = hotNumberpro * keys.size();
    for(int i=0; i<keys.size(); i++){
        if(i <= hotNumSize) 
            keys[i]->cost = hotcost;
        else 
            keys[i]->cost = coldcost;
    }
}
void KeyBuilder::GenKeysNormalCosts(std::vector<Slice *> &keys, int u, int d){
    for(int i=0; i<keys.size(); i++){
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::normal_distribution<double> distribution(u, d);
        keys[i]->cost = distribution(generator);
    }
}

void KeyBuilder::GenKeysZipfCosts(std::vector<Slice *> &keys, double a, double c){
    int r = 10000;
    double pf[10000];
    double sum = 0.0;
    for (int i = 0; i < r; i++)        
        sum += c/pow((double)(i+2), a);  
    for (int i = 0; i < r; i++){ 
        if (i == 0)
            pf[i] = c/pow((double)(i+2), a)/sum;
        else
            pf[i] = pf[i-1] + c/pow((double)(i+2), a)/sum;
    }
     for (int i = 0; i < keys.size(); i++){
        int index = 0;
        double data = (double)rand()/RAND_MAX;  
        while (data > pf[index])  
            index++;
        keys[i]->cost = index;
    }
}

#endif