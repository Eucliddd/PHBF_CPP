#include "tools.h"

Eigen::MatrixXd scaleData(Eigen::MatrixXd X) {
    Eigen::VectorXd min = X.colwise().minCoeff();
    Eigen::VectorXd max = X.colwise().maxCoeff();
    Eigen::MatrixXd scaledX = (X.rowwise() - min.transpose()).array().rowwise() / (max - min).transpose().array();
    return scaledX;
}