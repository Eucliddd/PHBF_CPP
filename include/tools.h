#ifndef TOOLS_H
#define TOOLS_H
#include <Eigen/Dense>
//#define SCALEDATA(X) (((X).rowwise() - (X).colwise().minCoeff().transpose()).array().rowwise() / \
//                        ((X).colwise().maxCoeff() - (X).colwise().minCoeff()).transpose().array())

inline Eigen::MatrixXd SCALER(const Eigen::MatrixXd& X, const Eigen::VectorXd& min, const Eigen::VectorXd& max) {
    return (((X).rowwise() - min.transpose()).array().rowwise() / (max - min).transpose().array());
}

inline Eigen::MatrixXd SCALEDATA(const Eigen::MatrixXd& X) {
    Eigen::VectorXd min = X.colwise().minCoeff();
    Eigen::VectorXd max = X.colwise().maxCoeff();
    return SCALER(X,min,max).array().isNaN().select(0.0,SCALER(X,min,max));
}

#endif