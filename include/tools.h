#ifndef TOOLS_H
#define TOOLS_H
#include <Eigen/Dense>
/**
 * @brief Scale each feature to [0, 1]
 * @param X: data to be scaled
 * @return scaledX: scaled data
 * @note X, scaledX are all Eigen::MatrixXd
*/
Eigen::MatrixXd scaleData(Eigen::MatrixXd X);

#endif