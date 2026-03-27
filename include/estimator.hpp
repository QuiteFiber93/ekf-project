#ifndef ESTIMATOR_HPP
#define ESTIMATOR_HPP

#include <eigen3/Eigen/Core>

Eigen::MatrixXd ekf(
    const Eigen::VectorXd state0,
    const Eigen::MatrixXd P0,
    const Eigen::MatrixXd observations,
    const Eigen::MatrixXd Q,
    const Eigen::MatrixXd R
){
    // Prediction Step

    // Update Step

    // Collect values
}

#endif