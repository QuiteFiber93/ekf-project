#ifndef DYNAMICS_HPP
#define DYNAMICS_HPP

#include <eigen3/Eigen/Core>

Eigen::VectorXd dynamics(const double t, const Eigen::VectorXd& y, const double mu){

    (void)t; // dynamics do not use time

    Eigen::Vector3d r = y.head(3);
    Eigen::Vector3d v = y.tail(3);
    Eigen::VectorXd ydot(6);

    double r_norm = r.norm();
    double r_norm_cubed = r_norm * r_norm * r_norm;

    ydot.head(3) = v;
    ydot.tail(3) = -mu * r / r_norm_cubed;

    return ydot;
}

Eigen::MatrixXd dynamics_jacobian(double t, const Eigen::VectorXd& y, const double mu){

    (void)t;
    int n = static_cast<int>(y.size());
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(n, n);

    Eigen::Vector3d r = y.head(3);
    double r_norm = r.norm();

    // Jacobian has the structure
    // [0       | I]
    // [dr/dr   | 0]

    jacobian.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();
    jacobian.block(3, 0, 3, 3) = - mu / (r_norm * r_norm * r_norm) * Eigen::Matrix3d::Identity()
        + 3 * mu / (r_norm * r_norm * r_norm * r_norm * r_norm) * (r * r.transpose());

    return jacobian;
}

#endif