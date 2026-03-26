#ifndef DYNAMICS_HPP
#define DYNAMICS_HPP

#include <eigen3/Eigen/Core>

Eigen::VectorXd dynamics(double t, const Eigen::VectorXd& y, double mu){

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

#endif