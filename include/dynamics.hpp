#ifndef DYNAMICS_HPP
#define DYNAMICS_HPP

#include <eigen3/Eigen/Core>

Eigen::VectorXd dynamics(double t, const Eigen::VectorXd& y, double mu);

#endif