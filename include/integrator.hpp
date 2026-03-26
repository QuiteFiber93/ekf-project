#ifndef INTEGRATOR_HPP
#define INTEGRATOR_HPP

#include <functional>
#include <eigen3/Eigen/Core>
#include <vector>

struct IntegratorResult{   
    Eigen::VectorXd t;
    Eigen::MatrixXd sol;
    bool completed;
};

// Integrators return a matrix because they have a time dimension
// Matrices will be of dimension (6xN) because C++ is row-major
IntegratorResult euler(std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> func, 
    double t_start, double t_stop, const Eigen::VectorXd& y0, double delta_t);

// Function meant to be like RKF45 solver
IntegratorResult ode45(
    std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> func,
    const double t_start, 
    const double t_stop, 
    const Eigen::VectorXd& y0, 
    const double min_step = 1E-8, 
    double tol = 1E-8
);

#endif