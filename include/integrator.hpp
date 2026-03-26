#ifndef INTEGRATOR_HPP
#define INTEGRATOR_HPP

#include <functional>
#include <eigen3/Eigen/Core>
#include <vector>

struct IntegratorResult{   
    Eigen::VectorXd t;
    Eigen::MatrixXd sol;
};

// Integrators return a matrix because they have a time dimension
// Matrices will be of dimension (6xN) because Eigen is column-major

struct RKF45{
    std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> func;
    double t_start;
    double t_stop;
    const Eigen::VectorXd y0;
    double min_step;
    double tol;
    bool err_flag;
    double current_step;

    // Constructor
    RKF45(
        std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> f, 
        double tstart, 
        double tstop, 
        Eigen::VectorXd y0, 
        double min_step = 1E-8, 
        double atol = 1E-6)
    : func(f), t_start(tstart), t_stop(tstop), y0(y0), min_step(min_step), tol(atol)
    {
        this->err_flag = false;
        this->current_step = 1.0;
    };

    // Function for a single step
    Eigen::VectorXd step(const double t, const Eigen::VectorXd& y, double h, double tol);

    // Function to integrate the entire time interval
    // This will call the step() function
    IntegratorResult integrate();
};

#endif