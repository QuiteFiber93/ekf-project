#ifndef INTEGRATOR_HPP
#define INTEGRATOR_HPP

#include <functional>
#include <eigen3/Eigen/Core>
#include <vector>

struct IntegratorResult {
    Eigen::VectorXd t;
    Eigen::MatrixXd sol;
};

// Integrators return a matrix because they have a time dimension
// Matrices will be of dimension (n_state x N) because Eigen is column-major

struct RKF45 {
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
    : func(std::move(f)), t_start(tstart), t_stop(tstop), y0(std::move(y0)),
      min_step(min_step), tol(atol), err_flag(false), current_step(1.0)
    {}

    // Single adaptive step.
    // f0 = func(tk, yk) is passed in to avoid recomputation on rejected steps.
    Eigen::VectorXd step(double tk, const Eigen::VectorXd& yk,
                         const Eigen::VectorXd& f0, double h, double tol);

    // Integrate the entire interval (adaptive output)
    IntegratorResult integrate();

    // Integrate and evaluate at specified times (with Hermite interpolation)
    IntegratorResult integrate(Eigen::VectorXd& teval);
};

#endif