#include <eigen3/Eigen/Core>
#include <cmath>
#include "integrator.hpp"

Eigen::VectorXd RKF45::step(const double tk, const Eigen::VectorXd& yk, double h, double tol) {
    // "Next step"
    Eigen::VectorXd yk1 = yk; // Initializing rk4 step
    Eigen::VectorXd zk1 = yk; // Initializing rk5 step

    double err; 
    double s;
    do {

        // Making sure we don't overflow on time
        if (tk + h > t_stop){
            h = this->t_stop - tk;
        }

        // If h has shrunk below min_step, give up
        if (h < this->min_step){
            this->err_flag = true;
            this->current_step = h;
            return yk1;
        }

        // Coefficients for RK4 and RK5
        Eigen::VectorXd k1 = h * func(tk, yk);
        Eigen::VectorXd k2 = h * func(tk + 0.25*h, yk + 0.25*k1);
        Eigen::VectorXd k3 = h * func(tk + 0.375*h, yk + 3.0/32.0*k1 + 9.0/32.0*k2);
        Eigen::VectorXd k4 = h * func(tk + 12.0/13.0*h, yk + 1932.0/2197.0*k1 - 7200.0/2197.0*k2 + 7296.0/2197.0*k3);
        Eigen::VectorXd k5 = h * func(tk + h, yk + 439.0/216.0*k1 - 8*k2 + 3680.0/513.0*k3 - 845.0/4104.0*k4);
        Eigen::VectorXd k6 = h * func(tk + 0.5*h, yk - 8.0/27.0*k1 + 2*k2 - 3544.0/2565.0*k3 + 1859.0/4104.0*k4 - 11.0/40.0*k5);

        // Calculating new value using RK4
        yk1 = yk + 25.0/216.0*k1 + 1408.0/2565.0*k3 + 2197.0/4104.0*k4 - 0.2*k5;

        // Calculating "True" value using RK5
        zk1 = yk + 16.0/135.0*k1 + 6656.0/12825.0*k3 + 28561.0/56340.0*k4 - 9.0/50.0*k5 + 2.0/55.0*k6;

        // Calculating error
        err = (zk1 - yk1).norm();

        // Returning value if we satisfy tolerance
        if (err <= tol) {
            this->current_step = h;
            return yk1;
        } 

        // Otherwise, reduce step size and try again
        s = pow(tol / (2 * err), 0.25);
        h = s * h;

    } while (h >= this->min_step);

    // If we exit the loop, min_step was violated
    this->err_flag = true;
    this->current_step = h;
    return yk1;
}

IntegratorResult RKF45::integrate(){

    // Initializing values to store solution results
    std::vector<double> t;
    std::vector<Eigen::VectorXd> sol;

    // Initial values
    Eigen::VectorXd yk = this->y0;

    // Time step — initialize current_step for the first call to step()
    double h = 1.0;
    current_step = h;

    // Beginning integration
    double tk = t_start;

    // Record initial conditions
    t.push_back(tk);
    sol.push_back(yk);

    while (tk < t_stop - 1E-12){

        yk = step(tk, yk, current_step, tol);

        if (err_flag){
            break;
        }

        // Advance time by the step that was actually accepted inside step()
        tk = tk + current_step;

        // Record the new state
        t.push_back(tk);
        sol.push_back(yk);

        // Grow h modestly for the next step to avoid getting stuck at small values
        current_step = std::min(current_step * 1.5, t_stop - tk);
    }

    // Converting results to Eigen matrices
    Eigen::MatrixXd sol_matrix(y0.size(), static_cast<int>(t.size()));
    for (int i = 0; i < static_cast<int>(t.size()); i++){
        sol_matrix.col(i) = sol[i];
    }

    Eigen::VectorXd t_vec = Eigen::Map<Eigen::VectorXd>(t.data(), static_cast<int>(t.size()));

    return IntegratorResult{t_vec, sol_matrix};
}