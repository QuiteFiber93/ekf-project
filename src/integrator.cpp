#include <eigen3/Eigen/Core>
#include <cmath>
#include "integrator.hpp"

// This is a constant time integrator
// But it will be returning time and trajectory
IntegratorResult euler(std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> func, 
    const double t_start, const double t_stop, const Eigen::VectorXd& y0, const double delta_t){

    // Number of steps to be taken
    int n_steps = static_cast<int>(std::floor((t_stop - t_start) / delta_t));
    
    // Returned matrix should have at most n+2 columns
    // t = t_start -> 1st column
    // t = t_start + n*delta_t -> column n+1 (index n)
    // if t_start + n*delta_t < t_stop, then column n+2
    Eigen::VectorXd t(n_steps + 2);
    Eigen::MatrixXd sol(6, n_steps+2);
    
    // First n+1 steps
    for (int i = 0; i <= n_steps; i++){
        
        // updating time series
        t(i) = t_start + i*delta_t;

        // updating state with euler step
        if (i > 0){
            // 
            sol.col(i) = sol.col(i - 1) + delta_t * func(t(i - 1), sol.col(i-1));
        }

        else {
            sol.col(i) = y0;
        }
    }

    // Check afterwards to make sure we are not equal to t_stop
    // If we are, resize vectors to have n+1 columns instead and return
    if (t(n_steps) >= t_stop - 1E-12){
        t.conservativeResize(n_steps + 1);
        sol.conservativeResize(6, n_steps + 1);
        return IntegratorResult{t, sol, true};
    }

    else {
        int k = n_steps + 1;
        t(k) = t_stop;
        
        double tn1 = t(k - 1);
        double dt = (t_stop - tn1);
        sol.col(k) = sol.col(k - 1) + dt * func(tn1, sol.col(k - 1));

        return IntegratorResult{t, sol, true};
    }
}

IntegratorResult ode45(
    std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> func,
    const double t_start, 
    const double t_stop, 
    const Eigen::VectorXd& y0, 
    const double min_step = 1E-8, 
    double tol = 1E-8
){  

    // For now, these are std::vectors because they are dynamic in memory
    // And I usually only need to access the most recent time step
    std::vector<Eigen::VectorXd> sol;
    std::vector<double> t;

    // Initial Conditions for integration
    Eigen::VectorXd yk = y0;

    // Initializing next step
    Eigen::VectorXd yk1;
    
    // Candidate step size
    double h;
    
    // Error of estimate
    // Will always be greater than tol upon init
    double err = 2*tol;

    // Counter for time step k
    int k;

    // This for loop cares about tk, the current time, and k, the current step
    // Until tk is greater than t_stop, this will continue to iterate
    // Upon each iterate, tk will increment by the current value of h
    // And k will increment by 1
    for (double tk = t_start, k = 0; tk <= t_stop; tk = tk + h, k++){

    // reset step size step size
    h = 0.1;

    // for (h = 0.1; err >= tol, h >= min_step; h = h * some scale factor)
    Eigen::VectorXd k1 = h * func(tk, yk);
    Eigen::VectorXd k2 = h * func(tk + 0.25*h, yk + 0.25*k1);
    Eigen::VectorXd k3 = h * func(tk + 0.325*h, yk + 3.0/32.0*k1 + 9.0/32.0*k2);
    Eigen::VectorXd k4 = h * func(tk + 12.0/13.0*h, yk + 1932.0/2197.0*k1 - 7200.0/2197.0*k2+ 7296.0/2197.0*k3);
    Eigen::VectorXd k5 = h * func(tk + h, yk + 439.0/216.0*k1 - 8*k2 + 3680.0/513.0*k3 - 845.0/4104.0*k4);
    Eigen::VectorXd k6 = h * func(tk + 0.5*h, yk - 8.0/27.0*k1 + 2*k2 - 3544.0/2565.0*k3 + 1859.0/4104.0*k4 - 11.0/40.0*k5);

    // Calculating new value using RK4

    // Calculating "True" value using RK5

    // Checking if tolerance is satisfied

    // If tolerance is satisfied, append values 

    // If tolerance is not satisfied, restart process with smaller h

    // If h becomes less than min_step and the tolerance is still not satisfied
    // Break the loop and return complete flag as false
    }
}