#include <eigen3/Eigen/Core>
#include <iostream>
#include "dynamics.hpp"
#include "integrator.hpp"

int main(){
    Eigen::VectorXd y0(6);
    y0 << 7E3, 0.0, 0.0, 0.0, 11.0, 0.0;
    double t_start = 0.0;
    double t_stop = 1000.0;
    double min_step = 1E-4;
    double tol = 1E-4;
    
    double mu = 3986500.4418;
    auto func = [mu](double t, const Eigen::VectorXd& y) -> Eigen::VectorXd {
        return dynamics(t, y, mu);
    };
    
    RKF45 solver = RKF45(func, t_start, t_stop, y0, min_step, tol);

    IntegratorResult result = solver.integrate();

    if (solver.err_flag){
        std::cout << "Integration failed — min step violated" << std::endl;
    } else {
        std::cout << "Integration complete: " << result.t.size() << " steps" << std::endl;
        std::cout << "Final time: " << result.t.tail(1) << std::endl;
        std::cout << "Final state:\n" << result.sol.rightCols(1) << std::endl;
    }
 
    return 0;
}