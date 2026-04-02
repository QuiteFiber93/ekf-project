#ifndef ESTIMATOR_HPP
#define ESTIMATOR_HPP

#include <eigen3/Eigen/Core>
#include <functional>
#include "integrator.hpp"
#include "observations.hpp"

struct EstimatorResult{
    Eigen::VectorXd state_estimate;
    Eigen::MatrixXd cov;
};

EstimatorResult ekf(
    const Eigen::VectorXd& last_estimate,
    const Eigen::MatrixXd& P,
    const Eigen::VectorXd& observations,
    const Eigen::MatrixXd& Q,
    std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> f,
    std::function<Eigen::MatrixXd(double, const Eigen::VectorXd&)> F,
    const StationParams& station,
    const double sample_period,
    const double theta
){
    // Prediction Step
    // First need to propagate from the last_estimate to the current time value
    // Need to propagate both the state estimate and the estimate covariance
    // Building an lambda function containing the dynamics of estimate and covariance
    auto augmented_dynamics = [&f, &F, &Q](double t, const Eigen::VectorXd& augmented_state) -> Eigen::VectorXd{
        
        // Extracting state variables
        Eigen::VectorXd x = augmented_state.head(6);
        Eigen::MatrixXd P = Eigen::Map<const Eigen::MatrixXd>(augmented_state.data() + 6, 6, 6);
        
        // Actual dynamics
        Eigen::VectorXd xdot =  f(t, x);
        Eigen::MatrixXd Ft = F(t, x);
        Eigen::MatrixXd Pdot = Ft * P + P * Ft.transpose() + Q;

        // Concatenating results
        Eigen::VectorXd augmented_statedot(42);
        augmented_statedot.head(6) = xdot;
        Eigen::Map<Eigen::MatrixXd>(augmented_statedot.data() + 6, 6, 6) = Pdot;
        return augmented_statedot;
    };
    
    // Building the augmented state
    Eigen::VectorXd augmented_state_last(42);
    augmented_state_last.head(6) = last_estimate;
    Eigen::Map<Eigen::MatrixXd>(augmented_state_last.data() + 6, 6, 6) = P;

    // Integrating
    RKF45 solver(augmented_dynamics, 0.0, sample_period, augmented_state_last);
    Eigen::VectorXd augmented_prediction = solver.integrate().sol.rightCols(1);

    // Unpacking
    Eigen::VectorXd state_prediction = augmented_prediction.head(6);
    Eigen::MatrixXd P_prediction = Eigen::Map<Eigen::MatrixXd>(augmented_prediction.data() + 6, 6, 6);

    // Update Step
    Eigen::VectorXd hk = h(state_prediction, station, theta);
    Eigen::MatrixXd Hk = H(state_prediction, station, theta);

    // Calculating Kalman gain using Cholesky decomposition for numerical stability
    Eigen::MatrixXd S = Hk * P_prediction * Hk.transpose() + station.obsv_cov;
    Eigen::MatrixXd K = S.llt().solve(Hk * P_prediction).transpose();

    Eigen::VectorXd estimate_update = state_prediction + K * (observations - hk);
    Eigen::MatrixXd P_update = (Eigen::MatrixXd::Identity(6,6) - K*Hk)*P_prediction;

    // Return estimate and covariance

    return EstimatorResult{estimate_update, P_update};
}

#endif