#include <eigen3/Eigen/Core>
#include <iostream>
#include <cmath>
#include <random>
#include "dynamics.hpp"
#include "integrator.hpp"
#include "observations.hpp"
#include "estimator.hpp"
#include "constants.hpp"

int main(){
    // --- Physical constants ---
    const double mu = 398600.4418;       // km^3/s^2
    const double omega_e = 2.0 * M_PI / 86164.1; // Earth rotation rate (rad/s)
    const double GMA0 = 0.0;            // Greenwich Mean Angle at t=0

    // --- True initial state (ECI) ---
    // LEO orbit: ~7000 km altitude, circular-ish
    Eigen::VectorXd x0_true(6);
    x0_true << 7000.0, 0.0, 0.0,        // position (km)
               0.0, 7.546, 0.0;          // velocity (km/s) — roughly circular

    // --- Initial estimate (perturbed from truth) ---
    Eigen::VectorXd x0_est(6);
    x0_est << 7010.0, 5.0, -3.0,         // +10 km, +5 km, -3 km position error
              0.01, 7.54, 0.005;          // small velocity errors

    // --- Initial covariance ---
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(6, 6);
    P0.diagonal() << 10000.0, 10000.0, 100.0,   // position variance (km^2)
                      0.01, 0.01, 0.01;      // velocity variance (km/s)^2

    // --- Process noise covariance ---
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6, 6);
    Q.diagonal() << 1E-8, 1E-8, 1E-8,       // position process noise
                     1E-6, 1E-6, 1E-6;       // velocity process noise

    // --- Ground station (roughly Cape Canaveral) ---
    StationParams station;
    station.lat = 28.5 * M_PI / 180.0;       // latitude (rad)
    station.lon = -80.6 * M_PI / 180.0;      // longitude (rad)
    station.alt = 0.0;                         // altitude above surface (km)

    // Measurement noise covariance: range (km^2), azimuth (rad^2), elevation (rad^2)
    station.obsv_cov = Eigen::MatrixXd::Zero(3, 3);
    station.obsv_cov.diagonal() << 0.01,              // range: 0.1 km std dev
                                    1E-6,              // azimuth: ~0.06 deg std dev
                                    1E-6;              // elevation: ~0.06 deg std dev

    // --- Measurement noise distribution ---
    MeasurementNoise noise;
    noise.loc = Eigen::VectorXd::Zero(3);
    noise.scale = station.obsv_cov;

    // --- Simulation parameters ---
    const double t_start = 0.0;
    const double t_end = 300.0;             // ~1 orbit
    const double sample_period = 10.0;       // measurement every 10 seconds
    const int n_steps = static_cast<int>((t_end - t_start) / sample_period);

    // --- Build evaluation time vector ---
    Eigen::VectorXd teval(n_steps + 1);
    for (int i = 0; i <= n_steps; i++){
        teval(i) = t_start + i * sample_period;
    }

    // --- Step 1: Propagate truth trajectory ---
    std::cout << "Propagating true trajectory..." << std::endl;

    auto dynamics_func = [mu](double t, const Eigen::VectorXd& y) -> Eigen::VectorXd {
        return dynamics(t, y, mu);
    };

    RKF45 truth_solver(dynamics_func, t_start, t_end, x0_true);
    IntegratorResult truth = truth_solver.integrate(teval);

    if (truth_solver.err_flag){
        std::cerr << "Truth propagation failed." << std::endl;
        return 1;
    }
    std::cout << "Truth trajectory: " << truth.t.size() << " points" << std::endl;

    // --- Step 2: Generate noisy measurements ---
    std::cout << "Generating measurements..." << std::endl;
    std::mt19937 rng(42); // fixed seed for reproducibility

    Eigen::MatrixXd measurements = generate_measurements(
        truth.t, truth.sol, station, noise, rng, GMA0
    );

    // --- Step 3: Run the EKF ---
    std::cout << "Running EKF..." << std::endl;

    // Dynamics and Jacobian lambdas for the estimator
    auto f = [mu](double t, const Eigen::VectorXd& y) -> Eigen::VectorXd {
        return dynamics(t, y, mu);
    };

    auto F = [mu](double t, const Eigen::VectorXd& y) -> Eigen::MatrixXd {
        return dynamics_jacobian(t, y, mu);
    };

    // EKF state
    Eigen::VectorXd estimate = x0_est;
    Eigen::MatrixXd P = P0;

    // Storage for position error over time
    Eigen::VectorXd pos_error(n_steps);

    for (int k = 0; k < n_steps; k++){
        // Earth rotation angle at measurement time
        double theta = omega_e * teval(k + 1) + GMA0;

        // Run one EKF step: predict from t_k to t_{k+1}, then update
        EstimatorResult result = ekf(
            estimate, P, measurements.col(k + 1),
            Q, f, F, station, sample_period, theta
        );

        estimate = result.state_estimate;
        P = result.cov;

        // Compute position error (truth vs estimate)
        Eigen::Vector3d true_pos = truth.sol.col(k + 1).head(3);
        Eigen::Vector3d est_pos = estimate.head(3);
        pos_error(k) = (true_pos - est_pos).norm();

        // Print progress every 60 seconds of sim time
        if ((k + 1) % 6 == 0){
            double t_now = teval(k + 1);
            std::cout << "t=" << t_now << "s  pos_err=" << pos_error(k)
                      << " km  P_trace=" << P.diagonal().head(3).sum() << std::endl;
        }
    }

    // --- Summary ---
    std::cout << "\n=== EKF Summary ===" << std::endl;
    std::cout << "Initial position error: "
              << (x0_true.head(3) - x0_est.head(3)).norm() << " km" << std::endl;
    std::cout << "Final position error:   " << pos_error(n_steps - 1) << " km" << std::endl;
    std::cout << "Final velocity error:   "
              << (truth.sol.col(n_steps).tail(3) - estimate.tail(3)).norm() << " km/s" << std::endl;
    std::cout << "\nFinal covariance diagonal:" << std::endl;
    std::cout << "  pos: " << P.diagonal().head(3).transpose() << std::endl;
    std::cout << "  vel: " << P.diagonal().tail(3).transpose() << std::endl;

    return 0;
}