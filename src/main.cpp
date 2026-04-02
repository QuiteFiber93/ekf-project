#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <functional>

#include "constants.hpp"
#include "types.hpp"
#include "dynamics.hpp"
#include "observations.hpp"
#include "estimator.hpp"
#include "integrator.hpp"

int main()
{
    // ================================================================
    // Physical constants
    // ================================================================
    const double mu  = 398600.4418;   // km^3/s^2  — Earth gravitational parameter
    const double GMA = 0.0;           // Greenwich Mean Angle at epoch (rad)

    // ================================================================
    // Orbit — circular LEO at 800 km altitude
    //   Orbital radius  r = R_E + 800 = 7178 km
    //   Circular speed  v = sqrt(mu/r) ≈ 7.452 km/s
    //   Period           ≈ 6052 s  ≈ 100.9 min
    //   Inertial angular rate ≈ 0.001038 rad/s
    //   Earth rotation rate   ≈ 0.0000729 rad/s
    //   Relative ground-track rate ≈ 0.000965 rad/s
    //   Over 300 s the sub-satellite point moves ≈ 16.6° along the equator.
    //   Starting directly overhead, the minimum elevation at t=300 s is
    //   still well above the horizon, so visibility is guaranteed.
    // ================================================================

    const double alt_orbit = 800.0;                 // km
    const double r0        = R_E + alt_orbit;       // 7178 km
    const double v_circ    = std::sqrt(mu / r0);    // ~7.452 km/s

    // True initial state (ECI): satellite on +x axis, velocity along +y
    Vec6 x0_true;
    x0_true << r0, 0.0, 0.0,
               0.0, v_circ, 0.0;

    // ================================================================
    // Ground station — equator, lon = 0, sea level
    // At t = 0 with GMA = 0 the ECI and ECEF frames coincide,
    // so the station sits directly beneath the satellite.
    // ================================================================
    StationParams station;
    station.lat = 0.0;
    station.lon = 0.0;
    station.alt = 0.0;

    // Measurement noise covariance  R = diag(σ_ρ², σ_az², σ_el²)
    //   σ_ρ  = 0.01 km   (10 m range noise)
    //   σ_az = 0.001 rad  (~0.057°)
    //   σ_el = 0.001 rad  (~0.057°)
    Eigen::Matrix3d R_meas = Eigen::Matrix3d::Zero();
    R_meas(0, 0) = 1e-4;   // (0.01 km)^2
    R_meas(1, 1) = 1e-6;   // (0.001 rad)^2
    R_meas(2, 2) = 1e-6;   // (0.001 rad)^2
    station.obsv_cov = R_meas;

    station.precompute();

    // ================================================================
    // Simulation timing
    // ================================================================
    const double t_final = 300.0;     // seconds
    const double dt_meas = 10.0;      // measurement cadence
    const int    n_meas  = static_cast<int>(t_final / dt_meas);  // 30 steps

    // Measurement time vector: t = 10, 20, ..., 300
    Eigen::VectorXd t_meas(n_meas);
    for (int i = 0; i < n_meas; i++) {
        t_meas(i) = (i + 1) * dt_meas;
    }

    // ================================================================
    // Propagate truth trajectory to all measurement times
    // ================================================================
    auto truth_dynamics = [&mu](double t, const Eigen::VectorXd& y) -> Eigen::VectorXd {
        return dynamics(t, y, mu);
    };

    Eigen::VectorXd y0_dyn = x0_true;
    RKF45 truth_propagator(truth_dynamics, 0.0, t_final, y0_dyn, 1e-10, 1e-12);
    IntegratorResult truth_result = truth_propagator.integrate(t_meas);
    Eigen::MatrixXd true_states = truth_result.sol;   // 6 x n_meas

    // ================================================================
    // Verify the satellite is above the horizon at every measurement
    // ================================================================
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== Elevation Check ===" << std::endl;
    double min_el_deg = 1e9;
    for (int i = 0; i < n_meas; i++) {
        double theta = OMEGA * t_meas(i) + GMA;
        Vec3 meas_noiseless = h(true_states.col(i), station, theta);
        double el_deg = meas_noiseless(2) * 180.0 / M_PI;
        min_el_deg = std::min(min_el_deg, el_deg);

        if (i == 0 || i == n_meas - 1) {
            std::cout << "  t = " << std::setw(6) << t_meas(i)
                      << " s   el = " << std::setw(7) << el_deg << " deg"
                      << "   range = " << std::setw(10) << meas_noiseless(0) << " km"
                      << std::endl;
        }
    }
    std::cout << "  Min elevation over pass: " << min_el_deg << " deg" << std::endl;
    if (min_el_deg < 0.0) {
        std::cerr << "ERROR: satellite drops below horizon!" << std::endl;
        return 1;
    }
    std::cout << std::endl;

    // ================================================================
    // Generate noisy measurements
    // ================================================================
    MeasurementNoise noise;
    noise.loc   = Eigen::VectorXd::Zero(3);
    noise.scale = R_meas;

    std::mt19937 rng(42);  // fixed seed for reproducibility
    Eigen::MatrixXd measurements = generate_measurements(
        t_meas, true_states, station, noise, rng, GMA);

    // ================================================================
    // EKF setup
    // ================================================================

    // Dynamics wrappers with captured mu
    std::function<Vec6(double, const Vec6&)> f_ekf =
        [&mu](double t, const Vec6& x) -> Vec6 {
            return dynamics(t, x, mu);
        };

    std::function<Mat6(double, const Vec6&)> F_ekf =
        [&mu](double t, const Vec6& x) -> Mat6 {
            return dynamics_jacobian(t, x, mu);
        };

    // Process noise — continuous-time spectral density
    // Models unmodeled accelerations (drag, J2, SRP, etc.)
    Mat6 Q = Mat6::Zero();
    double q_accel = 1e-8;   // km^2/s^5 — tuning parameter
    Q.block<3,3>(3, 3) = q_accel * Mat3::Identity();

    // Initial estimate: true state + deliberate perturbation
    Vec6 x0_est;
    x0_est << x0_true(0) + 1.0,     // +1 km in x
              x0_true(1) + 0.5,     // +0.5 km in y
              x0_true(2) + 0.5,     // +0.5 km in z
              x0_true(3) + 0.001,   // +1 m/s in vx
              x0_true(4) - 0.001,   // -1 m/s in vy
              x0_true(5) + 0.0005;  // +0.5 m/s in vz

    // Initial covariance — consistent with perturbation magnitudes
    Mat6 P0 = Mat6::Zero();
    P0.block<3,3>(0, 0) = 4.0 * Mat3::Identity();      // (2 km)^2 position
    P0.block<3,3>(3, 3) = 1e-4 * Mat3::Identity();     // (0.01 km/s = 10 m/s)^2 velocity

    // ================================================================
    // Run the EKF
    // ================================================================
    Vec6 x_est = x0_est;
    Mat6 P_est = P0;

    std::cout << "=== EKF Results ===" << std::endl;
    std::cout << std::scientific << std::setprecision(6);
    std::cout << std::setw(8)  << "t(s)"
              << std::setw(16) << "pos_err(km)"
              << std::setw(16) << "pos_3sig(km)"
              << std::setw(16) << "vel_err(km/s)"
              << std::setw(16) << "vel_3sig(km/s)"
              << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    for (int k = 0; k < n_meas; k++) {
        double t_k   = t_meas(k);
        double theta = OMEGA * t_k + GMA;

        Vec3 z_k = measurements.col(k);

        try {
            EstimatorResult result = ekf(
                x_est, P_est, z_k, Q,
                f_ekf, F_ekf,
                station, dt_meas, theta);

            x_est = result.state_estimate;
            P_est = result.cov;
        } catch (const std::runtime_error& e) {
            std::cerr << "EKF failed at step " << k
                      << " (t=" << t_k << "s): " << e.what() << std::endl;
            return 1;
        }

        // Errors
        Vec6 err = x_est - true_states.col(k);
        double pos_err_norm = err.head<3>().norm();
        double vel_err_norm = err.tail<3>().norm();

        // 3-sigma bounds (RSS of per-axis 3-sigma)
        double pos_3sig_norm = 3.0 * std::sqrt(
            P_est(0,0) + P_est(1,1) + P_est(2,2));
        double vel_3sig_norm = 3.0 * std::sqrt(
            P_est(3,3) + P_est(4,4) + P_est(5,5));

        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(8) << t_k
                  << std::scientific << std::setprecision(6)
                  << std::setw(16) << pos_err_norm
                  << std::setw(16) << pos_3sig_norm
                  << std::setw(16) << vel_err_norm
                  << std::setw(16) << vel_3sig_norm
                  << std::endl;
    }

    // ================================================================
    // Final summary
    // ================================================================
    Vec6 final_err = x_est - true_states.col(n_meas - 1);
    std::cout << std::endl << "=== Final Summary ===" << std::endl;
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  Position error :  " << final_err.head<3>().norm()  << " km" << std::endl;
    std::cout << "  Velocity error :  " << final_err.tail<3>().norm()  << " km/s" << std::endl;
    std::cout << "  Pos 3-sig bound:  "
              << 3.0 * std::sqrt(P_est(0,0) + P_est(1,1) + P_est(2,2))
              << " km" << std::endl;
    std::cout << "  Vel 3-sig bound:  "
              << 3.0 * std::sqrt(P_est(3,3) + P_est(4,4) + P_est(5,5))
              << " km/s" << std::endl;

    // Per-axis consistency check
    bool consistent = true;
    for (int i = 0; i < 6; i++) {
        if (std::abs(final_err(i)) > 3.0 * std::sqrt(P_est(i, i))) {
            consistent = false;
            std::cout << "  WARNING: state[" << i << "] error "
                      << std::abs(final_err(i)) << " exceeds 3-sigma "
                      << 3.0 * std::sqrt(P_est(i, i)) << std::endl;
        }
    }
    std::cout << "  Filter consistent: " << (consistent ? "YES" : "NO") << std::endl;

    return 0;
}