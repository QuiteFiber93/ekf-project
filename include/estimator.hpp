#ifndef ESTIMATOR_HPP
#define ESTIMATOR_HPP

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Cholesky>
#include <functional>
#include <stdexcept>
#include "types.hpp"
#include "integrator.hpp"
#include "observations.hpp"

struct EstimatorResult {
    Vec6 state_estimate;
    Mat6 cov;
};

inline EstimatorResult ekf(
    const Vec6&  last_estimate,
    const Mat6&  P,
    const Vec3&  observations,
    const Mat6&  Q,
    std::function<Vec6(double, const Vec6&)>  f,
    std::function<Mat6(double, const Vec6&)>  F,
    const StationParams& station,
    const double sample_period,
    const double theta)
{
    // ================================================================
    // Prediction Step — propagate state and covariance simultaneously
    // ================================================================

    // Augmented dynamics: d/dt [x; vec(P)] = [f(x); vec(F*P + P*Fᵀ + Q)]
    // Using VectorXd for the augmented state since the integrator interface
    // is generic, but the inner math uses fixed-size types.
    auto augmented_dynamics = [&f, &F, &Q](double t, const Eigen::VectorXd& aug) -> Eigen::VectorXd
    {
        // Unpack
        Vec6 x = aug.head<6>();
        Mat6 Pk = Eigen::Map<const Mat6>(aug.data() + 6);

        // State dynamics
        Vec6 xdot = f(t, x);

        // Covariance dynamics
        Mat6 Ft   = F(t, x);
        Mat6 Pdot = Ft * Pk + Pk * Ft.transpose() + Q;

        // Pack
        Eigen::VectorXd aug_dot(42);
        aug_dot.head<6>() = xdot;
        Eigen::Map<Mat6>(aug_dot.data() + 6) = Pdot;
        return aug_dot;
    };

    // Build augmented initial state
    Eigen::VectorXd aug0(42);
    aug0.head<6>() = last_estimate;
    Eigen::Map<Mat6>(aug0.data() + 6) = P;

    // Integrate from 0 to sample_period
    RKF45 solver(augmented_dynamics, 0.0, sample_period, aug0);
    Eigen::VectorXd aug_pred = solver.integrate().sol.rightCols(1);

    // Unpack predicted state and covariance
    Vec6 x_pred = aug_pred.head<6>();
    Mat6 P_pred = Eigen::Map<Mat6>(aug_pred.data() + 6);

    // Enforce symmetry after propagation (floating-point drift)
    P_pred = 0.5 * (P_pred + P_pred.transpose());

    // ================================================================
    // Update Step
    // ================================================================

    Vec3   hk = h(x_pred, station, theta);
    Mat36  Hk = H(x_pred, station, theta);

    // Innovation covariance  S = H P Hᵀ + R
    Eigen::Matrix3d S = Hk * P_pred * Hk.transpose() + station.obsv_cov;

    // Cholesky decomposition for numerically stable solve
    Eigen::LLT<Eigen::Matrix3d> llt(S);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("EKF: Cholesky decomposition of S failed — "
                                 "innovation covariance lost positive-definiteness");
    }

    // Kalman gain:  K = P Hᵀ S⁻¹  ⟹  Kᵀ = S⁻¹ H P  ⟹  solve S Kᵀ = H P
    Eigen::Matrix<double, 6, 3> K = llt.solve(Hk * P_pred).transpose();

    // State update
    Vec6 x_upd = x_pred + K * (observations - hk);

    // Covariance update — Joseph form for numerical stability
    //   P⁺ = (I - K H) P⁻ (I - K H)ᵀ + K R Kᵀ
    // This is algebraically equivalent to the standard form but preserves
    // symmetry and positive semi-definiteness under roundoff.
    Mat6 IKH = Mat6::Identity() - K * Hk;
    Mat6 P_upd = IKH * P_pred * IKH.transpose()
               + K * station.obsv_cov * K.transpose();

    // Final symmetry enforcement
    P_upd = 0.5 * (P_upd + P_upd.transpose());

    return EstimatorResult{x_upd, P_upd};
}

#endif