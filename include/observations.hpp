#ifndef OBSERVATIONS_HPP
#define OBSERVATIONS_HPP

#include <random>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Cholesky>
#include "constants.hpp"
#include "types.hpp"

struct MeasurementNoise {
    Eigen::VectorXd loc;    // mean of measurement noise
    Eigen::MatrixXd scale;  // covariance matrix of measurement noise

    Eigen::VectorXd sample(std::mt19937& rng) {
        int n = static_cast<int>(loc.size());
        std::normal_distribution<double> dist(0.0, 1.0);

        Eigen::VectorXd z(n);
        for (int i = 0; i < n; i++) {
            z(i) = dist(rng);
        }

        Eigen::LLT<Eigen::MatrixXd> llt(scale);
        return loc + llt.matrixL() * z;
    }
};

struct StationParams {
    double alt;
    double lon;
    double lat;
    Eigen::MatrixXd obsv_cov;

    // --- Precomputed fields (call precompute() after setting alt/lon/lat) ---
    Mat3 R_enu;        // ecf2enu rotation matrix
    Vec3 pos_ecef;     // station position in ECEF

    void precompute() {
        // ECF → ENU rotation (depends only on lat, lon)
        R_enu << -sin(lon),              cos(lon),             0.0,
                 -sin(lat)*cos(lon),    -sin(lat)*sin(lon),    cos(lat),
                  cos(lat)*cos(lon),     cos(lat)*sin(lon),    sin(lat);

        // Station ECEF position — includes altitude above spherical Earth
        double Re_alt = R_E + alt;
        pos_ecef << Re_alt * cos(lat) * cos(lon),
                    Re_alt * cos(lat) * sin(lon),
                    Re_alt * sin(lat);
    }
};

// ECI → ECEF rotation (simple z-axis rotation by theta)
inline Mat3 eci2ecf(const double theta)
{
    double ct = cos(theta);
    double st = sin(theta);
    Mat3 R;
    R <<  ct, st, 0.0,
         -st, ct, 0.0,
         0.0, 0.0, 1.0;
    return R;
}

// Measurement function: state (ECI) → [range, azimuth, elevation]
inline Vec3 h(
    const Eigen::VectorXd& state,
    const StationParams& station,
    const double theta)
{
    // Relative position in ECEF then rotate to ENU
    Vec3 rho_ecf = eci2ecf(theta) * state.head<3>() - station.pos_ecef;
    Vec3 pos_enu = station.R_enu * rho_ecf;

    double rho   = pos_enu.norm();
    Vec3 meas;
    meas << rho,
            atan2(pos_enu(0), pos_enu(1)),
            asin(pos_enu(2) / rho);

    return meas;
}

// Measurement Jacobian: ∂h/∂x  (3 × 6)
inline Mat36 H(
    const Eigen::VectorXd& state,
    const StationParams& station,
    const double theta)
{
    Vec3 rho_ecf = eci2ecf(theta) * state.head<3>() - station.pos_ecef;
    Vec3 pos_enu = station.R_enu * rho_ecf;

    const double rho  = pos_enu.norm();
    const double rho2 = rho * rho;
    const double dist = pos_enu.head<2>().norm();   // horizontal distance
    const double dist2 = dist * dist;

    // ∂h / ∂ρ_enu  (3×3)
    Mat3 dhdrho = Mat3::Zero();

    // ∂range / ∂ρ_enu = unit vector
    dhdrho.row(0) = pos_enu.normalized().transpose();

    // ∂az / ∂ρ_enu   (az = atan2(e, n))
    //   ∂az/∂e =  n / dist²
    //   ∂az/∂n = -e / dist²
    dhdrho(1, 0) =  pos_enu(1) / dist2;
    dhdrho(1, 1) = -pos_enu(0) / dist2;
    // dhdrho(1,2) = 0

    // ∂el / ∂ρ_enu   (el = asin(u / rho))
    dhdrho(2, 0) = -pos_enu(0) * pos_enu(2) / (rho2 * dist);
    dhdrho(2, 1) = -pos_enu(1) * pos_enu(2) / (rho2 * dist);
    dhdrho(2, 2) =  dist / rho2;

    // Full Jacobian: H = [∂h/∂ρ_enu * R_enu * R_ecf | 0₃ₓ₃]
    Mat36 jacobian;
    jacobian.block<3,3>(0, 0) = dhdrho * station.R_enu * eci2ecf(theta);
    jacobian.block<3,3>(0, 3) = Mat3::Zero();

    return jacobian;
}

// Generate noisy measurements for a trajectory
inline Eigen::MatrixXd generate_measurements(
    const Eigen::VectorXd& t,
    const Eigen::MatrixXd& states,
    const StationParams& station,
    MeasurementNoise& noise,
    std::mt19937& rng,
    const double GMA = 0.0)
{
    const int n = static_cast<int>(states.cols());
    Eigen::MatrixXd meas(3, n);

    for (int i = 0; i < n; i++) {
        double theta = OMEGA * t(i) + GMA;
        meas.col(i) = h(states.col(i), station, theta) + noise.sample(rng);
    }

    return meas;
}

#endif