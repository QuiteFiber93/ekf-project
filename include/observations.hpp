#ifndef OBSERVATIONS_HPP
#define OBSERVATIONS_HPP

#include <random>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Cholesky>
#include "constants.hpp"

struct MeasurementNoise {
    Eigen::VectorXd loc;    // mean of measurement noise
    Eigen::MatrixXd scale;  // covariance matrix of measurement noise
    
    Eigen::VectorXd sample(std::mt19937& rng){
        int n = static_cast<int>(loc.size());
        std::normal_distribution<double> dist(0.0, 1.0);

        // Draw n independent standard normal samples
        Eigen::VectorXd z(n);
        for (int i = 0; i < n; i++){
            z(i) = dist(rng);
        }

        // Transform: loc + L * z, where L is the Cholesky decomposition of scale
        // This gives us samples from N(loc, scale)
        Eigen::LLT<Eigen::MatrixXd> llt(scale);
        return loc + llt.matrixL() * z;
    }
};

struct StationParams {
  double alt;
  double lon;
  double lat;
  Eigen::MatrixXd obsv_cov;
};

// This returns a rotation matrix betwen the ECI and ECF frames
// Assumes simple rotation about the z axis
Eigen::Matrix3d eci2ecf(const double theta){
    Eigen::Matrix3d R;
    R << cos(theta), sin(theta), 0,
        -sin(theta), cos(theta), 0,
        0, 0, 1;

    return R ;
}

// This returns a rotation matrix between the ECF and ENU station frames
// Assumes a spherical Earth
Eigen::Matrix3d ecf2enu(const double lat, const double lon){
    Eigen::Matrix3d R;
    R << -sin(lon), cos(lon), 0,
        -sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat),
        cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat);

    return R;
}

// Function to generate measurements for a single state
Eigen::VectorXd h(
    const Eigen::VectorXd& state, 
    const StationParams& station,
    const double theta
){
    // Compute station position from lat/lon
    // Station position is in ECEF
    Eigen::Vector3d station_pos;
    station_pos << R_E * cos(station.lat) * cos(station.lon),
                   R_E * cos(station.lat) * sin(station.lon),
                   R_E * sin(station.lat);

    // Getting relative position in ENU frame
    Eigen::Vector3d rho_vec = eci2ecf(theta) * state.head(3) - station_pos;
    Eigen::Vector3d pos_enu = ecf2enu(station.lat, station.lon) * rho_vec;

    // Getting range and angle measurements in order rho, az, alt
    Eigen::VectorXd meas(3);
    meas << pos_enu.norm(), 
            atan2(pos_enu(0), pos_enu(1)), 
            asin(pos_enu(2) / pos_enu.norm());

    return meas;
}

// Function to generate measurements for a set of states
Eigen::MatrixXd generate_measurements(
    const Eigen::VectorXd& t, 
    const Eigen::MatrixXd& states,
    const StationParams& station,
    MeasurementNoise& noise,
    std::mt19937& rng,
    const double GMA = 0.0
){
    // Allocating memory for the measurements
    // 3 measurements per measurement step
    Eigen::MatrixXd meas(3, static_cast<int>(states.cols()));

    // Initializing variables to be used and change in for loop
    const double omega = 2 * M_PI / 86164.1;
    double theta; // rotation of Earth

    // Looping through t and calculating altitude and azimuth for each step
    for (int i = 0; i < static_cast<int>(meas.cols()); i++){
        theta = omega * t(i) + GMA;
        // Rotating position vector to station ENU frame
        // Adding measurement noise
        meas.col(i) = h(states.col(i), station, theta) + noise.sample(rng);
    }

    return meas;
}

Eigen::MatrixXd H(
    const Eigen::VectorXd& state, 
    const StationParams& station,
    const double theta
){
    // Compute station position from lat/lon
    Eigen::Vector3d station_pos;
    station_pos << R_E * cos(station.lat) * cos(station.lon),
                   R_E * cos(station.lat) * sin(station.lon),
                   R_E * sin(station.lat);

    // Getting relative position in ENU frame
    // Needed for computations
    Eigen::Vector3d rho_vec = eci2ecf(theta) * state.head(3) - station_pos;
    Eigen::Vector3d pos_enu = ecf2enu(station.lat, station.lon) * rho_vec;

    // Values that will be showing up repeatedly
    const double dist = pos_enu.head(2).norm();
    const double rho = rho_vec.norm();

    // H will have dimensions (n_obs x n_states)
    Eigen::MatrixXd jacobian(3, 6);

    // The measurement jacobian is a block matrix
    // H = [dy/dx | 0_(3x3)]
    // Calculating dh/dr
    // dh/dr = dh/drho * drho/dr
    Eigen::Matrix3d dhdrho = Eigen::Matrix3d::Zero();

    // d|rho|/drho_enu = rho_hat transposed
    dhdrho.row(0) = pos_enu.normalized();

    // daz/drho = rho_n / dist^2, -rho_e/dist^2, 0
    dhdrho(1, 0) = pos_enu(1) / (dist * dist);
    dhdrho(1, 1) = -pos_enu(0) / (dist * dist);

    // dalt/drho = -rho_e * rho_u / (rho^2 *dist)), -rho_n * rho_u / (rho^2 *dist)),  dist/rho^2
    dhdrho(2, 0) = -pos_enu(0) * pos_enu(2) / (rho * rho * dist);
    dhdrho(2, 1) = -pos_enu(1) * pos_enu(2) / (rho * rho * dist);
    dhdrho(2, 2) = dist / (rho * rho);

    // The first block is dh/drho * R_ENU * R_ECF
    jacobian.block(0, 0, 3, 3) = dhdrho * ecf2enu(station.lat, station.lon) * eci2ecf(theta);

    // The second block is just zeros
    jacobian.block(0, 3, 3, 3) = Eigen::Matrix3d::Zero();

    return jacobian;
}

#endif