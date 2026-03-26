#include <eigen3/Eigen/Core>
#include "constants.hpp"

Eigen::Matrix3d eci2ecf(const double theta){
    Eigen::Matrix3d R;
    R << cos(theta), -sin(theta), 0,
        sin(theta), cos(theta), 0,
        0, 0, 1;

    return R ;
}

Eigen::Matrix3d ecf2enu(const double lat, const double lon){
    Eigen::Matrix3d R;
    R << -sin(lon), cos(lon), 0,
        -sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat),
        cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat);

    return R;
}

Eigen::MatrixXd measurements(
    const Eigen::VectorXd& t, 
    const Eigen::MatrixXd& states, 
    const float meas_freq, 
    const double station_lat, 
    const double station_lon,
    const double station_el = 0.0,
    const double GMA = 0.0
){

    // Allocating memory for the measurements
    // 3 measurements per measurement step
    Eigen::MatrixXd meas(static_cast<int>(states.rows()), static_cast<int>(states.cols()));

    // Getting station position
    Eigen::Vector3d station_pos;
    station_pos << R_E * cos(station_lat)*cos(station_lon), 
                    R_E * cos(station_lat)*sin(station_lon),
                    R_E * sin(station_lat); 

    // Getting range vectors
    // Attempting column wise subtraction of position elements
    Eigen::MatrixXd rho_vec = states.topRows(3).colwise() - station_pos;
    
    // Assigns the first measurement row to the norm of each column in rho_vec
    // This should give the magnitude of relative position at each time step.
    meas.topRows(1) = rho_vec.colwise().norm();

    const double omega = 2*M_PI/(84000.0);
    double theta; // rotation of Earth

    // Looping through t and calculating altitude and azimuth for each step
    for (int i = 0; i < static_cast<int>(meas.cols()); i++){
        theta = omega * t(i) + GMA;
        // Rotating position vector to station ENU frame
        Eigen::Vector3d pos_enu = ecf2enu(station_lat, station_lon) * eci2ecf(theta) * rho_vec.col(i);

        // Obtaining azimuth
        meas(1, i) = atan2(pos_enu(0), pos_enu(1));

        // Obtaining altitude
        meas(2, i) = asin(pos_enu(2) / meas(0, i));
    }

    return meas;
}