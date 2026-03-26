#include <eigen3/Eigen/Core>
#include "constants.hpp"

Eigen::Vector3d eci2ecf(const double theta, const Eigen::MatrixXd& pos){
    Eigen::Matrix3d R;
    R << cos(theta), -sin(theta), 0,
        sin(theta), cos(theta), 0,
        0, 0, 1;

    return R * pos;
}

Eigen::Vector3d ecf2enu(const double lat, const double lon, const Eigen::MatrixXd& pos){
    Eigen::Matrix3d R;
    R << -sin(lon), cos(lon), 0,
        -sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat),
        cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat);
}

Eigen::MatrixXd measurements(
    const Eigen::VectorXd& t, 
    const Eigen::MatrixXd& states, 
    const float meas_freq, 
    const double station_lat, 
    const double station_lon,
    const double station_el = 0.0
){

    // Allocating memory for the measurements
    // 3 measurements per measurement step
    // the number of measurement steps is the number of total time steps
    // divided by the period of measurements
    // Then, make sure everything is an int
    Eigen::MatrixXd meas(static_cast<int>(states.rows()/2), static_cast<int>(states.cols()/meas_freq));

    // Getting station position
    Eigen::Vector3d station_pos;

    // Getting range vectors
    // Attempting column wise subtraction of position elements
    Eigen::MatrixXd rho_vec = states.topRows(3).colwise() - station_pos;
    
    // Assigns the first measurement row to the norm of each column in rho_vec
    // This should give the magnitude of relative position at each time step.
    meas.topRows(1) = rho_vec.colwise().norm();


    for (int i = 0; i < static_cast<int>(t.size()); i++){
        // Loop through and calculat
    }


    
}