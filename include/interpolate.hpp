#ifndef INTERPOLATE_HPP
#define INTERPOLATE_HPP

#include <eigen3/Eigen/Core>

Eigen::VectorXd hermite_interp(
    double t0, const Eigen::VectorXd& y0, const Eigen::VectorXd& f0,
    double t1, const Eigen::VectorXd& y1, const Eigen::VectorXd& f1,
    double t
){
    double dt = t1 - t0;
    double s = (t - t0) / dt;
    double s2 = s*s;
    double s3 = s2*s;

    double h00 = 2*s3 - 3*s2 + 1;
    double h10 = s3 - 2*s2 + s;
    double h01 = -2*s3 + 3*s2;
    double h11 = s3 - s2;

    return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1;
}

#endif