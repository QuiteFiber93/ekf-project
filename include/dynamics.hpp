#ifndef DYNAMICS_HPP
#define DYNAMICS_HPP

#include <eigen3/Eigen/Core>
#include "types.hpp"

inline Vec6 dynamics(const double t, const Vec6& y, const double mu)
{
    (void)t;

    Vec3 r = y.head<3>();
    Vec3 v = y.tail<3>();

    double r_norm = r.norm();
    double r3 = r_norm * r_norm * r_norm;

    Vec6 ydot;
    ydot.head<3>() = v;
    ydot.tail<3>() = -mu * r / r3;

    return ydot;
}

inline Mat6 dynamics_jacobian(double t, const Vec6& y, const double mu)
{
    (void)t;

    Vec3 r = y.head<3>();
    double r_norm = r.norm();
    double r3 = r_norm * r_norm * r_norm;
    double r5 = r3 * r_norm * r_norm;

    Mat6 jacobian = Mat6::Zero();

    // [0   | I ]
    // [A   | 0 ]
    jacobian.block<3,3>(0, 3) = Mat3::Identity();
    jacobian.block<3,3>(3, 0) = -mu / r3 * Mat3::Identity()
                                + 3.0 * mu / r5 * (r * r.transpose());

    return jacobian;
}

#endif