#ifndef TYPES_HPP
#define TYPES_HPP

#include <eigen3/Eigen/Core>

// Fixed-size types to eliminate heap allocation in hot paths.
// Eigen stores these on the stack and can fully unroll + vectorize operations.
using Vec3  = Eigen::Matrix<double, 3, 1>;
using Vec6  = Eigen::Matrix<double, 6, 1>;
using Vec42 = Eigen::Matrix<double, 42, 1>;
using Mat3  = Eigen::Matrix<double, 3, 3>;
using Mat6  = Eigen::Matrix<double, 6, 6>;
using Mat36 = Eigen::Matrix<double, 3, 6>;

#endif