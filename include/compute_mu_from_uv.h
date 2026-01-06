
#pragma once
#include <Eigen/Dense>

inline Eigen::Vector3d computeMuFromUV(const Eigen::Vector2d& uv, const Eigen::Matrix3d& K) {
    Eigen::Vector3d pixel_hom(uv.x(), uv.y(), 1.0);
    Eigen::Vector3d mu = K * pixel_hom;
    return mu.normalized();
}
