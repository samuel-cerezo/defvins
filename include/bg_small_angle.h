// bg_small_angle.h
#pragma once

#include <Eigen/Dense>
#include <vector>

/**
 * @brief Estimate gyroscope bias using the small-angle approximation.
 *
 * This function computes an approximate gyroscope bias correction assuming
 * that all angular displacements are small, which allows the IMU rotation 
 * to be linearized as a product of small exponential maps.
 *
 * The IMU preintegrated rotation is compared to the visual rotation estimate,
 * and the bias is estimated by minimizing the discrepancy under the small-angle model.
 *
 * Reference:
 *     \f[
 *     \prod_{k=i}^{j-1} \exp((\omega_k - b_g)\Delta t)
 *     \approx \prod_{k=i}^{j-1} \exp(\omega_k \Delta t) \cdot \exp(-b_g \Delta t)
 *     \f]
 *
 * @param omega_all  Vector of 3D angular velocity samples (rad/s), with bias included.
 * @param deltat     Scalar time step between IMU measurements (s).
 * @param Rpreint    Preintegrated rotation from IMU (SO(3) matrix).
 * @param Rij        Relative rotation between camera frames (SO(3) matrix).
 * @param tbodycam   4x4 homogeneous transform from body to camera.
 * @param tbodyimu   4x4 homogeneous transform from body to IMU.
 * @return           Estimated gyroscope bias vector (3x1, in rad/s).
 */
Eigen::Vector3d bg_small_angle(
    const std::vector<Eigen::Vector3d>& omega_all,
    double deltat,
    const Eigen::Matrix3d& Rpreint,
    const Eigen::Matrix3d& Rij,
    const Eigen::Matrix4d& tbodycam,
    const Eigen::Matrix4d& tbodyimu);

Eigen::Vector3d bg_constVel(
    const std::vector<Eigen::Vector3d>& omega_all,
    double deltat,
    const Eigen::Matrix3d& Rpreint,
    const Eigen::Matrix3d& Rij,
    const Eigen::Matrix4d& tbodycam,
    const Eigen::Matrix4d& tbodyimu);
