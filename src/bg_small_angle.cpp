// bg_small_angle.cpp
#include "bg_small_angle.h"
#include "lie_utils.h"
#include <iostream>
#include <Eigen/Dense>

// Estimate gyroscope bias using small-angle approximation.
// Inputs:
// - omega_all: sequence of biased angular velocity measurements (in body frame)
// - deltat: sampling time interval [s]
// - Rpreint: rotation from IMU preintegration
// - Rij: relative rotation from vision (camera)
// - tbodycam: 4x4 extrinsic transform from body to camera
// - tbodyimu: 4x4 extrinsic transform from body to IMU
//
// Output:
// - Estimated gyroscope bias vector

Eigen::Vector3d bg_small_angle(
    const std::vector<Eigen::Vector3d>& omega_all,
    double deltat,
    const Eigen::Matrix3d& Rpreint,
    const Eigen::Matrix3d& R_ij,
    const Eigen::Matrix4d& T_body_cam,
    const Eigen::Matrix4d& T_body_imu)
{
    // Extract 3x3 rotation matrices from extrinsics
    Eigen::Matrix3d R_imu = T_body_imu.topLeftCorner<3, 3>();
    Eigen::Matrix3d R_cam = T_body_cam.topLeftCorner<3, 3>();

    // Compose integrated and measured rotation in the same reference frame
    Eigen::Matrix3d R_integrated = R_imu * Rpreint * R_imu.transpose();
    Eigen::Matrix3d R_measured = R_cam * R_ij * R_cam.transpose();
    Eigen::Matrix3d R_error = R_integrated.transpose() * R_cam * R_ij * R_cam.transpose();
    Eigen::Vector3d log_rot = lie::LogMap(R_error);
    
    //std::cout << "[DEBUG] Rij:\n" << R_ij << std::endl;
    const int n = static_cast<int>(omega_all.size());
    Eigen::Vector3d b_g = - (1.0 / (n * deltat)) * (log_rot);

    return b_g;
}

Eigen::Vector3d bg_constVel(
    const std::vector<Eigen::Vector3d>& omega_all,
    double deltat,
    const Eigen::Matrix3d& Rpreint,
    const Eigen::Matrix3d& R_ij,
    const Eigen::Matrix4d& T_body_cam,
    const Eigen::Matrix4d& T_body_imu)
{
    // Extract 3x3 rotation matrices from extrinsics
    Eigen::Matrix3d R_imu = T_body_imu.topLeftCorner<3, 3>();
    Eigen::Matrix3d R_cam = T_body_cam.topLeftCorner<3, 3>();
    const int n = static_cast<int>(omega_all.size());

    // Compose integrated and measured rotation in the same reference frame
    Eigen::Vector3d log_rot = lie::LogMap(Rpreint);
    Eigen::Vector3d average_omega = (1.0 / (n * deltat)) * (log_rot);
    Eigen::Matrix3d R_omega = lie::ExpMap(average_omega*deltat);
    Eigen::Matrix3d R_omega_world = R_imu * R_omega * R_imu.transpose();
    Eigen::Matrix3d R_measured = R_cam * R_ij * R_cam.transpose();
    Eigen::Vector3d logRij = (lie::LogMap(R_measured)) / n;
    Eigen::Matrix3d R_log = lie::ExpMap(logRij);

    Eigen::Vector3d b_g_constVel = -(1.0 / deltat) * lie::LogMap(R_omega_world.transpose()*R_log);

    return b_g_constVel;
}