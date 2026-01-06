#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <ceres/ceres.h>

std::tuple<Eigen::Vector3d, double, ceres::Solver::Summary>
bg_optimization(const std::vector<Eigen::Vector3d>& omega_all_vec,
                      double deltat,
                      const Eigen::Matrix3d& Rij,
                      const Eigen::Vector3d& bg0,
                      const Eigen::Matrix4d& tbodycam,
                      const Eigen::Matrix4d& tbodyimu);
