#pragma once    // include este header una sola vez
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

Eigen::Matrix4d poseToHom(const Eigen::Quaterniond &q,
                          const Eigen::Vector3d &t);

void homToPose(const Eigen::Matrix4d& T,
                Eigen::Quaterniond &q,
                Eigen::Vector3d &t);                       

std::vector<Eigen::Quaterniond> transformQuaternionsLeft (const std::vector<std::pair<Eigen::Quaterniond, Eigen::Vector3d>>& poses,
                                                   const Eigen::Matrix4d& transform);
std::vector<Eigen::Quaterniond> transformQuaternionsRight (const std::vector<std::pair<Eigen::Quaterniond, Eigen::Vector3d>>& poses,
                                                   const Eigen::Matrix4d& transform);

std::vector<Eigen::Vector3d> transformPositionsLeft (const std::vector<std::pair<Eigen::Quaterniond, Eigen::Vector3d>>& poses,
                                                 const Eigen::Matrix4d& transform);


std::vector<Eigen::Vector3d> transformPositionsRight (const std::vector<std::pair<Eigen::Quaterniond, Eigen::Vector3d>>& poses,
                                                  const Eigen::Matrix4d& transform);