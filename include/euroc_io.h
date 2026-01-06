// euroc_io.h
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

bool load_euroc_sequence(const std::string& euroc_path,
                         std::vector<double>& timu,
                         std::vector<Eigen::Vector3d>& omega,
                         std::vector<Eigen::Vector3d>& acc,
                         std::vector<double>& tcam,
                         std::vector<std::string>& cam0_image_names,
                         std::vector<double>& tgt,
                         std::vector<Eigen::Quaterniond>& qgt,
                         std::vector<Eigen::Vector3d>& bg_gt,
                         std::vector<Eigen::Vector3d>& ba_gt,
                         std::vector<Eigen::Vector3d>& v_gt,
                         std::vector<Eigen::Vector3d>& g_gt,
                         std::vector<Eigen::Vector3d>& p_gt,
                         Eigen::Matrix4d& tbodycam,
                         Eigen::Matrix4d& tbodyimu,
                         int& starting_frame_);
