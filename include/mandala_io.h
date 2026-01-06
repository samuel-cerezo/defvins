#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

bool load_mandala_sequence(const std::string& mandala_path,
                         std::vector<double>& timu,
                         std::vector<Eigen::Vector3d>& omega,
                         std::vector<Eigen::Vector3d>& acc,
                         std::vector<double>& tcam,
                         std::vector<double>& tdepth,                         
                         std::vector<std::string>& cam0_image_names,
                         std::vector<std::string>& depth_image_names,
                         std::vector<double>& tgt,
                         std::vector<Eigen::Quaterniond>& qgt,
                         std::vector<Eigen::Vector3d>& bg_gt,
                         std::vector<Eigen::Vector3d>& ba_gt,
                         std::vector<Eigen::Vector3d>& v_gt,
                         std::vector<Eigen::Vector3d>& g_gt,
                         std::vector<Eigen::Vector3d>& p_gt,
                         Eigen::Matrix4d& tbodycam,
                         Eigen::Matrix4d& tbodyimu,
                         Eigen::Matrix4d& tflangebody,                         
                        int& starting_frame_);