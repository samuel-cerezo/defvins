#pragma once

#include <Eigen/Dense>
#include <string>
#include <utility>
#include <vector>

// Structure to hold intrinsic calibration and distortion
struct CameraCalibration {
    Eigen::Matrix3d K;
    std::vector<double> distortion_params;
    std::pair<int, int> image_size;
};

// Loads camera intrinsics and distortion from EuRoC-style YAML file
CameraCalibration load_camera_calibration(const std::string& yaml_file);
