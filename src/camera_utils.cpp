#include "camera_utils.h"
#include <yaml-cpp/yaml.h>

CameraCalibration load_camera_calibration(const std::string& yaml_file) {
    YAML::Node config = YAML::LoadFile(yaml_file);
    CameraCalibration calib;

    auto intrinsics = config["intrinsics"];
    double fx = intrinsics[0].as<double>();
    double fy = intrinsics[1].as<double>();
    double cx = intrinsics[2].as<double>();
    double cy = intrinsics[3].as<double>();

    calib.K << fx, 0, cx,
               0, fy, cy,
               0, 0, 1;

    auto dist = config["distortion_coefficients"];
    for (size_t i = 0; i < dist.size(); ++i)
        calib.distortion_params.push_back(dist[i].as<double>());

    auto resolution = config["resolution"];
    if (resolution && resolution.size() == 2)
        calib.image_size = { resolution[0].as<int>(), resolution[1].as<int>() };

    return calib;
}
