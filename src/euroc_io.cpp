// euroc_io.cpp
#include "euroc_io.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <limits>
#include <vector>
#include <cmath>

static bool load_TBS(const std::string& yaml_file, Eigen::Matrix4d& T) {
    YAML::Node config = YAML::LoadFile(yaml_file);
    if (!config["T_BS"]) return false;

    auto data = config["T_BS"]["data"];
    if (!data || data.size() != 16) return false;

    for (int i = 0; i < 16; ++i) {
        T(i / 4, i % 4) = data[i].as<double>();
    }
    return true;
}

static bool load_csv_imu(const std::string& filename,
                         std::vector<double>& timestamps,
                         std::vector<Eigen::Vector3d>& omega,
                         std::vector<Eigen::Vector3d>& acc)
{
    std::ifstream file(filename); std::string line;
    if (!file.is_open()) return false;
    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line); std::string v;
        std::vector<double> row;
        while (std::getline(ss, v, ',')) row.push_back(std::stod(v));
        if (row.size() == 7) {
            timestamps.push_back(row[0] * 1e-9);
            omega.emplace_back(row[1], row[2], row[3]);
            acc.emplace_back(row[4], row[5], row[6]);
        }
    }
    return true;
}

static bool load_csv_cam(const std::string& filename, const std::string& base_path,
                         std::vector<double>& timestamps,
                         std::vector<std::string>& filenames)
{
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) return false;

    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        // Eliminar posibles retornos de carro o espacios
        line.erase(std::remove_if(line.begin(), line.end(),
                                  [](unsigned char c) { return std::isspace(c); }),
                   line.end());

        std::stringstream ss(line);
        std::string ts, fn;

        std::getline(ss, ts, ',');
        std::getline(ss, fn);

        if (ts.empty() || fn.empty()) continue;

        timestamps.push_back(std::stod(ts) * 1e-9);
        filenames.push_back(base_path + "/" + fn);


    }
    return true;
}


static bool load_csv_gt(const std::string& filename,
                        std::vector<double>& timestamps,
                        std::vector<Eigen::Quaterniond>& quats,
                        std::vector<Eigen::Vector3d>& biases_g,
                        std::vector<Eigen::Vector3d>& biases_a,
                        std::vector<Eigen::Vector3d>& velocities,
                        std::vector<Eigen::Vector3d>& gravities,
                        std::vector<Eigen::Vector3d>& positions)
{
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) return false;

    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string v;
        std::vector<double> row;
        while (std::getline(ss, v, ',')) row.push_back(std::stod(v));
        if (row.size() >= 17) {
            timestamps.push_back(row[0] * 1e-9);
            positions.emplace_back(row[1], row[2], row[3]);  // x, y, z
            quats.emplace_back(row[4], row[5], row[6], row[7]);  // qw, qx, qy, qz
            velocities.emplace_back(row[8], row[9], row[10]);
            biases_g.emplace_back(row[11], row[12], row[13]);
            biases_a.emplace_back(row[14], row[15], row[16]);
            gravities.emplace_back(0.0, 0.0, -9.81);
        }
    }
    return true;
}



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
                        int& starting_frame_)
{
    std::string imu_file = euroc_path + "/mav0/imu0/data.csv";
    std::string cam_csv = euroc_path + "/mav0/cam0/data.csv";
    std::string cam_img_dir = euroc_path + "/mav0/cam0/data";
    std::string gt_file = euroc_path + "/mav0/state_groundtruth_estimate0/data.csv";
    std::string cam_yaml = euroc_path + "/mav0/cam0/sensor.yaml";
    std::string imu_yaml = euroc_path + "/mav0/imu0/sensor.yaml";

    bool ok = load_csv_imu(imu_file, timu, omega, acc)
            && load_csv_cam(cam_csv, cam_img_dir, tcam, cam0_image_names)
            && load_csv_gt(gt_file, tgt, qgt, bg_gt, ba_gt, v_gt, g_gt, p_gt)
            && load_TBS(cam_yaml, tbodycam)
            && load_TBS(imu_yaml, tbodyimu);
    if (!ok) return false;

    std::cerr << "IMU samples: " << timu.size() << std::endl;
    std::cerr << "CAM samples: " << tcam.size() << std::endl;
    std::cerr << "GT  samples: " << tgt.size() << std::endl;

    // Cálculo del índice starting_frame_
    double t0_cam = tcam.front();
    double t0_imu = timu.front();
    double t0_gt  = tgt.front();

    if (t0_cam >= t0_imu && t0_cam >= t0_gt) {
        starting_frame_ = 0;
    } else if (t0_imu >= t0_cam && t0_imu >= t0_gt) {
        // Buscar el índice de la imagen más cercana a t_imu[0]
        double min_diff = std::numeric_limits<double>::max();
        for (size_t i = 0; i < tcam.size(); ++i) {
            double diff = std::abs(timu[0] - tcam[i]);
            if (diff < min_diff) {
                min_diff = diff;
                starting_frame_ = static_cast<int>(i);
            }
        }
    } else {
        // Buscar el índice de la imagen más cercana a t_gt[0]
        double min_diff = std::numeric_limits<double>::max();
        for (size_t i = 0; i < tcam.size(); ++i) {
            double diff = std::abs(tgt[0] - tcam[i]);
            if (diff < min_diff) {
                min_diff = diff;
                starting_frame_ = static_cast<int>(i);
            }
        }
    }

    return true;
    
}
