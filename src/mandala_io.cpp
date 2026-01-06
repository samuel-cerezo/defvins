#include "mandala_io.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <limits>
#include <vector>
#include <cmath>
#include <string>
#include <cctype>



static bool load_TBS(const std::string& yaml_file, Eigen::Matrix4d& T) {
    YAML::Node config = YAML::LoadFile(yaml_file);
    if (!config["T_BS"]) return false;

    auto data = config["T_BS"]["data"];
    if (!data || data.size() != 16) return false;

    for (int i = 0; i < 16; ++i) {
        T(i / 4, i % 4) = data[i].as<double>();
    }
    // --- CONVERSIÓN mm -> m (solo traslación) ---
    T.block<3,1>(0,3) /= 1000.0;
    std::cout << "[EXTR] |t|=" << T.block<3,1>(0,3).norm()
          << " t=" << T.block<3,1>(0,3).transpose() << "\n";

    return true;
}

static bool load_txt_imu(const std::string& filename,
                         std::vector<double>& timestamps,
                         std::vector<Eigen::Vector3d>& omega,
                         std::vector<Eigen::Vector3d>& acc)
{
    std::ifstream file(filename); std::string line;
    if (!file.is_open()) return false;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double v;
        // Lee números separados por espacios o tabs
        while (ss >> v) row.push_back(v);

        if (row.size() == 7) {
            timestamps.push_back(row[0]);   // * 1e-9
            acc.emplace_back(row[1], row[2], row[3]);
            omega.emplace_back(row[4], row[5], row[6]);
        }
    }

    return true;
}

static bool load_txt_cam(const std::string& filename, const std::string& base_path,
                         std::vector<double>& timestamps_rgb,
                         std::vector<double>& timestamps_depth,
                         std::vector<std::string>& filenames_rgb,
                         std::vector<std::string>& filenames_depth)
{
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) return false;

    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        if (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();

        std::stringstream ss(line);
        std::string ts_RGB, rgb_file, ts_depth, depth_file;

        ss >> ts_RGB >> rgb_file >> ts_depth >> depth_file; 

        if (ts_RGB.empty() || rgb_file.empty() || ts_depth.empty() || depth_file.empty()) continue;

        // Convierte timestamp y guarda ruta completa
        timestamps_rgb.push_back(std::stod(ts_RGB));
        timestamps_depth.push_back(std::stod(ts_depth));
        filenames_rgb.push_back(base_path + "/" + rgb_file);
        filenames_depth.push_back(base_path + "/" + depth_file);
    }

    return true;
}


static bool load_txt_flange(const std::string& filename,
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
        std::vector<double> row;
        double v;
        // Lee números separados por espacios o tabs
        while (ss >> v) row.push_back(v);

        if (row.size() == 8) {
            timestamps.push_back(row[0]);   // * 1e-9
            quats.emplace_back(row[4], row[1], row[2], row[3]);  // qw, qx, qy, qz
            positions.emplace_back(row[5]/1000.0, row[6]/1000.0, row[7]/1000.0);  // x, y, z
            velocities.emplace_back(0.0, 0.0, 0.0);
            biases_g.emplace_back(0.0, 0.0, 0.0);
            biases_a.emplace_back(0.0, 0.0, 0.0);
            gravities.emplace_back(0.0, -9.81, 0.0);
            std::cout << "[GT] t=" << timestamps.back()
                    << " q(wxyz)=" << quats.back().w() << " "
                                    << quats.back().x() << " "
                                    << quats.back().y() << " "
                                    << quats.back().z()
                    << " |q|=" << quats.back().norm()
                    << " p[m]=" << positions.back().transpose()
                    << "\n";

        }
    }

    return true;
}



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
                        int& starting_frame_)
{
    std::string imu_file = mandala_path + "/imu_data.txt";
    std::string cam_file = mandala_path + "/associations.txt";
    std::string cam_img_dir = mandala_path ;
    std::string flange_file = mandala_path + "/flange_poses.txt";
    std::string cam_yaml = mandala_path + "/params/camera.yaml";
    std::string imu_yaml = mandala_path + "/params/imu.yaml";
    std::string flange_yaml = mandala_path + "/params/flange.yaml";

    bool ok = load_txt_imu(imu_file, timu, omega, acc)
            && load_txt_cam(cam_file, cam_img_dir, tcam, tdepth, cam0_image_names, depth_image_names)
            && load_txt_flange(flange_file, tgt, qgt, bg_gt, ba_gt, v_gt, g_gt, p_gt)
            && load_TBS(cam_yaml, tbodycam)
            && load_TBS(imu_yaml, tbodyimu)
            && load_TBS(flange_yaml, tflangebody);
    if (!ok) return false;

    std::cerr << "IMU samples: " << timu.size() << std::endl;
    std::cerr << "CAM samples: " << tcam.size() << std::endl;
    std::cerr << "Depth samples: " << tdepth.size() << std::endl;
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
