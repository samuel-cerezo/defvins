#pragma once
#include <optional>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "deformation_graph.h"  

void saveTrajectoryToCsv(const std::string& filename,
                         const std::vector<int>& optimized_kfs,
                         const std::vector<double>& tcam,
                         const std::unordered_map<int, KFPose>& g_cam_poses);

double rmse(const Eigen::Vector3d &est, const Eigen::Vector3d &gt);

double rmse_ang(const Eigen::Vector3d &est, const Eigen::Vector3d &gt);

double median(std::vector<double> v);

Eigen::Matrix3d quatToRot(const Eigen::Quaterniond& q_in);

double mean(std::vector<double> v);

std::string make_csv_basename(const std::string& seq_name, int ref_idx);

std::string make_results_path(const std::string& seq_name, int ref_idx);

std::string make_project_results_path(const std::string& seq_input_path,
                                             const std::string& seq_name,
                                             int ref_idx);



Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& wdt);

Eigen::Matrix3d quatToRotSafe(const Eigen::Quaterniond& q_in);

// p1, p2 en pixeles (cv::Point2f). Devuelve mediana del residual (px) tras compensar la rotación.
double rotCompParallaxMedianPx(const std::vector<cv::Point2f>& pts1,
                               const std::vector<cv::Point2f>& pts2,
                               const Eigen::Matrix3d& K,
                               const Eigen::Matrix3d& R_k_k1);

std::string infer_sequence_name(const std::string& input_path);

std::optional<Eigen::Matrix3d> compute_Rij_with_c2p(
    const std::string& img_path_i,
    const std::string& img_path_j,
    const Eigen::Matrix3d& K_eigen,
    const cv::Mat& K_cv,
    const cv::Mat& dist_cv);

