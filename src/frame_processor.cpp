#include "frame_processor.h"
#include "feature_tracking.h"
#include "lie_utils.h"
#include "bg_small_angle.h"
#include "bg_optimization.h"
#include "c2p_wrapper.h"
#include <ceres/ceres.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include <vector>


void evaluate_rotation_error(const std::vector<Eigen::Vector3d>& omega_all_vec,
                              double deltat,
                              const Eigen::Matrix3d& Rij,
                              const Eigen::Vector3d& bg,
                              const Eigen::Matrix4d& tbodycam,
                              const Eigen::Matrix4d& tbodyimu) {
    // retrieving rotation matrices
    Eigen::Matrix3d R_imu = tbodyimu.block<3, 3>(0, 0);
    Eigen::Matrix3d R_cam = tbodycam.block<3, 3>(0, 0);

    // preintegrated rotation including bias correction
    Eigen::Matrix3d Rpreint = Eigen::Matrix3d::Identity();
    for (const auto& omega : omega_all_vec) {
        Eigen::Vector3d omega_corr = omega - bg;
        Rpreint = Rpreint * lie::ExpMap(omega_corr * deltat);
    }

    // rotation in the camera frame 
    Eigen::Matrix3d R_integrated = R_imu * Rpreint * R_imu.transpose();
    Eigen::Matrix3d R_measured = R_cam * Rij * R_cam.transpose();

    Eigen::Matrix3d R_error = R_integrated.transpose() * R_measured;
    Eigen::Vector3d rot_error = lie::LogMapRobust(R_error);
    double error_norm = rot_error.norm();

    std::cout << "Rotation error vector: " << rot_error.transpose() << std::endl;
    std::cout << "Rotation error norm: " << error_norm << std::endl;
}

void process_frame_pair(int starting_frame,
                        const std::vector<std::string>& cam0_image_names,
                        const std::vector<double>& tcam,
                        const std::vector<double>& timu,
                        const std::vector<Eigen::Vector3d>& omega,
                        const std::vector<Eigen::Vector3d>& acc,
                        double deltat,
                        const std::vector<double>& tgt,
                        const std::vector<Eigen::Quaterniond>& qgt,
                        const std::vector<Eigen::Vector3d>& bg_gt,
                        const std::vector<Eigen::Vector3d>& ba_gt,
                        const std::vector<Eigen::Vector3d>& v_gt,
                        const std::vector<Eigen::Vector3d>& g_gt,
                        const std::vector<Eigen::Vector3d>& p_gt,
                        const Eigen::Matrix3d& K,
                        const cv::Mat& K_cv,
                        const cv::Mat& dist_cv,
                        const Eigen::Matrix4d& tbodycam,
                        const Eigen::Matrix4d& tbodyimu,
                        std::ofstream& log_bias_file) {
    std::cout << "\n[Frame " << starting_frame << "] Processing image pair" << std::endl;

    // Load and track features
    std::string img_path_1 = cam0_image_names[starting_frame];
    std::string img_path_2 = cam0_image_names[starting_frame + 1];
    std::vector<cv::Point2f> old_points, new_points;
    cv::Mat image_1, image_2;

    if (!detect_and_track_features(img_path_1, img_path_2, old_points, new_points, image_1, image_2)) return;
    if (old_points.size() < 8) return;

    //cv::undistortPoints(old_points, old_undist, K_cv, dist_cv, cv::noArray(), K_cv);
    //cv::undistortPoints(new_points, new_undist, K_cv, dist_cv, cv::noArray(), K_cv);

    // Undistort to pixel coordinates using P=K 
    cv::Mat old_undist, new_undist, mask;
    cv::undistortPoints(old_points, old_undist, K_cv, dist_cv, cv::noArray(), K_cv);
    cv::undistortPoints(new_points, new_undist, K_cv, dist_cv, cv::noArray(), K_cv);

    // Transform to Eigen
    int num_points = old_undist.rows;
    Eigen::MatrixXd f0(3, num_points), f1(3, num_points);

    Eigen::Matrix3d K_eigen;
    cv::cv2eigen(K_cv, K_eigen);  

    Eigen::Matrix3d Kinv = K_eigen.inverse();

    for (int i = 0; i < num_points; ++i) {
        double x0 = old_undist.at<cv::Point2f>(i).x;
        double y0 = old_undist.at<cv::Point2f>(i).y;
        double x1 = new_undist.at<cv::Point2f>(i).x;
        double y1 = new_undist.at<cv::Point2f>(i).y;

        Eigen::Vector3d p0 = Kinv * Eigen::Vector3d(x0, y0, 1.0);
        Eigen::Vector3d p1 = Kinv * Eigen::Vector3d(x1, y1, 1.0);

        f0.col(i) = p0.normalized();
        f1.col(i) = p1.normalized();
    }


    old_undist.convertTo(old_undist, CV_64F);
    new_undist.convertTo(new_undist, CV_64F);

    cv::theRNG().state = 42;
    // Estimate Essential matrix to get inliers
    cv::Mat E = cv::findEssentialMat(old_undist, new_undist, K_cv, cv::RANSAC, 0.999, 1.0, mask);

   // debug_essential(old_undist, new_undist, mask);

    std::vector<Eigen::Vector3d> f0_inliers, f1_inliers;
    int total_valid = 0, discarded = 0;

    for (int i = 0; i < num_points; ++i) {
        double norm_f0 = f0.col(i).norm();
        double norm_f1 = f1.col(i).norm();

        if (mask.at<uchar>(i, 0) && norm_f0 > 1e-6 && norm_f1 > 1e-6 &&
            std::isfinite(norm_f0) && std::isfinite(norm_f1)) {
            
            f0_inliers.push_back(f0.col(i).normalized());
            f1_inliers.push_back(f1.col(i).normalized());
            total_valid++;
        } else {
            discarded++;
        }
    }

        
    C2PResult c2p_result;
    try {
        c2p_result = run_c2p(f0_inliers, f1_inliers, starting_frame);
    } catch (const std::exception& e) {
        std::cerr << "[Frame " << starting_frame << "] Error in run_c2p: " << e.what() << std::endl;
        return;
    }

    Eigen::Matrix3d R01 = c2p_result.R;
    Eigen::Vector3d t_dir = c2p_result.t.normalized();  // direction without scale


    // Extract IMU segment
    double t1 = tcam[starting_frame];
    double t2 = tcam[starting_frame + 1];
    std::vector<double> imu_times_segment;
    std::vector<Eigen::Vector3d> omega_segment;
    std::vector<Eigen::Vector3d> acc_segment;
    

    for (size_t i = 0; i < timu.size(); ++i)
        if (timu[i] >= t1 && timu[i] <= t2) {
            imu_times_segment.push_back(timu[i]);
            omega_segment.push_back(omega[i]);
            acc_segment.push_back(acc[i]);
        }

    if (imu_times_segment.size() < 2) return;
    acc_segment.resize(acc_segment.size() - 1);  

    if (acc_segment.empty()) {
        std::cerr << "acc_segment is empty!" << std::endl;
        return;
    }

    // Preintegration
    std::vector<double> dts;
    for (size_t i = 1; i < imu_times_segment.size(); ++i)
        dts.push_back(imu_times_segment[i] - imu_times_segment[i - 1]);
    omega_segment.resize(omega_segment.size() - 1);

    Eigen::Matrix3d Rpreint = Eigen::Matrix3d::Identity();
    std::vector<Eigen::Vector3d> omega_all;
    for (size_t j = 0; j < omega_segment.size(); ++j) {
        Eigen::Vector3d dtheta = omega_segment[j] * dts[j];
        Rpreint *= lie::ExpMap(dtheta);
        omega_all.push_back(omega_segment[j]);
    }
    
    Eigen::Matrix3d R_imu = tbodyimu.block<3, 3>(0, 0);
    Eigen::Matrix3d R_cam = tbodycam.block<3, 3>(0, 0);
    //std::cout << "R_cam:\n" << R_cam << std::endl;
    //std::cout << "R_imu:\n" << R_imu << std::endl;


    //----------------------- bias estimation ----------------------------
    // Small angle bias estimation
    auto t_small_start = std::chrono::high_resolution_clock::now();
    Eigen::Vector3d b_g = bg_small_angle(omega_all, deltat, Rpreint, R01, tbodycam, tbodyimu);
    auto t_small_end = std::chrono::high_resolution_clock::now();

    // const vel bias estimation
    auto t_constVel_start = std::chrono::high_resolution_clock::now();
    Eigen::Vector3d b_g_constVel = bg_constVel(omega_all, deltat, Rpreint, R01, tbodycam, tbodyimu);
    auto t_constVel_end = std::chrono::high_resolution_clock::now();

    // Optimization-based estimation
    std::vector<Eigen::Vector3d> omega_scaled;
    for (const auto& o : omega_all)
        omega_scaled.push_back(o / deltat);

    Eigen::Vector3d b_g_init = Eigen::Vector3d::Zero();
    auto t_opt_start = std::chrono::high_resolution_clock::now();
    auto [bgopt, cost, summary] = bg_optimization(omega_all, deltat, R01, b_g_init, tbodycam, tbodyimu);
    auto t_opt_end = std::chrono::high_resolution_clock::now();

    // Compare with ground truth
    auto it_gt = std::min_element(tgt.begin(), tgt.end(), [t1](double a, double b) {
        return std::abs(a - t1) < std::abs(b - t1);
    });
    size_t idx_gt1 = std::distance(tgt.begin(), it_gt);
    Eigen::Vector3d bg_gt_ = bg_gt[idx_gt1];

    double error_b_g = (bg_gt_ - b_g).norm();
    double error_b_gopt = (bg_gt_ - bgopt).norm();
    double error_b_g_constVel = (bg_gt_ - b_g_constVel).norm();

    std::chrono::duration<double, std::micro> small_elapsed = t_small_end - t_small_start;
    std::chrono::duration<double, std::micro> opt_elapsed = t_opt_end - t_opt_start;
    std::chrono::duration<double, std::micro> constVel_elapsed = t_constVel_end - t_constVel_start;

    std::cout   << "[Frame " << starting_frame 
                << "] --> small: " << error_b_g 
                << " t:" << small_elapsed.count() << "µs /"
                << "constVel:" << error_b_g_constVel
                << "t:" << constVel_elapsed.count() << "µs /"
                << "opt: " << error_b_gopt 
                << " t:" << opt_elapsed.count() << "µs" << std::endl;

    log_bias_file   << starting_frame << ","
                    << std::fixed << std::setprecision(9) 
                    << error_b_g << "," << small_elapsed.count() << ","
                    << error_b_gopt << "," << opt_elapsed.count() << ","
                    << error_b_g_constVel << "," << constVel_elapsed.count()
                    <<"\n";
}
