#include "utils.h"
#include <optional>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "feature_tracking.h" 
#include <opencv2/opencv.hpp>
#include "c2p_wrapper.h"
#include "track_manager.h"
#include "feature_tracker.h"
#include "camera_utils.h"
#include "frame_processor.h"
#include "euroc_io.h"
#include <filesystem> 
#include <pybind11/embed.h>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip> 
#include <Eigen/Dense>
#include <map>
#include <vector>
#include <utility> 
#include <algorithm>
#include <numeric>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <sstream>
#include <map>           
#include <type_traits>   
#include <unordered_map>
#include "lie_utils.h"
#include <fstream>
#include <random>

namespace fs = std::filesystem;


// Guarda en CSV: k, tiempo, posicion y rotación en cuaternión
void saveTrajectoryToCsv(const std::string& filename,
                         const std::vector<int>& optimized_kfs,
                         const std::vector<double>& tcam,
                         const std::unordered_map<int, KFPose>& g_cam_poses)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "[CSV] No pude abrir " << filename << " para escribir.\n";
        return;
    }

    ofs << std::fixed << std::setprecision(9);
    ofs << "k,t,px,py,pz,qx,qy,qz,qw\n";

    for (int k : optimized_kfs) {
        auto it = g_cam_poses.find(k);
        if (it == g_cam_poses.end()) continue;

        const KFPose& pose = it->second;
        Eigen::Quaterniond q(pose.R);
        q.normalize();

        ofs << k        << ","
            << tcam[k]  << ","
            << pose.p.x() << ","
            << pose.p.y() << ","
            << pose.p.z() << ","
            << q.x()      << ","
            << q.y()      << ","
            << q.z()      << ","
            << q.w()      << "\n";
    }

    std::cout << "[CSV] Trayectoria guardada en: " << filename << "\n";
}


double rmse(const Eigen::Vector3d &est, const Eigen::Vector3d &gt) {
    Eigen::Vector3d diff = est - gt;
    return std::sqrt(diff.squaredNorm() / 3.0);  // 3 componentes
}

double rmse_ang(const Eigen::Vector3d &est, const Eigen::Vector3d &gt) {
    double dot = est.dot(gt);
    double norm_product = est.norm() * gt.norm();
    double cosine = std::max(-1.0, std::min(1.0, dot / norm_product));
    double angle_rad = std::acos(cosine);  // en radianes
    double angle_deg = angle_rad * 180.0 / M_PI; //en grados
    return angle_deg;
}

double median(std::vector<double> v){
    if(v.empty()) return 0.0;

    const size_t n = v.size();
    const size_t mid = n / 2;

    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double m = v[mid];

    if((n % 2) == 0){
        auto max_lower = std::max_element(v.begin(), v.begin() + mid);
        m = 0.5 * (m + *max_lower);
    }
    return m;

}

std::string make_csv_basename(const std::string& seq_name, int ref_idx) {
    auto now = std::chrono::system_clock::now();
    std::time_t t  = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << "full_" << seq_name
        //<< "_ref" << ref_idx << "_"
        << "_" 
        << (tm.tm_year + 1900)
        << std::setw(2) << std::setfill('0') << tm.tm_mon + 1
        << std::setw(2) << std::setfill('0') << tm.tm_mday << "_"
        << std::setw(2) << std::setfill('0') << tm.tm_hour
        << std::setw(2) << std::setfill('0') << tm.tm_min
        << std::setw(2) << std::setfill('0') << tm.tm_sec
        << ".csv";
    return oss.str();
}

std::string make_results_path(const std::string& seq_name, int ref_idx) {
    fs::path dir = fs::path("results");          // carpeta destino
    fs::create_directories(dir);                 // idempotente
    fs::path file = make_csv_basename(seq_name, ref_idx);
    return (dir / file).string();
}

std::string make_project_results_path(const std::string& seq_input_path,
                                             const std::string& seq_name,
                                             int ref_idx)
{
    fs::path seq_path(seq_input_path);

    if (seq_path.filename().empty()) seq_path = seq_path.parent_path();

    fs::path data_dir   = seq_path.parent_path();      // .../data
    fs::path project_root = data_dir.parent_path();    // .../Vi-init-cpp

    fs::path outdir = project_root / "results";
    std::error_code ec;
    fs::create_directories(outdir, ec); // idempotente

    fs::path file = make_csv_basename(seq_name, ref_idx);
    return (outdir / file).string();
}

inline Eigen::Matrix3d quatToRot(const Eigen::Quaterniond& q_in) {
    if (!std::isfinite(q_in.w()) || !std::isfinite(q_in.x()) ||
        !std::isfinite(q_in.y()) || !std::isfinite(q_in.z())) {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Quaterniond q = q_in;
    if (q.norm() == 0.0) return Eigen::Matrix3d::Identity();
    q.normalize(); 
    return q.toRotationMatrix();
}

double mean(std::vector<double> v){
    if (v.empty()) throw std::runtime_error("mean() of empty vector");
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}



Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& wdt){
  double th = wdt.norm();
  if (th < 1e-12) return Eigen::Matrix3d::Identity();
  Eigen::Vector3d a = wdt / th;
  Eigen::Matrix3d K;
  K <<     0,-a.z(), a.y(),
        a.z(),    0,-a.x(),
       -a.y(), a.x(),    0;
  return Eigen::Matrix3d::Identity() + std::sin(th)*K + (1-std::cos(th))*(K*K);
}



Eigen::Matrix3d quatToRotSafe(const Eigen::Quaterniond& q_in){
    Eigen::Quaterniond q = q_in;
    if (q.norm() == 0.0) return Eigen::Matrix3d::Identity();
    return q.normalized().toRotationMatrix();
}


double rotCompParallaxMedianPx(const std::vector<cv::Point2f>& pts1,
                               const std::vector<cv::Point2f>& pts2,
                               const Eigen::Matrix3d& K,
                               const Eigen::Matrix3d& R_k_k1)
{
    // Construir H = K * R * K^{-1}
    Eigen::Matrix3d H = K * R_k_k1 * K.inverse();

    cv::Mat Hcv(3,3,CV_64F);
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) Hcv.at<double>(r,c) = H(r,c);

    std::vector<cv::Point2f> pts1_warp;
    cv::perspectiveTransform(pts1, pts1_warp, Hcv);

    std::vector<double> resid;
    resid.reserve(std::min(pts1.size(), pts2.size()));
    for(size_t i=0;i<pts1_warp.size() && i<pts2.size();++i){
        cv::Point2f d = pts2[i] - pts1_warp[i];
        resid.push_back(std::hypot(d.x, d.y));
    }
    if(resid.empty()) return 0.0;

    // Mediana
    std::nth_element(resid.begin(), resid.begin()+resid.size()/2, resid.end());
    return resid[resid.size()/2];
}


std::string infer_sequence_name(const std::string& input_path) {
    namespace fs = std::filesystem;
    fs::path p(input_path);
    if (p.filename().empty()) {         // path con barra final
        p = p.parent_path();
    }
    return p.filename().string();
}




std::optional<Eigen::Matrix3d> compute_Rij_with_c2p(
    const std::string& img_path_i,
    const std::string& img_path_j,
    const Eigen::Matrix3d& K_eigen,
    const cv::Mat& K_cv,
    const cv::Mat& dist_cv)
{
    std::vector<cv::Point2f> pts_i, pts_j;
    cv::Mat img_i, img_j;

    if (!detect_and_track_features(img_path_i, img_path_j, pts_i, pts_j, img_i, img_j))
        return std::nullopt;
    if (pts_i.size() < 8) return std::nullopt;

    cv::Mat und_i, und_j, mask;
    cv::undistortPoints(pts_i, und_i, K_cv, dist_cv, cv::noArray(), K_cv);
    cv::undistortPoints(pts_j, und_j, K_cv, dist_cv, cv::noArray(), K_cv);

    int N = und_i.rows;
    Eigen::MatrixXd f0(3, N), f1(3, N);
    Eigen::Matrix3d Kinv = K_eigen.inverse();

    for (int i = 0; i < N; ++i) {
        double x0 = und_i.at<cv::Point2f>(i).x;
        double y0 = und_i.at<cv::Point2f>(i).y;
        double x1 = und_j.at<cv::Point2f>(i).x;
        double y1 = und_j.at<cv::Point2f>(i).y;

        Eigen::Vector3d p0 = Kinv * Eigen::Vector3d(x0, y0, 1.0);
        Eigen::Vector3d p1 = Kinv * Eigen::Vector3d(x1, y1, 1.0);

        f0.col(i) = p0.normalized();
        f1.col(i) = p1.normalized();
    }

    und_i.convertTo(und_i, CV_64F);
    und_j.convertTo(und_j, CV_64F);
    cv::theRNG().state = 42;

    // Recalcular mask con Essential Matrix
    cv::Mat E = cv::findEssentialMat(und_i, und_j, K_cv, cv::RANSAC, 0.999, 1.0, mask);

    std::vector<Eigen::Vector3d> f0_inliers, f1_inliers;
    for (int i = 0; i < N; ++i) {
        double n0 = f0.col(i).norm();
        double n1 = f1.col(i).norm();
        if (mask.at<uchar>(i, 0) && std::isfinite(n0) && std::isfinite(n1) && n0 > 1e-6 && n1 > 1e-6) {
            f0_inliers.push_back(f0.col(i).normalized());
            f1_inliers.push_back(f1.col(i).normalized());
        }
    }
    
    int num_inliers = static_cast<int>(f0_inliers.size());
    int num_outliers = N - num_inliers;

    if (f0_inliers.size() < 8) return std::nullopt;

    C2PResult result = run_c2p(f0_inliers, f1_inliers, 0);
    return result.R;
}
