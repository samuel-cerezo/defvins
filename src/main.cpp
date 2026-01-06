#include "compute_mu_from_uv.h"
#include "utils.h"
#include "bg_small_angle.h"
#include "track_manager.h"
#include "feature_tracker.h"
#include "camera_utils.h"
#include "feature_tracking.h"
#include "frame_processor.h"
#include "mandala_io.h"
#include "c2p_wrapper.h"
#include "lie_utils.h"
#include <filesystem> 
#include <pybind11/embed.h>
#include <opencv2/core/eigen.hpp>
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
#include "sliding_window.h"
#include "imu_preint.h"
#include "factors.h"
#include "ceres_refine.h"
#include "deformation_graph.h"
#include <ceres/ceres.h>
#include "vins_closed_form.h"
#include "viewer.h"
#include "defgraph_3d_viewer.h"
#include <pangolin/pangolin.h>
#include "alignment_utils.h"
#include <fstream>


static inline int nearest_gt_index(const std::vector<double>& tgt, double t_ref) {
    int idx = 0;
    double best = 1e18;
    for (int i = 0; i < (int)tgt.size(); ++i) {
        double d = std::abs(tgt[i] - t_ref);
        if (d < best) { best = d; idx = i; }
    }
    return idx;
}

static inline void get_gt_pose_at(
    double t_ref,
    const std::vector<double>& tgt,
    const std::vector<Eigen::Vector3d>& p_gt,
    const std::vector<Eigen::Quaterniond>& q_gt,
    Eigen::Matrix3d& R_WC_out,
    Eigen::Vector3d& p_WC_out_m)
{
    int idx = nearest_gt_index(tgt, t_ref);

    Eigen::Quaterniond q = q_gt[idx];
    q.normalize();
    R_WC_out = q.toRotationMatrix();

    p_WC_out_m = p_gt[idx];  // mm -> m
}


// metemos aca los modos de configuracion que tendra el sistema
enum class ConfigMode {
    DEFVINS_NO_NR,
    DEFVINS_NO_IMU,
    DEFVINS_FULL
};


// Helper para parsear modo desde argv[2]
ConfigMode parseConfigMode(int argc, char** argv)
{
    // Esperamos: argv[0] = ./defvins
    //           argv[1] = dataset_path
    //           argv[2] = mode_full | mode_no_nr | mode_no_imu
    if (argc < 3) {
        std::cout << "[CONFIG] Uso: " << argv[0]
                  << " <dataset_path> <mode_full|mode_no_nr|mode_no_imu>\n";
        std::cout << "[CONFIG] No se especificó modo, usando FULL.\n";
        return ConfigMode::DEFVINS_FULL;
    }

    std::string mode = argv[2];
    std::cout << "---- Seleccion de modos: " << mode << std::endl;

    if (mode == "mode_no_nr") {
        std::cout << "[MODE] -- Non-Rigid terms are avoided ----" << std::endl;
        return ConfigMode::DEFVINS_NO_NR;
    }
    else if (mode == "mode_no_imu") {
        std::cout << "[MODE] -- IMU terms are avoided ----" << std::endl;
        return ConfigMode::DEFVINS_NO_IMU;
    }
    else if (mode == "mode_full") {
        std::cout << "[MODE] -- Full mode ----" << std::endl;
        return ConfigMode::DEFVINS_FULL;
    }
    else {
        std::cout << "[CONFIG] Invalid mode '" << mode
                  << "', using Full mode.\n";
        return ConfigMode::DEFVINS_FULL;
    }
}




struct Pose {
    Eigen::Quaterniond q; // rotación W←X  (world wrt X)
    Eigen::Vector3d    p; // traslación en W
};



// Caso común: T_WF disponible (flange wrt world)
inline Pose camera_wrt_world(
    const Eigen::Quaterniond& qgt_world_flange, // W←F
    const Eigen::Vector3d&    pgt_world_flange, // en W
    const Eigen::Matrix3d&    R_FC,             // F←C
    const Eigen::Vector3d&    p_FC              // en F
){
    // T_WC = T_WF * T_FC
    const Eigen::Matrix3d R_WF = quatToRotSafe(qgt_world_flange);
    const Eigen::Matrix3d R_WC = R_WF * R_FC;
    const Eigen::Vector3d p_WC = R_WF * p_FC + pgt_world_flange;

    Pose out;
    out.q = Eigen::Quaterniond(R_WC).normalized();
    out.p = p_WC;
    return out;
}

// --- cache para R_c2p (A.k -> B.k) ---
struct PairHash {
  size_t operator()(const std::pair<int,int>& p) const noexcept {
    return (static_cast<size_t>(p.first) << 32) ^ static_cast<size_t>(p.second);
  }
};

struct PoseStamped {
    double t;                 // timestamp [s]
    Eigen::Vector3d p;        // position (m)
    Eigen::Quaterniond q;     // orientation (wxyz)
};

static void write_pose_csv(std::ofstream& f, const PoseStamped& s) {
  f << std::setprecision(18)
    << s.t << ","
    << s.p.x() << "," << s.p.y() << "," << s.p.z() << ","
    << s.q.x() << "," << s.q.y() << "," << s.q.z() << "," << s.q.w()
    << "\n";
}

//static std::unordered_map<std::pair<int,int>, Eigen::Matrix3d, PairHash> Rc2p_cache;



static PreintMeas preintegrate_segment(
    const std::vector<ImuSample>& imu, double ti, double tj,
    const Eigen::Vector3d& bg, const Eigen::Vector3d& ba,
    const Eigen::Vector3d& g)
{
  PreintMeas out;
  out.dR.setIdentity();
  out.dv.setZero();
  out.dp.setZero();
  out.dt = tj - ti;
  if (tj <= ti || imu.size() < 2) return out;

  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();  // rotación acumulada
  Eigen::Vector3d beta  = Eigen::Vector3d::Zero();  // dv preintegrado (sin g)
  Eigen::Vector3d alpha = Eigen::Vector3d::Zero();  // dp preintegrado (sin g)

  for (size_t k = 0; k + 1 < imu.size(); ++k) {
    if (imu[k].t < ti || imu[k+1].t > tj) continue;

    double dt = imu[k+1].t - imu[k].t;
    if (dt <= 0.0) continue;

    Eigen::Vector3d w = imu[k].omega - bg;  // [rad/s]
    Eigen::Vector3d a = imu[k].acc   - ba;  // specific force (sin g)

    // incremento de rotación
    Eigen::Matrix3d dR = ExpSO3(w * dt);
    Eigen::Vector3d a_world = R * a;  // aún sin g

    // integración semi-implícita:
    alpha += beta * dt + 0.5 * a_world * dt * dt;
    beta  += a_world * dt;
    R     = R * dR;
  }

  out.dR = R;
  out.dv = beta;
  out.dp = alpha;
  return out;
}

// ---- Factor IMU preintegrado (prototipo; R como 9 params) ----




template <typename Map, typename Key>
inline auto try_at(Map& m, const Key& k)
    -> typename Map::mapped_type*                  // puntero a mapped_type (no-const)
{
    auto it = m.find(k);
    return (it == m.end()) ? nullptr : &it->second;
}

template <typename Map, typename Key>
inline auto try_at(const Map& m, const Key& k)
    -> const typename Map::mapped_type*            // puntero a mapped_type (const)
{
    auto it = m.find(k);
    return (it == m.end()) ? nullptr : &it->second;
}



struct InitResult {
    int ref_idx; 
    int n_frames;
    double t_frames;            // n_frames*t_cam
    Eigen::Vector3d gravity;   // gravedad
    Eigen::Vector3d velocity;    // velocity
    Eigen::Vector3d ba;        // bias acc
    double rmse_g_deg;
    double rmse_v_ms;
    double rmse_ba_ms2;
    double rmse_bg_rads;
    std::string status;          // "OK" o "Stable"
};

class CsvLogger {
public:
    explicit CsvLogger(const std::string& path, bool write_header_if_new = true)
        : path_(path)
    {
        const bool file_exists = std::filesystem::exists(path_);
        ofs_.open(path_, std::ios::out | std::ios::app);
        if (!ofs_) throw std::runtime_error("No pude abrir CSV: " + path_);
        if (write_header_if_new && !file_exists) writeHeader();
        ofs_ << std::fixed << std::setprecision(6);
    }

    void append(const InitResult& r) {
        ofs_  << r.ref_idx << ","                 
              << r.n_frames << ","
              << r.t_frames << ","
              << r.gravity.x() << "," << r.gravity.y() << "," << r.gravity.z() << ","
              << r.velocity.x()  << "," << r.velocity.y()  << "," << r.velocity.z()  << ","
              << r.ba.x()      << "," << r.ba.y()      << "," << r.ba.z()      << ","
              << r.rmse_g_deg    << ","
              << r.rmse_v_ms     << ","
              << r.rmse_ba_ms2   << ","
              << r.rmse_bg_rads  << ","
              << r.status
              << "\n";
        ofs_.flush();
    }

private:
    void writeHeader() {
        ofs_ << "ref_idx,"         
             << "n_frames,"
             << "t_frames,"
             << "g_w_x,g_w_y,g_w_z,"
             << "v_x,v_y,v_z,"
             << "ba_w_x,ba_w_y,ba_w_z,"
             << "rmse_g_deg,rmse_v_ms,rmse_ba_ms2,rmse_bg_rads,"
             << "status\n";
    }

    std::string path_;
    std::ofstream ofs_;
};



void preprocessAllTracks(
    const std::vector<std::string>& cam0_image_names,
    FeatureTracker& tracker,
    TrackManager& track_manager)
{
    std::cout << "[PREPROCESS] Processing..." << std::endl;
    for (int i = 0; i < cam0_image_names.size(); ++i) {
        cv::Mat image = cv::imread(cam0_image_names[i], cv::IMREAD_GRAYSCALE);
        tracker.processFrame(image, i, track_manager);
    }
    std::cout << "[PREPROCESS] Tracks completed almacenados en TrackManager." << std::endl;
}

std::vector<PointTrack> selectTracksForInit(
    TrackManager& track_manager,
    int ref_frame_idx,
    int n_frames,
    int max_tracks)
{
    // Buscar tracks que aparezcan al menos en X frames después de ref_frame_idx
    int min_obs = 3; // configurable
    auto valid_tracks = track_manager.getTracksWithMinObservations(min_obs);

    std::vector<PointTrack> subset;
    for (const auto& tr : valid_tracks) {
        bool has_ref = tr.hasObservation(ref_frame_idx);
        if (!has_ref) continue;

        bool has_future = false;
        for (int j = 1; j <= n_frames; ++j) {
            if (tr.hasObservation(ref_frame_idx + j)) {
                has_future = true;
                break;
            }
        }
        if (has_future) {
            subset.push_back(tr);
            if ((int)subset.size() >= max_tracks) break;
        }
    }
    return subset;
}


FeatureTracker tracker;
TrackManager track_manager;
using Clock = std::chrono::steady_clock;

namespace py = pybind11;
using namespace Eigen;



bool compute_translation_parallax_score(
        const cv::Mat& image1,
        const cv::Mat& image2,
        const Eigen::Matrix3d& K,
        const Eigen::Matrix3d& R_k_k1,
        double& parallax_median_px,
        std::vector<cv::Point2f>& pts1_out,   // <-- salida
        std::vector<cv::Point2f>& pts2_out)
{
    int max_features=1000;
    // parámetros mínimos
    const int    MIN_INLIERS = 20;
    const double EPS_Z       = 1e-12;

    parallax_median_px = 0.0;
    pts1_out.clear();
    pts2_out.clear();

    // Asegurar escala de grises
    cv::Mat g1, g2;
    if(image1.channels() == 1) g1 = image1; else cv::cvtColor(image1, g1, cv::COLOR_BGR2GRAY);
    if(image2.channels() == 1) g2 = image2; else cv::cvtColor(image2, g2, cv::COLOR_BGR2GRAY);

    auto t0_condition1 = Clock::now();
    // 1) Detect corners en image1
    std::vector<cv::Point2f> pts1;
    cv::goodFeaturesToTrack(g1, pts1, max_features, /*quality*/0.01, /*minDist*/12.0,
                            cv::Mat(), /*blockSize*/3, /*useHarris*/false, /*k*/0.04);
    if (pts1.size() < (size_t)MIN_INLIERS) return false;

    // 2) Lucas–Kanade a image2
    std::vector<cv::Point2f> pts2;
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
        g1, g2, pts1, pts2, status, err,
        cv::Size(21,21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 30, 0.01),
        0, 1e-4);
    auto t1_condition1 = Clock::now();
    double time_cond1 = std::chrono::duration<double, std::milli>(t1_condition1 - t0_condition1).count();
    // 3) Residuales tras compensar la rotación: x2_hat = K * R^T * K^{-1} * x1_h
    std::vector<double> residuals;
    residuals.reserve(pts1.size());
    pts1_out.reserve(pts1.size());
    pts2_out.reserve(pts1.size());

    Eigen::Matrix3d H = K * R_k_k1.transpose() * K.inverse();

    for (size_t i = 0; i < pts1.size(); ++i){

        if(!status[i]) continue;

        // x1_h en homogéneas
        Eigen::Vector3d x1_h(pts1[i].x, pts1[i].y, 1.0);
        Eigen::Vector3d x2_h = H * x1_h;

        if (std::abs(x2_h.z()) < EPS_Z) continue;

        Eigen::Vector2d x2_hat(x2_h.x() / x2_h.z(), x2_h.y() / x2_h.z());  // predicción sin traslación
        Eigen::Vector2d x2_obs(pts2[i].x, pts2[i].y);                      // observación LK

        double r = (x2_obs - x2_hat).norm(); // [px]
        residuals.push_back(r);

        // Guardamos SOLO las correspondencias para las que hemos podido calcular residual
        pts1_out.push_back(pts1[i]);
        pts2_out.push_back(pts2[i]);
    }

    if (residuals.size() < (size_t)MIN_INLIERS) {
        pts1_out.clear();
        pts2_out.clear();
        return false;
    }

    parallax_median_px = median(residuals); // mediana robusta

    return true;
}


////////////////////////////////////////////




/////////////////////////////////////////////////// main ////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    py::scoped_interpreter guard{};

        if (argc < 2) {
        std::cerr << "Use: " << argv[0] << " <path_to_dataset>" << std::endl;
        return -1;
    }
    
    // para guardar trayectorias
    static std::ofstream est_ofs;
    static std::ofstream gt_ofs;
    static bool files_open = false;
    std::string path_to_data = argv[1];
    ConfigMode config = parseConfigMode(argc, argv);
    //  hyperparameters
    double sigma = 0.02;       // en metros, tuning
    double d_max = 5;   //tuning
    double b_ij = std::exp( - (d_max*d_max) / (2.0 * sigma * sigma) );
    double k_elas = 1.0; // tuning global

    // Dataset containers
    std::vector<double> timu, tcam, tdepth, tgt;
    std::vector<Vector3d> omega, acc, bg_gt, ba_gt, v_gt, g_gt, flange_positions_gt, p_gt;
    std::vector<Quaterniond> q_gt, flange_quaternions_gt;
    std::vector<std::string> cam0_image_names, depth_image_names;

    Matrix4d tbodycam, tbodyimu, tflangebody;
    int starting_frame_;


    // Load Mandala dataset
    if (!load_mandala_sequence(path_to_data, timu, omega, acc, tcam, tdepth, cam0_image_names, depth_image_names,
                             tgt, flange_quaternions_gt, bg_gt, ba_gt, v_gt, g_gt, flange_positions_gt, tbodycam, tbodyimu, tflangebody, starting_frame_)) {
        std::cerr << "Error loading Mandala dataset." << std::endl;
        return -1;
    }
    // positions: all in m.

    // here we convert flange_positions to camera_positions
    std::vector<std::pair<Eigen::Quaterniond, Eigen::Vector3d>> poses;
    poses.reserve(flange_positions_gt.size());

    for (size_t i = 0; i < flange_positions_gt.size(); ++i) {
        const auto& position = flange_positions_gt[i];
        const auto& quat     = flange_quaternions_gt[i];
        poses.emplace_back(quat, position);
    }
    p_gt = transformPositionsRight(poses, tflangebody); // camera w.r.t world
    q_gt = transformQuaternionsRight(poses, tflangebody); // camera w.r.t world

    //---------------------------------------------
   
    std::vector<int> optimized_kfs;
    optimized_kfs.reserve(1000);   // opcional

    std::cout << "--- Starting frame: " << starting_frame_ << " ----" << std::endl;

    double delta_Tcam = 0.0;
    if(tcam[2] && tcam[1]){
        delta_Tcam = tcam[2]-tcam[1];
    }
    
    std::cerr << "Dataset loaded." << std::endl;

    // Construir IMU samples una vez
    std::vector<ImuSample> imu_all;
    imu_all.reserve(timu.size());
    for (size_t i = 0; i < timu.size(); ++i) {
        ImuSample s;
        s.t = timu[i];
        s.omega = omega[i];
        s.acc = acc[i];
        imu_all.push_back(s);
    }


    // Load camera calibration
    std::string yaml_file = path_to_data + "/params/camera.yaml";
    //auto [K, distortion_params, image_size] = load_camera_calibration(yaml_file);
    auto calib = load_camera_calibration(yaml_file);
    auto& K = calib.K;
    auto& distortion_params = calib.distortion_params;
    auto& image_size = calib.image_size;
    cv::Mat K_cv; cv::eigen2cv(K, K_cv);
    //std::cout << "K matrix: " << K << std::endl;
    cv::Mat dist_cv(distortion_params);

    preprocessAllTracks(cam0_image_names, tracker, track_manager);  //preproceso todas las imagenes
    
    //    general parameters

    std::string seq_name = infer_sequence_name(argv[1]);
    CsvLogger logger(make_project_results_path(argv[1], seq_name, 0));

    double depth_scale = 0.001;

    //for(int ref_frame_idx = 1; ref_frame_idx < tcam.size()-10; ref_frame_idx += 10){  
    for(int ref_frame_idx = 1; ref_frame_idx < 2; ref_frame_idx += 25){            
        
        buildDeformationNodesFromTracks(track_manager, ref_frame_idx, K, depth_image_names, depth_scale);

        //int ref_frame_idx = 508;    //108

        //std::cout << "/// Processing frame " << ref_frame_idx << std::endl;
        bool stability=false;

        double first_threshold = 45.0;  //45
        double first_threshold_rel = 0.7;  //0.7
        int id_threshold = 0;
        size_t N = 7;  //number of features. 11
        double ratio_new=0;
        int MAX_FRAMES = 25;
        std::map<int, double> deriv_ratio;
        std::map<int, double> times_deriv;
        deriv_ratio[1] = 100.0;
        times_deriv[1] = 0.0;
        int gt_starting_frame_ = 0;

        std::cout << "\n Frame " << ref_frame_idx << " | Detecting translation..." << std::endl;

        double parallax_media_px = 0.0;            
        double parallax_anterior = 0.0;            
        // FIX: mover las ventanas fuera del loop para acumular historia

        std::deque<double> win_parallax, win_rotcomp;
        const int W = 6, M = 3;
        const double ABS_PX = 6.0, SOFT_PX = 4.0, MIN_TX_PX = 1.2, REL_TH = 0.25;


        for (size_t trigg_disp = 0; trigg_disp  <= 20 ; ++trigg_disp){

            id_threshold = (int)trigg_disp;
            int actual_frame = (int)trigg_disp + ref_frame_idx;

            // FIX: blindar límites (evita out-of-range en la última parte del dataset)
            if (ref_frame_idx < 0 || ref_frame_idx >= (int)cam0_image_names.size()) break;
            if (actual_frame + 1 >= (int)cam0_image_names.size()) break;

            const auto& img1_path = cam0_image_names[ref_frame_idx];
            const auto& img2_path = cam0_image_names[actual_frame + 1];

            cv::Mat image_1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
            cv::Mat image_2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);
            if (image_1.empty() || image_2.empty()) continue;

            auto R_k_k1_opt = compute_Rij_with_c2p(
                img1_path,
                img2_path,
                K, K_cv, dist_cv);
            
            if (!R_k_k1_opt) {
                std::cout << "[C2P] without solution (few inliers) in " 
                        << actual_frame << "->" << actual_frame+1 << "\n";
                continue; // try the following pair
            }
            
            const Eigen::Matrix3d& R_k_k1 = *R_k_k1_opt;    //reference to the value
            
        auto t0 = Clock::now();

            std::vector<cv::Point2f> pts1, pts2;
            bool criteria_achieved = compute_translation_parallax_score(
                image_1, image_2, K, R_k_k1, parallax_media_px, pts1, pts2);

            auto t1 = Clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();


                        
            if(!criteria_achieved) continue;
            
            double parallax_rel = std::abs(parallax_media_px - parallax_anterior) /
                                std::max(1e-6, std::abs(parallax_anterior));

            
            
            double parallax_rotcomp = rotCompParallaxMedianPx(pts1, pts2, K, R_k_k1);
            
            win_parallax.push_back(parallax_media_px);
            win_rotcomp.push_back(parallax_rotcomp);
            if((int)win_parallax.size() > W){
                win_parallax.pop_front();
                win_rotcomp.pop_front();
            }
            

            //std::cout << parallax_media_px << std::endl;   

            if(trigg_disp <= 2) { parallax_anterior = parallax_media_px; continue; }

            bool hard_ok = (parallax_media_px >= ABS_PX) && (parallax_rotcomp >= MIN_TX_PX);
            int cnt = 0;
            for(int i=0;i<(int)win_parallax.size(); ++i)
            if(win_parallax[i] >= SOFT_PX && win_rotcomp[i] >= MIN_TX_PX) cnt++;
            bool soft_ok = (cnt >= M);
            bool rel_ok = (parallax_rel >= REL_TH) && (parallax_rotcomp >= MIN_TX_PX);

            if( (hard_ok || soft_ok || rel_ok) ){
            //    std::cout << " - Criteria achieved in " << trigg_disp << " frames." << std::endl;
                break;
            }

            // Actualizar "anterior" al final de la iteración
            parallax_anterior = parallax_media_px;

            //if (parallax_media_px >= first_threshold) {
            //if(((parallax_rel >= first_threshold_rel) && (trigg_disp > 2)) || ((parallax_media_px >= first_threshold) && (trigg_disp > 2))){
            //    std::cout << " - Criterio alcanzado en " << trigg_disp << " frames." << std::endl;
            //    break; // the criteria is achieved
            //}

        }
        ///////////////////////////////////////////////////////////////////////////

        // ---------- estimacion de bias ------------
        auto estimate_bg_once = [&](){
            int j_bg = ref_frame_idx + 1;
            double t1_bg = tcam[ref_frame_idx];
            double t2_bg = tcam[j_bg];

            auto it1 = std::lower_bound(timu.begin(), timu.end(), t1_bg);
            auto it2 = std::upper_bound(timu.begin(), timu.end(), t2_bg);
            if (std::distance(it1, it2) < 2) {
                throw std::runtime_error("Not enough IMU data to estimate bias.");
            }

            std::vector<double> imu_times_seg(it1, it2);
            std::vector<Vector3d> omega_seg_bg, acc_seg_bg;
            {
                size_t idx1 = std::distance(timu.begin(), it1);
                size_t idx2 = std::distance(timu.begin(), it2);
                omega_seg_bg.reserve(idx2-idx1);
                acc_seg_bg.reserve(idx2-idx1);
                for (size_t k = idx1; k < idx2; ++k) {
                    omega_seg_bg.push_back(omega[k]);
                    acc_seg_bg.push_back(acc[k]);
                }
            }

            acc_seg_bg.pop_back();
            omega_seg_bg.pop_back();

            std::vector<double> dts_bg; dts_bg.reserve(imu_times_seg.size()-1);
            for (size_t i = 1; i < imu_times_seg.size(); ++i)
                dts_bg.push_back(imu_times_seg[i] - imu_times_seg[i - 1]);

            Matrix3d Rpreint_bg = Matrix3d::Identity();
            for (size_t j = 0; j < omega_seg_bg.size(); ++j)
                Rpreint_bg *= lie::ExpMap(omega_seg_bg[j] * dts_bg[j]);

            // R_ij de C2P solo i->i+1
            auto Ropt_bg = compute_Rij_with_c2p(
                cam0_image_names[ref_frame_idx],
                cam0_image_names[j_bg],
                K, K_cv, dist_cv);
            if (!Ropt_bg) throw std::runtime_error("[Bias Estimation] Could not compute R_ij i->i+1.");
            Matrix3d R01_bg = *Ropt_bg;

            double deltat_bg = std::accumulate(dts_bg.begin(), dts_bg.end(), 0.0) / double(dts_bg.size());
            return bg_small_angle(omega_seg_bg, deltat_bg, Rpreint_bg, R01_bg, tbodycam, tbodyimu);
        };

        Vector3d bg_est; 
        try { bg_est = estimate_bg_once(); }
        catch (const std::exception& e) { std::cerr << e.what() << std::endl; return -1; }
        std::cout << "[DEBUG] bg_estimation: " << bg_est.transpose() << std::endl;

        // -----------------------------------------

        // buscamos el indice de GT que mas se acerca a la imagen0
        double min_diff = std::numeric_limits<double>::max();
        for (size_t gt_id0 = 0; gt_id0 < tcam.size(); ++gt_id0) {
            double diff = std::abs(tcam[ref_frame_idx] - tgt[gt_id0]);
            if (diff < min_diff) {
                min_diff = diff;
                gt_starting_frame_ = static_cast<int>(gt_id0);
            }
        }

        // -------------- Obtenemos valores Ground Truth -----

        /*
        // GT del flange wrt mundo
        Eigen::Quaterniond q_WF = q_gt[gt_starting_frame_];     // W←F
        Eigen::Vector3d    p_WF = p_gt[gt_starting_frame_];    // en W

        Eigen::Matrix4d T_FC = tflangebody;
        Eigen::Matrix3d R_FC = T_FC.block<3,3>(0,0);
        Eigen::Vector3d p_FC = T_FC.block<3,1>(0,3);

        // Cámara wrt mundo:
        Pose camW_gt = camera_wrt_world(q_WF, p_WF, R_FC, p_FC);
        Eigen::Quaterniond q_WC_gt = camW_gt.q;        
        Eigen::Matrix3d R0_gt = quatToRotSafe(q_WC_gt);
        Eigen::Vector3d p0_gt = camW_gt.p;  //initial camera position
        */

        ////////////////////////////////////////////////
        double t1_gt = tgt[gt_starting_frame_];
        Eigen::Vector3d V0_gt = v_gt[gt_starting_frame_];
        Eigen::Vector3d G_gt = g_gt[gt_starting_frame_];
        Eigen::Vector3d Bg_gt = bg_gt[gt_starting_frame_];
        Eigen::Vector3d Ba_gt = ba_gt[gt_starting_frame_];
        Eigen::Matrix3d R0_gt = q_gt[gt_starting_frame_].normalized().toRotationMatrix();

        std::map<int, double> historico_rmseG, historico_rmseBA, historico_rmseV;
        std::map<int, Vector3d> historicoG, historicoB, historicoV;
        historico_rmseG[id_threshold-1] = 999;
        historico_rmseBA[id_threshold-1] = 999;
        historico_rmseV[id_threshold-1] = 999;
        historico_rmseG[id_threshold] = 999;
        historico_rmseBA[id_threshold] = 999;
        historico_rmseV[id_threshold] = 999;    

        const Vector3d big = Vector3d::Constant(999.0);
        // Asignación en std::map<int, Vector3d>
        if (id_threshold >= 1) {
            historicoG[id_threshold - 1] = big;
            historicoB[id_threshold - 1] = big;
            historicoV[id_threshold - 1] = big;
        }
        historicoG[id_threshold] = big;
        historicoB[id_threshold] = big;
        historicoV[id_threshold] = big;


        /// aca hace el ciclo grande 
        //for (size_t n_frames=id_threshold; n_frames <= MAX_FRAMES; ++n_frames){ // extensive evaluation

        Eigen::Vector3d init_vel = Vector3d::Zero();
        Eigen::Vector3d init_gravity = Vector3d::Zero();
        Eigen::Vector3d init_ba = Vector3d::Zero();
        Eigen::Vector3d init_bg = Vector3d::Zero();

        //////////////// inicialization ///////////////////
        for (size_t n_frames=id_threshold; n_frames < id_threshold+1; ++n_frames){ // initial experiments
            // === cada inicialización con un frame de referencia distinto ===
            int MAX_PUNTOS_CARACTERISTICOS = 200;

            auto valid_tracks = selectTracksForInit(
                track_manager, ref_frame_idx, n_frames, MAX_PUNTOS_CARACTERISTICOS);

            //std::cout << "[INIT] Usando " << valid_tracks.size() 
            //        << " tracks para inicialización desde frame " << ref_frame_idx << std::endl;


            /*
            std::cout << "Processing features in multiple images..." << std::endl;
            for (int i = 0; i < cam0_image_names.size(); ++i) {
                cv::Mat image = cv::imread(cam0_image_names[i], cv::IMREAD_GRAYSCALE);
                tracker.processFrame(image, i, track_manager);
            }

            int MAX_PUNTOS_CARACTERISTICOS = 20;
            auto valid_tracks = track_manager.getTracksWithMinObservations(4, MAX_PUNTOS_CARACTERISTICOS); //para obtener solo aquellos puntos que fueron vistos en al menos X imágenes distintas.

            std::cout << "Numero de puntos caracteristicos: " << valid_tracks.size() << std::endl;
            */

            size_t count_frames = 0;
            //size_t count_points = 0;
            
            std::map<int, Matrix3d> R_ij_map;
            std::map<int, Matrix3d> Gamma_j_map;
            std::map<int, Vector3d> S_j_map;
            std::map<std::pair<int, int>, Vector3d> mu_ij_map;
            std::map<int, double> tj_map;

            R_ij_map.clear();
            Gamma_j_map.clear();
            S_j_map.clear();
            tj_map.clear();

            // Columna de referencia (key=1)
            R_ij_map[1]    = Matrix3d::Identity();
            Gamma_j_map[1] = Matrix3d::Identity();
            S_j_map[1]     = Vector3d::Zero();
            tj_map[1]      = 0.0;
            //std::cout << "[DEBUG] 1" << std::endl;
            
            // --------- precomputo 
            std::vector<int> keys_disponibles; // guardamos qué columnas j+1 existen realmente
            for (int rel_j = 1; rel_j <= (int)n_frames; ++rel_j) {
                int abs_j = ref_frame_idx + rel_j;
                if (abs_j >= (int)tcam.size()) break;

                const double t1 = tcam[ref_frame_idx];
                const double t2 = tcam[abs_j];
                if (t2 <= t1) continue;

                // Extraer segmento IMU en [t1, t2]
                std::vector<Vector3d> omega_seg, acc_seg;
                std::vector<double> dts;
                for (size_t k = 1; k < timu.size(); ++k) {
                    if (timu[k-1] >= t1 && timu[k] <= t2) {
                        const double dt = timu[k] - timu[k-1];
                        if (dt <= 0) continue;
                        omega_seg.push_back(omega[k-1]);
                        acc_seg.push_back(acc[k-1]);
                        dts.push_back(dt);
                    }
                }
                if (dts.empty()) continue;

                Matrix3d R_ij = Matrix3d::Identity();
                Matrix3d Gamma = Matrix3d::Zero();
                Vector3d S = Vector3d::Zero();
                double tau_k = t1;

                for (size_t k = 0; k < dts.size(); ++k) {
                    const double dt = dts[k];
                    tau_k += dt;
                    const double w = t2 - tau_k;
                    if (w < 0) continue;
                    R_ij *= lie::ExpMap((omega_seg[k] - bg_est) * dt);
                    Gamma += w * R_ij * dt;
                    S     += w * R_ij * acc_seg[k] * dt;
                }

                const int key = rel_j + 1;  // columna 2..n
                R_ij_map[key]    = R_ij;
                Gamma_j_map[key] = Gamma;
                S_j_map[key]     = S;
                tj_map[key]      = (t2 - t1);
                keys_disponibles.push_back(key);
            }

            // ------ recorremos tracks (corregido) ------
            size_t count_points = 0;

            for (const auto& track : valid_tracks) {
                if (!track.hasObservation(ref_frame_idx)) continue;
                ++count_points;
                if (count_points > N) { break; }

                int i = (int)count_points; // id compacto del punto
                size_t usados = 0;

                // IMPORTANTE: obs.frame_idx es ABSOLUTO
                for (const auto& obs : track.observations) {
                    int abs_j = obs.frame_idx;
                    int rel_j = abs_j - ref_frame_idx; // relativo para nuestros mapas
                    if (rel_j < 0 || rel_j > (int)n_frames) continue;

                    // j == 0 → columna referencia
                    if (rel_j == 0) {
                        Vector3d mu_1 = computeMuFromUV(obs.uv, K);
                        if (mu_1.norm() < 1e-5) continue;
                        mu_ij_map[{i, 1}] = mu_1;
                        continue;
                    }

                    // Usar R_ij precomputado (key = rel_j+1)
                    int key = rel_j + 1;
                    auto itR = R_ij_map.find(key);
                    if (itR == R_ij_map.end()) {
                        // no tenemos integración válida para este j
                        continue;
                    }

                    // NO consultar el track con 'j' relativo; ya tenemos la obs actual
                    Vector3d mu_j = computeMuFromUV(obs.uv, K);
                    if (mu_j.norm() < 1e-5) continue;

                    Vector3d mu_ij = itR->second * mu_j;
                    mu_ij_map[{i, key}] = mu_ij;

                    // límite por frames efectivos por punto (opcional)
                    //if (++usados >= n_frames) break;
                }
            }

            // -------- fin recorrido tracks (corregido) ----

            // Construir el vector de tiempos t (t[0] = 0.0; luego las diferencias):
            std::sort(keys_disponibles.begin(), keys_disponibles.end());
            keys_disponibles.erase(
                std::unique(keys_disponibles.begin(), keys_disponibles.end()),
                keys_disponibles.end()
            );

            std::vector<double> t_vals;
            t_vals.reserve(keys_disponibles.size());
            t_vals.push_back(0.0); // t0

            for (int key : keys_disponibles) {
                // key = rel_j + 1; queremos Δt_j para cada columna existente
                auto it = tj_map.find(key);
                if (it == tj_map.end()) continue;
                t_vals.push_back(it->second);
            }

            // 1) Columnas efectivas: presentes en TODOS los mapas y con al menos una mu
            std::vector<int> eff_cols;
            eff_cols.reserve(Gamma_j_map.size());

            // ¿tiene al menos una mu para esta columna?
            auto has_mu_for_col = [&](int col)->bool{
                for (const auto& kv : mu_ij_map){
                    if (kv.first.second == col) return true; // {i, col}
                }
                return false;
            };

            // Candidatas: usa las keys de Gamma_j_map (podrías usar S_j_map o R_ij_map igualmente)
            for (const auto& kv : Gamma_j_map){
                int col = kv.first;
                if (R_ij_map.find(col)==R_ij_map.end()) continue;
                if (S_j_map.find(col)==S_j_map.end())   continue;
                if (tj_map.find(col)==tj_map.end())     continue;
                if (!has_mu_for_col(col))               continue;
                eff_cols.push_back(col);
            }

            // Asegura que esté la columna de referencia (1) si existe en los mapas
            if (std::find(eff_cols.begin(), eff_cols.end(), 1) == eff_cols.end()){
                if (Gamma_j_map.count(1) && S_j_map.count(1) && R_ij_map.count(1) && tj_map.count(1) && has_mu_for_col(1))
                    eff_cols.insert(eff_cols.begin(), 1);
            }

            // Ordenar por si vinieron desordenadas
            std::sort(eff_cols.begin(), eff_cols.end());

            // Si hay muy pocas columnas efectivas, sal con gracia
            if (eff_cols.size() < 2){
                std::cerr << "[WARN] Muy pocas columnas efectivas (size=" << eff_cols.size() << "). Saltando esta init.\n";
                continue;
            }

            // 2) Reindexa a 1..C (col_real -> col_compacta)
            std::unordered_map<int,int> col_index;
            col_index.reserve(eff_cols.size());
            int C = 0;
            for(int col_real : eff_cols) col_index[col_real] = ++C;

            // 3) Construye mapas compactos y vector de tiempos alineado
            std::map<int, Eigen::Matrix3d> Gamma_comp;
            std::map<int, Eigen::Vector3d> S_comp;
            std::map<std::pair<int,int>, Eigen::Vector3d> mu_comp;
            std::vector<double> t_vals_comp;  // tamaño = C, con t_vals_comp[0] = 0.0

            Gamma_comp.clear(); S_comp.clear(); mu_comp.clear();
            t_vals_comp.assign(C, 0.0); // t(1)=0.0 para la columna 1 compacta

            for(int col_real : eff_cols){
                int jc = col_index[col_real];

                // Copia segura (ya sabemos que existen por haber filtrado eff_cols)
                Gamma_comp[jc] = *try_at(Gamma_j_map, col_real);
                S_comp[jc]     = *try_at(S_j_map,     col_real);

                // tiempos: compacta; jc=1 (ref) -> 0.0; jc>=2 -> tj_map[col_real]
                if (jc >= 2) {
                    auto* tptr = try_at(tj_map, col_real);
                    t_vals_comp[jc-1] = tptr ? *tptr : 0.0; // por seguridad, aunque debería existir
                } else {
                    t_vals_comp[0] = 0.0;
                }
            }

            // Remapea mu_ij: (i, col_real) -> (i, jc)
            for(const auto& kv : mu_ij_map){
                int i_point  = kv.first.first;
                int col_real = kv.first.second;
                auto itc = col_index.find(col_real);
                if(itc == col_index.end()) continue; // mu de columna descartada
                int jc = itc->second;
                mu_comp[{i_point, jc}] = kv.second;
            }

            // ================== /COMPACTACIÓN ==================
            // ================== DENSIFICAR PUNTOS (i) ==================
            // Requiere: eff_cols, col_index, mu_comp, Gamma_comp, S_comp, t_vals_comp

            // 1) Elige puntos que tengan mu en TODAS las columnas compactas 1..C
            std::vector<int> dense_points_orig; // índices de punto 'i' originales
            dense_points_orig.reserve(N);

            C = (int)Gamma_comp.size();               // columnas compactas (incluye ref=1)
            auto has_all_cols = [&](int i_orig)->bool{
                for (int jc = 1; jc <= C; ++jc){
                    if (mu_comp.find({i_orig, jc}) == mu_comp.end()) return false;
                }
                return true;
            };

            // Nota: 'count_points' es cuántos puntos llegaste a cargar (1..count_points)
            for (int i_orig = 1; i_orig <= (int)count_points; ++i_orig){
                if (has_all_cols(i_orig)) {
                    dense_points_orig.push_back(i_orig);
                    if ((int)dense_points_orig.size() >= (int)N) break; // usa hasta N densos
                }
            }

            if (dense_points_orig.size() < 3){
                //std::cerr << "[WARN] Demasiado pocos puntos densos ("
                //        << dense_points_orig.size() << "). Saltando esta init.\n";
                continue;
            }

            // 2) Reindexa puntos densos a 1..N_eff y reconstruye mu_comp_dense
            std::map<std::pair<int,int>, Eigen::Vector3d> mu_comp_dense;
            mu_comp_dense.clear();

            int N_eff = (int)dense_points_orig.size();
            int i_new = 0;
            for (int i_orig : dense_points_orig){
                ++i_new; // 1..N_eff
                for (int jc = 1; jc <= C; ++jc){
                    auto it = mu_comp.find({i_orig, jc});
                    // Debe existir por construcción:
                    if (it == mu_comp.end()){
                        // defensa extra: si no está, aborta esta init limpia
                        std::cerr << "[WARN] Faltó mu para i=" << i_orig << " col=" << jc
                                << " tras comprobar densidad. Skip.\n";
                        i_new = 0; // marca invalidez
                        break;
                    }
                    mu_comp_dense[{i_new, jc}] = it->second;
                }
                if (i_new == 0) break; // abortó por falta inesperada
            }
            if (i_new == 0) continue;

            VISolution sol;

            /////////////////
            double gravity_mag = 9.81;
            // 1) Intentar warm-start con tu solver cerrado
            VISolution sol_closed;
            bool have_closed = solveVINSWithBias(
                Gamma_comp, S_comp, mu_comp_dense, t_vals_comp, gravity_mag, sol_closed);

            // 2) Si el cerrado falla, arma un warm-start seguro por defecto
            VISolution warm_default;
            warm_default.gravity  = Eigen::Vector3d(0, -gravity_mag, 0); // ajusta signo/eje según tu convención
            warm_default.velocity = Eigen::Vector3d::Zero();
            warm_default.bias     = Eigen::Vector3d::Zero();

            // 3) Llamar SIEMPRE a Ceres con warm-start (cerrado si existe; si no, default)
            const VISolution* warm_ptr = have_closed ? &sol_closed : &warm_default;

            VISolution sol_ceres;
            bool ok_ceres = defvins::SolveWithCeres(
                Gamma_comp, S_comp, mu_comp_dense, t_vals_comp,
                /*N_eff=*/N_eff, /*C=*/C,
                gravity_mag,
                warm_ptr,
                sol_ceres
            );

            // 4) Elegir solución final (si Ceres falla, cae al cerrado si estaba disponible)
            VISolution sol_final;
            if (ok_ceres) {
                sol_final = sol_ceres;
            } else if (have_closed) {
                std::cerr << "[Ceres] Optimization failed. Using closed-form result.\n";
                sol_final = sol_closed;
            } else {
                std::cerr << "[Init] Both Ceres and closed-form failed. Using default seed.\n";
                sol_final = warm_default;
            }

            // 5) Usar sol_final como antes (comparar contra GT, logging, etc.)
            Eigen::Vector3d gravity = R0_gt * sol_final.gravity;
            Eigen::Vector3d bias_acc = R0_gt * sol_final.bias;

            double rmse_gravity = rmse_ang(gravity, G_gt);
            double rmse_velocity = rmse(sol_final.velocity, V0_gt);
            double rmse_bias_a   = rmse(bias_acc, Ba_gt);
            double rmse_bias_g   = rmse(bg_est, Bg_gt);

            // Historiales (conservar tu lógica existente)
            historico_rmseG[n_frames]  = rmse_gravity;
            historico_rmseBA[n_frames] = rmse_bias_a;
            historico_rmseV[n_frames]  = rmse_velocity;

            historicoG[n_frames] = gravity;
            historicoB[n_frames] = bias_acc;
            historicoV[n_frames] = sol_final.velocity;

            std::cout << "[INIT] n_frames=" << n_frames
                    << "  |G|=" << sol_final.gravity.norm()
                    << "  rmse_g[deg]=" << rmse_gravity
                    << "  rmse_v[m/s]=" << rmse_velocity
                    << "  rmse_ba[m/s^2]=" << rmse_bias_a
                    << "  rmse_bg[rad/s]=" << rmse_bias_g << "\n";

            std::cout << "       G: " << sol_final.gravity.transpose()
                    << "  V: "     << sol_final.velocity.transpose()
                    << "  Ba: "    << sol_final.bias.transpose() << "\n";
        
            init_vel = sol_final.velocity;
            //init_gravity = sol_final.gravity;
            init_gravity = gravity;
            init_ba = sol_final.bias;
            init_bg = bg_est;
        
        }

        
        std::cout << "-------- Optimizacion de KFs ---------" << std::endl;
            
        // ====== VENTANA (prototipo 10 KFs) ======
        static GlobalParams gp;              // g, bg, ba globales
        static bool gp_inited = false;
        static int WINDOWS_SIZE = 5;    // numero de KF en la ventana
        //std::cout << "\n --- ID th: " << id_threshold << std::endl;
        static int INIT_FRAME = 1;
        const int first_kf = id_threshold + INIT_FRAME;
        //const int last_kf  = static_cast<int>(cam0_image_names.size()) - 1;
        const int last_kf = first_kf + 4000;


        // sanity check opcional:
        if (first_kf + WINDOWS_SIZE >= last_kf) {
            std::cerr << "[WARN] Muy pocos frames para esta ventana.\n";
        }

        std::deque<KFState> win;
        std::vector<int> optimized_kfs;
        g_cam_poses.clear();

        ///// cargamos gt
        if (!files_open) {
            std::string csv_eval_filename;
            std::string csv_gt_filename;
            switch (config) {
                case ConfigMode::DEFVINS_NO_NR:
                    csv_eval_filename = seq_name + "_noNR.csv";
                    break;
                case ConfigMode::DEFVINS_NO_IMU:
                    csv_eval_filename = seq_name + "_noIMU.csv";
                    break;
                case ConfigMode::DEFVINS_FULL:
                    default:
                    csv_eval_filename = seq_name + "_full.csv";
                    break;
                }
            
            csv_gt_filename = seq_name + "_gt.csv";
            est_ofs.open(csv_eval_filename);
            gt_ofs.open(csv_gt_filename);

            est_ofs << "timestamp,px,py,pz,qx,qy,qz,qw\n";
            gt_ofs  << "timestamp,px,py,pz,qx,qy,qz,qw\n";

            // dumpear GT completo una vez
            for (size_t i = 0; i < tgt.size(); ++i) {
                Eigen::Vector3d p_m = p_gt[i];  // si GT está en mm
                Eigen::Quaterniond q = q_gt[i]; q.normalize();

                gt_ofs << std::setprecision(18)
                    << tgt[i] << ","
                    << p_m.x() << "," << p_m.y() << "," << p_m.z() << ","
                    << q.x() << "," << q.y() << "," << q.z() << "," << q.w()
                    << "\n";
            }

            files_open = true;
        }

        ////// for (n_frames)

        for (int n_frames = first_kf; n_frames <= last_kf; ++n_frames)
        {
           
            KFState kf;
            kf.k = n_frames;
            kf.t = tcam[n_frames];
            kf.R = Eigen::Matrix3d::Identity();

            // ------------------- Init global params + first KF -------------------
            if (!gp_inited) {
                // --------- parámetros globales ----------
                gp.g  = init_gravity;  
                gp.bg = init_bg;
                gp.ba = init_ba;

                // índice GT más cercano
                const double t_ref = tcam[n_frames];
                int idx_gt_ref = 0;
                double best_ref = 1e9;
                for (int i = 0; i < (int)tgt.size(); ++i) {
                    const double d = std::abs(tgt[i] - t_ref);
                    if (d < best_ref) { best_ref = d; idx_gt_ref = i; }
                }
                        
                Eigen::Quaterniond q_gt_ref = q_gt[idx_gt_ref];
                q_gt_ref.normalize();
                kf.R = q_gt_ref.toRotationMatrix();
                Eigen::Vector3d p_gt_ref = p_gt[idx_gt_ref];

                kf.p = p_gt_ref;
                kf.v = Eigen::Vector3d::Zero();    // velocidad inicial 0 
                kf.v = init_vel;

                gp_inited = true;
                std::cout << "[INIT] KF " << kf.k
                        << " inicializado desde GT. p0 = " << kf.p.transpose() << "\n";
            } else {
                Eigen::Matrix3d R_gt;
                Eigen::Vector3d p_gt_m;
                get_gt_pose_at(tcam[n_frames], tgt, p_gt, q_gt, R_gt, p_gt_m);

                kf.R = R_gt;
                kf.p = p_gt_m;       
                kf.v = init_vel;     
            }

            if (config == ConfigMode::DEFVINS_NO_IMU) {
                static std::mt19937 gen(12345);
                std::normal_distribution<double> dist(0.0, 0.5);
                kf.p = Eigen::Vector3d(dist(gen), dist(gen), dist(gen));
            }
                    

            win.push_back(kf);  
            std::cout << "[DEBUG] Tamaño de ventana: " << win.size() << std::endl;
            
            if (win.size() == WINDOWS_SIZE)
            {
                // ------------------ bucle de ventana -----------------
                for (size_t wind_loc = 0; wind_loc + 1 < win.size(); ++wind_loc)
                {
                    KFState& A = win[wind_loc];
                    KFState& B = win[wind_loc + 1];

                    cv::Mat imgA_mat = cv::imread(cam0_image_names[A.k], cv::IMREAD_GRAYSCALE);
                    cv::Mat imgB_mat = cv::imread(cam0_image_names[B.k], cv::IMREAD_GRAYSCALE);

                    if (imgA_mat.empty() || imgB_mat.empty()) {
                        std::cerr << "[PHOTO] Could not load images for KFs "
                                << A.k << " and " << B.k << "\n";
                        continue;
                    }

                    auto sampleIntensityNN = [](const cv::Mat& img, const cv::Point2f& pt)->double {
                        if (img.empty()) return 0.0;
                        int x = (int)std::lround(pt.x);
                        int y = (int)std::lround(pt.y);
                        x = std::max(0, std::min(x, img.cols - 1));
                        y = std::max(0, std::min(y, img.rows - 1));
                        return (double)img.at<uint8_t>(y, x);
                    };

                    // ------------------ IMU segment (FIX) -----------------
                    std::vector<ImuSample> seg;
                    seg.reserve(512);

                    // incluir muestras con t en [A.t, B.t]
                    for (const auto& s : imu_all) {
                        if (s.t >= A.t && s.t <= B.t) seg.push_back(s);
                    }

                    if (seg.size() < 2) {
                        std::cout << "[SW-IMU] skip pair (" << A.k << "->" << B.k
                                << "): not enough IMU.\n";
                        continue;
                    }

                    PreintMeas meas = preintegrate_segment(seg, A.t, B.t, gp.bg, gp.ba, gp.g);
                    const double dt = meas.dt;

                    // ------------------ C2P cache (evitar shadowing) -----------------
                    struct PairHash {
                        size_t operator()(const std::pair<int,int>& p) const noexcept {
                            return (static_cast<size_t>(p.first) << 32) ^ static_cast<size_t>(p.second);
                        }
                    };
                    static std::unordered_map<std::pair<int,int>, Eigen::Matrix3d, PairHash> Rc2p_cache;

                    Eigen::Matrix3d R_c2p = Eigen::Matrix3d::Identity();
                    bool have_c2p = false;

                    const auto key = std::make_pair(A.k, B.k);
                    if (auto itC = Rc2p_cache.find(key); itC != Rc2p_cache.end()) {
                        R_c2p = itC->second;
                        have_c2p = true;
                    } else {
                        const std::string& imgA_path = cam0_image_names[A.k];
                        const std::string& imgB_path = cam0_image_names[B.k];
                        if (auto Ropt = compute_Rij_with_c2p(imgA_path, imgB_path, K, K_cv, dist_cv)) {
                            R_c2p = *Ropt;
                            Rc2p_cache[key] = R_c2p;
                            have_c2p = true;
                        }
                    }

                    using Quat = Eigen::Quaterniond;
                    Quat qa(A.R);
                    Quat qb;

                    Eigen::Vector3d pa = A.p;
                    Eigen::Vector3d va = A.v;
                    Eigen::Vector3d pb = B.p;
                    Eigen::Vector3d vb = B.v;

                    // ------------------ seed IMU (USARLO) -----------------
                    Eigen::Vector3d p_pred = pa + va * dt + 0.5 * gp.g * dt * dt + A.R * meas.dp;
                    Eigen::Vector3d v_pred = va + gp.g * dt + A.R * meas.dv;

                    // Si NO querés seedear pb/vb desde GT, al menos usa IMU seed:
                    if (config != ConfigMode::DEFVINS_NO_IMU) {
                        pb = p_pred;
                        vb = v_pred;
                    }

                    Eigen::Matrix3d Rb_seed = have_c2p ? (A.R * R_c2p) : (A.R * meas.dR);
                    qb = Quat(Rb_seed);
                    qb.normalize();

                    // ------------------ CERES -----------------
                    ceres::Problem prob;

                    prob.AddParameterBlock(qa.coeffs().data(), 4);
                    prob.AddParameterBlock(qb.coeffs().data(), 4);

                    #if CERES_VERSION_MAJOR >= 2
                        ceres::Manifold* quat_manifold = new ceres::QuaternionManifold();
                        prob.SetManifold(qa.coeffs().data(), quat_manifold);
                        prob.SetManifold(qb.coeffs().data(), quat_manifold);
                    #else
                        ceres::LocalParameterization* quat_param = new ceres::QuaternionParameterization();
                        prob.SetParameterization(qa.coeffs().data(), quat_param);
                        prob.SetParameterization(qb.coeffs().data(), quat_param);
                    #endif

                    prob.AddParameterBlock(pa.data(), 3);
                    prob.AddParameterBlock(pb.data(), 3);
                    prob.AddParameterBlock(va.data(), 3);
                    prob.AddParameterBlock(vb.data(), 3);

                    // Ancla KF A
                    prob.SetParameterBlockConstant(qa.coeffs().data());
                    prob.SetParameterBlockConstant(pa.data());
                    prob.SetParameterBlockConstant(va.data());

                    // (tu bloque node_states + residuos viscoso/elástico + foto + reproj va igual)
                    // IMPORTANTE: usa imgA_mat/imgB_mat al construir PhotometricCost:
                    // auto* photo_cost = new PhotometricCost(imgA_mat, imgB_mat, uvA, K);

                    // Gravity prior + Preint
                    if (config != ConfigMode::DEFVINS_NO_IMU) {
                        // Asegura que gp.g sea parámetro (Ceres la agrega si aparece, pero esto lo deja explícito)
                        prob.AddParameterBlock(gp.g.data(), 3);

                        double w_g = 10.0;
                        ceres::CostFunction* cG =
                            new ceres::AutoDiffCostFunction<GravityPriorCost, 3, 3>(
                                new GravityPriorCost(init_gravity, w_g));
                        prob.AddResidualBlock(cG, nullptr, gp.g.data());

                        auto* cf = new ceres::AutoDiffCostFunction<PreintFactor, 9, 4,3,3, 4,3,3,3>(
                            new PreintFactor(meas.dR, meas.dv, meas.dp, meas.dt, gp.g));
                        ceres::LossFunction* huber = new ceres::HuberLoss(1.0);
                        prob.AddResidualBlock(cf, huber,
                            qa.coeffs().data(), pa.data(), va.data(),
                            qb.coeffs().data(), pb.data(), vb.data(),
                            gp.g.data());
                    }

                    ceres::Solver::Options o;
                    o.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                    o.max_num_iterations = 20;

                    ceres::Solver::Summary sum;
                    ceres::Solve(o, &prob, &sum);

                    // update B
                    B.R = qb.normalized().toRotationMatrix();
                    B.p = pb;
                    B.v = vb;
                    g_cam_poses[B.k] = KFPose{B.R, B.p};

                }   // fin for(wind_loc)

                // escribimos results
                for (const auto& S : win) {
                    Eigen::Quaterniond q(S.R);
                    q.normalize();
                    est_ofs << std::setprecision(18)
                            << S.t << ","
                            << S.p.x() << "," << S.p.y() << "," << S.p.z() << ","
                            << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";
                }
                
                win.clear();
            }
        } // end windows optimization
    }
            ///----------
    if (est_ofs.is_open()) est_ofs.close();
    if (gt_ofs.is_open())  gt_ofs.close();

    return 0;
}

  