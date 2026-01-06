#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <unordered_map>
#include <vector>
#include "track_manager.h"  // para TrackManager, PointTrack
#include <ceres/ceres.h>
#include <ceres/jet.h>

using NodeId = int;

// Nodo de deformación 
struct NodeRest {
    Eigen::Vector3d X_ref;      // posición de referencia
};

struct KFPose {
    Eigen::Matrix3d R;  // R_WC
    Eigen::Vector3d p;  // p_WC
};

// Arista viscosa entre dos nodos
struct ViscousEdge {
    NodeId i, j;
    double b_ij;    // peso viscoso
};

// Hash para pares (i,j) de nodos (para mapa de longitudes de referencia)
struct PairHash2 {
    size_t operator()(const std::pair<NodeId,NodeId>& p) const noexcept {
        return (static_cast<size_t>(p.first) << 32) ^ static_cast<size_t>(p.second);
    }
};

using NodePositions = std::unordered_map<NodeId, Eigen::Vector3d>;
extern std::map<int, NodePositions> g_nodes_per_kf;
extern std::unordered_map<int, KFPose> g_cam_poses;



// Mapas globales (definidos en deformation_graph.cpp)
extern std::unordered_map<NodeId, NodeRest> g_nodes_rest;
extern std::vector<ViscousEdge> g_viscous_edges;
extern std::unordered_map<std::pair<NodeId,NodeId>, double, PairHash2> g_rest_length;


// Dibuja el grafo en reposo (X_ref, verde) y optimizado (X_B, rojo) sobre una imagen
cv::Mat drawRestAndOptimizedGraph(
    const cv::Mat& img,
    const Eigen::Matrix3d& K,
    const std::unordered_map<NodeId, NodeRest>& g_nodes_rest,
    const std::unordered_map<NodeId, Eigen::Vector3d>& X_B_map, // X_B de cada nodo
    const std::vector<ViscousEdge>& g_viscous_edges);

// funcion para exportar video
void exportDeformationVideo(
    const std::vector<int>& kf_indices_in_order,
    const std::vector<std::string>& cam0_image_names,
    const Eigen::Matrix3d& K,
    const std::unordered_map<NodeId, NodeRest>& g_nodes_rest,
    const std::map<int, NodePositions>& g_nodes_per_kf,
    const std::vector<ViscousEdge>& g_viscous_edges,
    const std::string& out_path);

// Construye nodos y grafo de vecinos a partir de tracks
void buildDeformationNodesFromTracks(
    TrackManager& track_manager,
    int ref_frame_idx,
    const Eigen::Matrix3d& K,
    const std::vector<std::string>& depth_image_names,
    double depth_scale);

// Dibuja el grafo de deformación sobre una imagen
cv::Mat drawDeformationGraphOnFrame(
    const cv::Mat& img,
    int frame_idx,
    const TrackManager& track_manager,
    const std::unordered_map<NodeId, NodeRest>& g_nodes_rest,
    const std::vector<ViscousEdge>& g_viscous_edges);

// Costos Ceres 
struct ViscousCost {
    explicit ViscousCost(double weight)
        : sqrt_w_(std::sqrt(weight)) {}

    template <typename T>
    bool operator()(const T* const Xi_B,  // X_i^t
                    const T* const Xi_A,  // X_i^{t-1}
                    const T* const Xj_B,  // X_j^t
                    const T* const Xj_A,  // X_j^{t-1}
                    T* residuals) const
    {
        for (int k = 0; k < 3; ++k) {
            T si = Xi_B[k] - Xi_A[k];
            T sj = Xj_B[k] - Xj_A[k];
            residuals[k] = T(sqrt_w_) * (si - sj);
        }
        return true;
    }

    double sqrt_w_;
};

struct ElasticCost {
    ElasticCost(double d0, double k)
        : d0_(d0),
          inv_sqrt_d0_(1.0 / std::sqrt(d0)),
          sqrt_k_(std::sqrt(k)) {}

    template <typename T>
    bool operator()(const T* const Xi,  // X_i^t
                    const T* const Xj,  // X_j^t
                    T* residual) const
    {
        T dx0 = Xi[0] - Xj[0];
        T dx1 = Xi[1] - Xj[1];
        T dx2 = Xi[2] - Xj[2];

        // std::sqrt funciona bien con ceres::Jet
        T dij = sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2);
        residual[0] = T(sqrt_k_) * (dij - T(d0_)) * T(inv_sqrt_d0_);
        return true;
    }

    double d0_;
    double inv_sqrt_d0_;
    double sqrt_k_;
};


// Helper general para extraer la parte escalar
template <typename T>
inline double extract_scalar(const T& x) {
    return static_cast<double>(x);
}

template <int N>
inline double extract_scalar(const ceres::Jet<double, N>& x) {
    return x.a;  // parte escalar del Jet
}


// Bilineal en double, para imagen A (no depende de autodiff)
inline double bilinearSampleDouble(const cv::Mat& img, double u, double v)
{
    if (img.empty()) return 0.0;

    int w = img.cols;
    int h = img.rows;

    int x0 = static_cast<int>(std::floor(u));
    int y0 = static_cast<int>(std::floor(v));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = std::max(0, std::min(x0, w - 1));
    x1 = std::max(0, std::min(x1, w - 1));
    y0 = std::max(0, std::min(y0, h - 1));
    y1 = std::max(0, std::min(y1, h - 1));

    double I00 = static_cast<double>(img.at<uint8_t>(y0, x0));
    double I10 = static_cast<double>(img.at<uint8_t>(y0, x1));
    double I01 = static_cast<double>(img.at<uint8_t>(y1, x0));
    double I11 = static_cast<double>(img.at<uint8_t>(y1, x1));

    double fx = u - x0;
    double fy = v - y0;

    double w00 = (1.0 - fx) * (1.0 - fy);
    double w10 = fx         * (1.0 - fy);
    double w01 = (1.0 - fx) * fy;
    double w11 = fx         * fy;

    return w00*I00 + w10*I10 + w01*I01 + w11*I11;
}

// Bilineal templado, para imagen B (autodiff)
template <typename T>
inline T bilinearSampleTemplate(const cv::Mat& img, const T& u, const T& v)
{
    if (img.empty()) return T(0);

    int w = img.cols;
    int h = img.rows;

    double u_d = extract_scalar(u);
    double v_d = extract_scalar(v);

    int x0 = static_cast<int>(std::floor(u_d));
    int y0 = static_cast<int>(std::floor(v_d));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = std::max(0, std::min(x0, w - 1));
    x1 = std::max(0, std::min(x1, w - 1));
    y0 = std::max(0, std::min(y0, h - 1));
    y1 = std::max(0, std::min(y1, h - 1));

    double I00 = static_cast<double>(img.at<uint8_t>(y0, x0));
    double I10 = static_cast<double>(img.at<uint8_t>(y0, x1));
    double I01 = static_cast<double>(img.at<uint8_t>(y1, x0));
    double I11 = static_cast<double>(img.at<uint8_t>(y1, x1));

    T fx = u - T(static_cast<double>(x0));
    T fy = v - T(static_cast<double>(y0));
    T one = T(1.0);

    T w00 = (one - fx) * (one - fy);
    T w10 = fx         * (one - fy);
    T w01 = (one - fx) * fy;
    T w11 = fx         * fy;

    return w00*T(I00) + w10*T(I10) + w01*T(I01) + w11*T(I11);
}


// ---- PhotometricCost con parche 3x3 (9 residuos) ----
struct PhotometricCost {
    static constexpr int kRadius = 1;     // 3x3
    static constexpr int kPatchSize = (2 * kRadius + 1) * (2 * kRadius + 1);

    PhotometricCost(const cv::Mat& imgA,
                    const cv::Mat& imgB,
                    const Eigen::Vector2d& uvA_center,
                    const Eigen::Matrix3d& K)
        : imgA_(imgA)
        , imgB_(imgB)
        , uvA_center_(uvA_center)
        , K_(K)
    {}

    template <typename T>
    bool operator()(const T* const X_B, T* residuals) const
    {
        // 1) proyectar X_B en B
        T X = X_B[0];
        T Y = X_B[1];
        T Z = X_B[2];

        T uB = T(K_(0,0)) * X / Z + T(K_(0,2));
        T vB = T(K_(1,1)) * Y / Z + T(K_(1,2));

        // 2) recorro el parche (3x3) alrededor de uvA_center_
        int idx = 0;
        for (int dy = -kRadius; dy <= kRadius; ++dy) {
            for (int dx = -kRadius; dx <= kRadius; ++dx, ++idx) {

                double uA = uvA_center_.x() + dx;
                double vA = uvA_center_.y() + dy;

                double IA = bilinearSampleDouble(imgA_, uA, vA);
                T      IB = bilinearSampleTemplate(imgB_,
                                                   uB + T(dx),
                                                   vB + T(dy));

                // opcional: normalizar por 255.0
                residuals[idx] = (IB - T(IA));
            }
        }

        return true;
    }

private:
    cv::Mat imgA_;             // se pasan por copia superficial (ref al buffer)
    cv::Mat imgB_;
    Eigen::Vector2d uvA_center_;
    Eigen::Matrix3d K_;
};
