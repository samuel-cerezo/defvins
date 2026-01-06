#include "deformation_graph.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/jet.h>


using NodeId = int;

// Posiciones 3D de cada nodo en un frame concreto
using NodePositions = std::unordered_map<NodeId, Eigen::Vector3d>;

// Mapa: kf_index -> posiciones de nodos en ese KF
extern std::map<int, NodePositions> g_nodes_per_kf;
std::map<int, NodePositions> g_nodes_per_kf;
std::unordered_map<int, KFPose> g_cam_poses;


// Definición de los mapas globales
std::unordered_map<NodeId, NodeRest> g_nodes_rest;
std::vector<ViscousEdge> g_viscous_edges;
std::unordered_map<std::pair<NodeId,NodeId>, double, PairHash2> g_rest_length;

static bool sampleDepthAt(
    const cv::Mat& depth,
    const Eigen::Vector2d& uv,
    double depth_scale,
    double& depth_m)
{
    if (depth.empty()) return false;

    int x = static_cast<int>(std::round(uv.x()));
    int y = static_cast<int>(std::round(uv.y()));

    if (x < 0 || x >= depth.cols || y < 0 || y >= depth.rows)
        return false;

    double d = 0.0;

    if (depth.type() == CV_16UC1) {
        uint16_t raw = depth.at<uint16_t>(y, x);
        if (raw == 0) return false;      // suele ser "sin medida"
        d = static_cast<double>(raw) * depth_scale;
    } 
    else if (depth.type() == CV_32FC1) {
        float raw = depth.at<float>(y, x);
        if (!std::isfinite(raw) || raw <= 0.0f) return false;
        d = static_cast<double>(raw) * depth_scale;
    } 
    else {
        // tipo no soportado
        return false;
    }

    if (d <= 0.0) return false;
    depth_m = d;
    return true;
}


// Dibuja el grafo en reposo (X_ref, verde) y optimizado (X_B, rojo) sobre una imagen
cv::Mat drawRestAndOptimizedGraph(
    const cv::Mat& img,
    const Eigen::Matrix3d& K,
    const std::unordered_map<NodeId, NodeRest>& g_nodes_rest,
    const std::unordered_map<NodeId, Eigen::Vector3d>& X_B_map, // X_B de cada nodo
    const std::vector<ViscousEdge>& g_viscous_edges)
{
    cv::Mat out;
    if (img.channels() == 1)
        cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
    else
        img.copyTo(out);

    auto project = [&](const Eigen::Vector3d& X) -> cv::Point2f {
        Eigen::Vector3d x = K * X;
        if (std::abs(x.z()) < 1e-9) return cv::Point2f(-1.f,-1.f);
        return cv::Point2f(
            static_cast<float>(x.x()/x.z()),
            static_cast<float>(x.y()/x.z()));
    };

    // 1) Aristas en REPOSO (X_ref) – verde
    for (const auto& e : g_viscous_edges) {
        auto itI = g_nodes_rest.find(e.i);
        auto itJ = g_nodes_rest.find(e.j);
        if (itI == g_nodes_rest.end() || itJ == g_nodes_rest.end()) continue;

        cv::Point2f pi = project(itI->second.X_ref);
        cv::Point2f pj = project(itJ->second.X_ref);
        if (pi.x < 0 || pj.x < 0) continue;

        cv::line(out, pi, pj, cv::Scalar(0,255,0), 1, cv::LINE_AA); // verde
    }

    // 2) Aristas OPTIMIZADAS (X_B) – rojo
    for (const auto& e : g_viscous_edges) {
        auto itI = X_B_map.find(e.i);
        auto itJ = X_B_map.find(e.j);
        if (itI == X_B_map.end() || itJ == X_B_map.end()) continue;

        cv::Point2f pi = project(itI->second);
        cv::Point2f pj = project(itJ->second);
        if (pi.x < 0 || pj.x < 0) continue;

        cv::line(out, pi, pj, cv::Scalar(0,0,255), 1, cv::LINE_AA); // rojo
    }

    // 3) Opcional: dibujar nodos
    for (const auto& kv : g_nodes_rest) {
        NodeId id = kv.first;
        cv::Point2f p_ref = project(kv.second.X_ref);
        if (p_ref.x < 0) continue;
        cv::circle(out, p_ref, 2, cv::Scalar(0,255,0), -1); // nodo en reposo

        auto itB = X_B_map.find(id);
        if (itB != X_B_map.end()) {
            cv::Point2f p_B = project(itB->second);
            if (p_B.x >= 0) {
                cv::circle(out, p_B, 2, cv::Scalar(0,0,255), -1); // nodo optimizado
            }
        }
    }

    return out;
}

void exportDeformationVideo(
    const std::vector<int>& kf_indices_in_order,
    const std::vector<std::string>& cam0_image_names,
    const Eigen::Matrix3d& K,
    const std::unordered_map<NodeId, NodeRest>& g_nodes_rest,
    const std::map<int, NodePositions>& g_nodes_per_kf,
    const std::vector<ViscousEdge>& g_viscous_edges,
    const std::string& out_path)
{
    if (kf_indices_in_order.empty()) {
        std::cerr << "[VIDEO] No KF indices provided.\n";
        return;
    }

    int first_kf = kf_indices_in_order.front();
    if (first_kf < 0 || first_kf >= (int)cam0_image_names.size()) {
        std::cerr << "[VIDEO] First KF index out of range.\n";
        return;
    }

    cv::Mat first = cv::imread(cam0_image_names[first_kf], cv::IMREAD_GRAYSCALE);
    if (first.empty()) {
        std::cerr << "[VIDEO] Could not read first frame image.\n";
        return;
    }

    cv::Size sz = first.size();
    double fps = 10.0;  // elige lo que quieras para la animación (10–30)

    // Codec: prueba MP4V; si da problemas siempre puedes cambiar a MJPG y .avi
    int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    cv::VideoWriter writer(out_path, fourcc, fps, sz, true);

    if (!writer.isOpened()) {
        std::cerr << "[VIDEO] Could not open VideoWriter for " << out_path << "\n";
        return;
    }

    for (int kf_idx : kf_indices_in_order) {
        if (kf_idx < 0 || kf_idx >= (int)cam0_image_names.size()) continue;

        auto it_nodes = g_nodes_per_kf.find(kf_idx);
        if (it_nodes == g_nodes_per_kf.end()) {
            std::cerr << "[VIDEO] No nodes stored for KF " << kf_idx << ", skipping.\n";
            continue;
        }

        cv::Mat img = cv::imread(cam0_image_names[kf_idx], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "[VIDEO] Could not read image for KF " << kf_idx << "\n";
            continue;
        }

        // Construimos X_B_map (unordered_map) desde NodePositions (map/unordered_map)
        std::unordered_map<NodeId, Eigen::Vector3d> X_B_map;
        X_B_map.reserve(it_nodes->second.size());
        for (const auto& kv : it_nodes->second) {
            X_B_map[kv.first] = kv.second;
        }

        cv::Mat vis = drawRestAndOptimizedGraph(
            img, K, g_nodes_rest, X_B_map, g_viscous_edges);

        writer.write(vis);
    }

    writer.release();
    std::cout << "[VIDEO] Saved deformation video to " << out_path << "\n";
}


// ------------------- Construcción de nodos + grafo ----------------
void buildDeformationNodesFromTracks(
    TrackManager& track_manager,
    int ref_frame_idx,
    const Eigen::Matrix3d& K,
    const std::vector<std::string>& depth_image_names,
    double depth_scale)
{
    g_nodes_rest.clear();
    g_viscous_edges.clear();
    g_rest_length.clear();

    if (ref_frame_idx < 0 ||
        ref_frame_idx >= (int)depth_image_names.size()) {
        std::cerr << "[DEFORM] ref_frame_idx out of range for depth.\n";
        return;
    }

    cv::Mat depth = cv::imread(depth_image_names[ref_frame_idx], cv::IMREAD_UNCHANGED);

    if (depth.empty()) {
        std::cerr << "[DEFORM] depth can not be uploaded"
                  << ref_frame_idx << " (" << depth_image_names[ref_frame_idx] << ")\n";
        return;
    }


    // 1) usamos como referencia el frame ref_frame_idx
    //auto tracks = track_manager.getTracksWithMinObservations(5);
    auto tracks = track_manager.getAllTracks();


    // un nodo por track que tenga observación en ref_frame_idx
    for (const auto& tr : tracks) {
        if (!tr.hasObservation(ref_frame_idx)) continue;
        
        NodeId id = tr.id;

        // Buscar la observación en ref_frame_idx dentro de las observations
        Eigen::Vector2d uv;
        bool found = false;
        for (const auto& obs : tr.observations) {
            if (obs.frame_idx == ref_frame_idx) {
                uv = obs.uv;   
                found = true;
                break;
            }
        }
        if (!found) continue;  // por seguridad

        double d_m = 0.0;
        if (!sampleDepthAt(depth, uv, depth_scale, d_m)) {

            continue;
        }

        Eigen::Vector3d ray = K.inverse() * Eigen::Vector3d(uv.x(), uv.y(), 1.0);
        Eigen::Vector3d x_cam = ray * d_m;

        NodeRest node;
        node.X_ref = x_cam;

        g_nodes_rest[id] = node;

    }

    // 2) construir grafo de vecinos y pesos viscosos
    const double sigma = 0.05;
    const double max_radius = 0.2;   
    const double K_neigh = 6;

    g_viscous_edges.clear();
    g_rest_length.clear();

    std::vector<std::pair<NodeId, Eigen::Vector3d>> tmp;
    tmp.reserve(g_nodes_rest.size());
    for (auto& kv : g_nodes_rest)
        tmp.emplace_back(kv.first, kv.second.X_ref);
    

    for (size_t a = 0; a < tmp.size(); ++a) {
        NodeId id_a = tmp[a].first;
        const Eigen::Vector3d& Xa = tmp[a].second;

        // (distancia, id_b)
        std::vector<std::pair<double, NodeId>> neigh;
        neigh.reserve(tmp.size());


        for (size_t b = 0; b < tmp.size(); ++b) {
            
            if (b == a) continue;
            const Eigen::Vector3d& Xb = tmp[b].second;
            double d = (Xa - Xb).norm();
            if (d > max_radius) continue;       // fuera de radio, ignoramos
            neigh.emplace_back(d, tmp[b].first);
        }

        // Ordenar por distancia
        std::sort(neigh.begin(), neigh.end(),
                [](const auto& p1, const auto& p2){
                    return p1.first < p2.first;
                });

        int used = 0;
        for (const auto& nd : neigh) {
            if (used >= K_neigh) break;

            double d   = nd.first;
            NodeId id_b = nd.second;

            // Normalizar el par para evitar duplicados (i<j)
            NodeId i = std::min(id_a, id_b);
            NodeId j = std::max(id_a, id_b);
            std::pair<NodeId,NodeId> key{i,j};

            // Si ya existe, saltamos
            if (g_rest_length.find(key) != g_rest_length.end())
                continue;

            ++used;

            ViscousEdge e;
            e.i = i;
            e.j = j;
            e.b_ij = std::exp(-(d*d) / (2.0 * sigma * sigma));
            g_viscous_edges.push_back(e);

            g_rest_length[key] = d; // longitud de referencia
        }
    }

    double avg_deg = 0.0;
    if (!g_nodes_rest.empty()) {
        avg_deg = 2.0 * (double)g_viscous_edges.size() / (double)g_nodes_rest.size();
    }
    std::cout << "[DEFORM] Nodos: " << g_nodes_rest.size()
            << "  edges viscosas: " << g_viscous_edges.size()
            << "  avg_deg: " << avg_deg << std::endl;
}


// ------------------- Visualización ----------------
cv::Mat drawDeformationGraphOnFrame(
    const cv::Mat& img,
    int frame_idx,
    const TrackManager& track_manager,
    const std::unordered_map<NodeId, NodeRest>& g_nodes_rest,
    const std::vector<ViscousEdge>& g_viscous_edges)
{
    cv::Mat out;
    if (img.channels() == 1)
        cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
    else
        img.copyTo(out);



    ///// 1) Dibujar nodos
    auto tracks = track_manager.getAllTracks();
    for (const auto& tr : tracks) {
        NodeId id = tr.id;
        if (!tr.hasObservation(frame_idx)) continue;

        // Buscar la observación en ese frame
        Eigen::Vector2d uv;
        bool found = false;
        for (const auto& obs : tr.observations) {
            if (obs.frame_idx == frame_idx) {
                uv = obs.uv;
                found = true;
                break;
            }
        }
        if (!found) continue;

        cv::Point2f pt( static_cast<float>(uv.x()),
                        static_cast<float>(uv.y()) );

        cv::circle(out, pt, 3, cv::Scalar(0, 255, 0), -1);
        cv::putText(out,
                    std::to_string(id),
                    pt + cv::Point2f(3.f, -3.f),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.3,
                    cv::Scalar(0, 255, 0),
                    1,
                    cv::LINE_AA);
    }


    ///////// 2) dibuja edges
    for (const auto& e : g_viscous_edges) {
        NodeId i = e.i;
        NodeId j = e.j;

        const PointTrack* tri = track_manager.getTrackById(i);
        const PointTrack* trj = track_manager.getTrackById(j);
        if (!tri || !trj) continue;

        if (!tri->hasObservation(frame_idx) || !trj->hasObservation(frame_idx))
            continue;

        // Buscar obs_i y obs_j en ese frame
        Eigen::Vector2d uvi, uvj;
        bool fi = false, fj = false;

        for (const auto& obs : tri->observations) {
            if (obs.frame_idx == frame_idx) {
                uvi = obs.uv;
                fi = true;
                break;
            }
        }
        for (const auto& obs : trj->observations) {
            if (obs.frame_idx == frame_idx) {
                uvj = obs.uv;
                fj = true;
                break;
            }
        }
        if (!fi || !fj) continue;

        cv::Point2f pi( static_cast<float>(uvi.x()),
                        static_cast<float>(uvi.y()) );
        cv::Point2f pj( static_cast<float>(uvj.x()),
                        static_cast<float>(uvj.y()) );

        cv::line(out, pi, pj, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    }


    return out;
}





