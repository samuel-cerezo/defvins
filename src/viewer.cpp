#include "viewer.h"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static cv::Mat renderFrameWithTracks(
    int frame_idx,
    const cv::Mat& img_gray,
    const TrackManager& track_manager,
    const std::unordered_map<NodeId, NodeRest>* nodes_rest,
    const std::vector<ViscousEdge>* viscous_edges)
{
    cv::Mat out;
    if (img_gray.channels() == 1)
        cv::cvtColor(img_gray, out, cv::COLOR_GRAY2BGR);
    else
        img_gray.copyTo(out);

    // --- 1) Dibujar todos los tracks visibles en este frame ---
    auto tracks = track_manager.getAllTracks();
    for (const auto& tr : tracks) {
        if (!tr.hasObservation(frame_idx)) continue;

        auto obs_opt = tr.getObservation(frame_idx);
        if (!obs_opt.has_value()) continue;

        const Eigen::Vector2d& uv = *obs_opt;
        cv::Point2f pt((float)uv.x(), (float)uv.y());

        cv::circle(out, pt, 2, cv::Scalar(0, 255, 0), -1); // punto verde

        // Opcional: ID
        // cv::putText(out, std::to_string(tr.id), pt + cv::Point2f(3,-3),
        //             cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,255,0), 1, cv::LINE_AA);
    }

    // --- 2) Dibujar grafo si nos pasan los datos ---
    if (nodes_rest && viscous_edges) {
        for (const auto& e : *viscous_edges) {
            const PointTrack* tri = track_manager.getTrackById(e.i);
            const PointTrack* trj = track_manager.getTrackById(e.j);
            if (!tri || !trj) continue;
            if (!tri->hasObservation(frame_idx) || !trj->hasObservation(frame_idx)) continue;

            auto obsi_opt = tri->getObservation(frame_idx);
            auto obsj_opt = trj->getObservation(frame_idx);
            if (!obsi_opt || !obsj_opt) continue;

            Eigen::Vector2d uvi = *obsi_opt;
            Eigen::Vector2d uvj = *obsj_opt;
            cv::Point2f pi((float)uvi.x(), (float)uvi.y());
            cv::Point2f pj((float)uvj.x(), (float)uvj.y());

            cv::line(out, pi, pj, cv::Scalar(255, 0, 0), 1, cv::LINE_AA); // arista azul
        }
    }

    return out;
}


void runTrackViewer(
    const std::vector<std::string>& cam0_image_names,
    TrackManager& track_manager,
    const Eigen::Matrix3d& K,
    const std::vector<std::string>& depth_image_names,
    double depth_scale)
{
    if (cam0_image_names.empty()) {
        std::cerr << "[VIEWER] No hay imágenes.\n";
        return;
    }

    int max_frame = static_cast<int>(cam0_image_names.size()) - 1;
    int current   = 0;
    bool playing  = false;
    const int delay_ms = 33; // ~30 fps

    cv::namedWindow("TrackViewer", cv::WINDOW_NORMAL);
    cv::resizeWindow("TrackViewer", 1280, 720);

    // Barra para moverse en la secuencia
    cv::createTrackbar("Frame", "TrackViewer", &current, max_frame);

    // Opcional: solo reconstruir grafo cada N frames para ir más ligero
    const int REBUILD_EVERY = 1; // pon 5 o 10 si quieres menos coste
    int last_graph_frame = -1000;

    while (true) {
        if (current < 0) current = 0;
        if (current > max_frame) current = max_frame;

        cv::Mat img = cv::imread(cam0_image_names[current], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "[VIEWER] No pude cargar imagen " << cam0_image_names[current] << "\n";
        } else {

            // 🔁 Rebuild del grafo para ESTE frame (solo visualización)
            if (current == 0 || std::abs(current - last_graph_frame) >= REBUILD_EVERY) {
                buildDeformationNodesFromTracks(track_manager, current, K, depth_image_names, depth_scale);
                last_graph_frame = current;
                std::cout << "[VIEWER] Rebuild grafo en frame " << current
                          << " | nodos=" << g_nodes_rest.size()
                          << " edges=" << g_viscous_edges.size() << "\n";
            }

            cv::Mat vis = renderFrameWithTracks(
                current,
                img,
                track_manager,
                &g_nodes_rest,
                &g_viscous_edges);

            cv::imshow("TrackViewer", vis);
        }

        int key = cv::waitKey(playing ? delay_ms : 0);

        if (key == 'q' || key == 27) {          // 'q' o ESC
            break;
        } else if (key == ' ') {                // espacio: play/pause
            playing = !playing;
        } else if (key == 'd' || key == 83) {   // 'd' o flecha derecha
            current = std::min(current + 1, max_frame);
            cv::setTrackbarPos("Frame", "TrackViewer", current);
        } else if (key == 'a' || key == 81) {   // 'a' o flecha izquierda
            current = std::max(current - 1, 0);
            cv::setTrackbarPos("Frame", "TrackViewer", current);
        } else if (key == 'f') {                // saltar +10
            current = std::min(current + 10, max_frame);
            cv::setTrackbarPos("Frame", "TrackViewer", current);
        } else if (key == 'b') {                // saltar -10
            current = std::max(current - 10, 0);
            cv::setTrackbarPos("Frame", "TrackViewer", current);
        }

        if (playing) {
            if (current < max_frame) {
                ++current;
                cv::setTrackbarPos("Frame", "TrackViewer", current);
            } else {
                playing = false;
            }
        }
    }

    cv::destroyWindow("TrackViewer");
}