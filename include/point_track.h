// This file define the structure that represent a point
// observed in multiple images.

#pragma once
#include <vector>
#include <Eigen/Core>

struct PointObservation {
    int frame_idx;  // índice de la imagen
    Eigen::Vector2d uv;  // coordenadas 
};

struct PointTrack {
    int id;  // ID único del punto
    std::vector<PointObservation> observations;

    void addObservation(int frame_idx, const Eigen::Vector2d& uv) {
        observations.push_back({frame_idx, uv});
    }

    size_t numObservations() const {
        return observations.size();
    }

    // Verifica si el punto fue observado en un frame específico
    bool hasObservation(int frame_idx) const {
        for (const auto& obs : observations) {
            if (obs.frame_idx == frame_idx) return true;
        }
        return false;
    }

    // Devuelve la observación en un frame específico (si existe)
    std::optional<Eigen::Vector2d> getObservation(int frame_idx) const {
        for (const auto& obs : observations) {
            if (obs.frame_idx == frame_idx) return obs.uv;
        }
        return std::nullopt;  // no encontrada
    }
    
};
