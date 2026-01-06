/*
This file will be responsible of all tracked points.
This module enables adding correspondences while holding ID coherence
*/

#include "track_manager.h"

void TrackManager::addObservation(int track_id, int frame_idx, const Eigen::Vector2d& uv) {
    tracks[track_id].addObservation(frame_idx, uv);
}

int TrackManager::createTrack(int frame_idx, const Eigen::Vector2d& uv) {
    int id = next_id++;
    PointTrack pt;
    pt.id = id;
    pt.addObservation(frame_idx, uv);
    tracks[id] = pt;
    return id;
}

std::vector<PointTrack> TrackManager::getTracksWithMinObservations(int min_views) const {
    std::vector<PointTrack> result;
    for (const auto& kv : tracks) {
        if (kv.second.numObservations() >= static_cast<size_t>(min_views)) {
            result.push_back(kv.second);
        }
    }
    return result;
}

// devolver todos los tracks
std::vector<PointTrack> TrackManager::getAllTracks() const {
    std::vector<PointTrack> result;
    result.reserve(tracks.size());
    for (const auto& kv : tracks) {
        result.push_back(kv.second);
    }
    return result;
}

// obtener puntero constante a un track por id
const PointTrack* TrackManager::getTrackById(int track_id) const {
    auto it = tracks.find(track_id);
    if (it == tracks.end()) return nullptr;
    return &it->second;
}
