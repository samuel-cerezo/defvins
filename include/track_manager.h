/*
This file will be responsible of all tracked points.
This module enables adding correspondences while holding ID coherence
*/

#pragma once
#include "point_track.h"
#include <map>

class TrackManager {
public:
    // Mapa de id de track a sus observaciones
    std::map<int, PointTrack> tracks;
    

    // Agrega una observación a un track existente
    void addObservation(int track_id, int frame_idx, const Eigen::Vector2d& uv);

    // Crea un nuevo track
    int createTrack(int frame_idx, const Eigen::Vector2d& uv);

    // Obtiene tracks con al menos n observaciones
    std::vector<PointTrack> getTracksWithMinObservations(int min_views) const;

    std::vector<PointTrack> getAllTracks() const;
    const PointTrack*       getTrackById(int track_id) const;


private:
    int next_id = 0;
    
};
